"""Request policy for the internal SCRFD endpoint."""

import asyncio
import logging
import os
from ipaddress import IPv6Address, ip_address, ip_network

from starlette._utils import get_route_path
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from seiyuu_faceid.api.errors import ApiError

DETECT_PATH = "/api/v1/detect"
LOGGER = logging.getLogger(__name__)
_PRIVATE = tuple(
    ip_network(cidr) for cidr in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "fc00::/7")
)


def env_integer(name: str, default: int, maximum: int) -> int:
    raw = os.getenv(name) or str(default)
    if not raw.isascii() or not raw.isdecimal() or not 1 <= int(raw) <= maximum:
        raise ValueError(f"{name} must be an integer between 1 and {maximum}.")
    return int(raw)


class DetectionTimeout(ApiError, OSError):
    # Starlette also uses OSError to clean up partially spooled multipart files.
    def __init__(self):
        super().__init__(503, "detection_timeout", "The detection request timed out.")


class DetectionDeadline:
    """Start the budget before multipart parsing and restrict the new route's peers."""

    def __init__(self, app: ASGIApp, *, timeout_ms: int, client_cidr: str):
        self.app = app
        self.timeout = timeout_ms / 1000
        self.network = ip_network(client_cidr) if client_cidr else None
        if self.network and not any(
            self.network.version == net.version and self.network.subnet_of(net) for net in _PRIVATE
        ):
            raise ValueError("SEIYUU_FACEID_DETECT_CLIENT_CIDR must be a private subnet.")

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or get_route_path(scope) != DETECT_PATH:
            await self.app(scope, receive, send)
            return
        try:
            peer = ip_address((scope.get("client") or ("",))[0])
            if isinstance(peer, IPv6Address) and peer.ipv4_mapped:
                peer = peer.ipv4_mapped
        except ValueError:
            peer = None
        if not peer or not (peer.is_loopback or self.network and peer in self.network):
            error = ApiError(403, "internal_only", "This endpoint requires a local connection.")
            await JSONResponse(error.payload(), status_code=403)(scope, receive, send)
            return

        loop = asyncio.get_running_loop()
        started = loop.time()
        deadline = started + self.timeout
        scope.setdefault("state", {})["detect_deadline"] = deadline

        async def timed_receive() -> Message:
            try:
                async with asyncio.timeout_at(deadline):
                    return await receive()
            except TimeoutError as exc:
                raise DetectionTimeout() from exc

        async def log_status(message: Message) -> None:
            if message["type"] == "http.response.start":
                LOGGER.info(
                    "Face detection status=%d elapsed_ms=%.1f",
                    message["status"],
                    (loop.time() - started) * 1000,
                )
            await send(message)

        await self.app(scope, timed_receive, log_status)
