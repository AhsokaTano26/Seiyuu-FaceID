"""Stable client-facing errors, without model paths or exception tracebacks."""

from starlette.exceptions import HTTPException


class ApiError(HTTPException):
    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(status_code, detail=message)
        self.code = code

    def payload(self) -> dict[str, object]:
        return {"error": {"code": self.code, "message": self.detail}}
