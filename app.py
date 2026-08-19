"""Local Streamlit interface for single and batch face identification."""

from __future__ import annotations

import base64
import json
import re
import sys
from hashlib import sha256
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from seiyuu_faceid.dataset.cleaner import build_aligned_dataset  # noqa: E402
from seiyuu_faceid.dataset.detections import (  # noqa: E402
    DetectionRecord,
    current_detection_records,
    load_detection_records,
    save_detection_records,
    scan_raw_detections,
)
from seiyuu_faceid.dataset.metadata import read_jsonl  # noqa: E402
from seiyuu_faceid.dataset.scanner import IMAGE_SUFFIXES, RawImage, scan_raw_dataset  # noqa: E402
from seiyuu_faceid.dataset.selections import (  # noqa: E402
    load_face_selections,
    save_face_selections,
)
from seiyuu_faceid.gallery.builder import build_gallery as build_embedding_gallery  # noqa: E402
from seiyuu_faceid.runtime.config import load_config  # noqa: E402
from seiyuu_faceid.runtime.engine import FaceIdEngine  # noqa: E402
from seiyuu_faceid.runtime.pipeline import serialize_result  # noqa: E402
from seiyuu_faceid.vision.detector import FaceDetection  # noqa: E402

CONFIG_PATH = ROOT / "configs/default.yaml"
MAX_UPLOADS = 50
MAX_DISPLAY_LONG_EDGE = 640
GROUP_REVIEW_RENDER_HEIGHT = 960
THUMBNAIL_LONG_EDGE = 240
GRID_COLUMNS = 4
GRID_PAGE_SIZE = 24
RAW_ROOT = ROOT / "data/raw"
FACE_SELECTIONS_PATH = ROOT / "data/review/raw_faces/face_selections.jsonl"
DETECTION_CACHE_PATH = ROOT / "data/runtime/detections.jsonl"

# OpenCV uses BGR. Selection and review must remain visually distinct.
FACE_BOX_COLORS = {
    "detected": (145, 145, 145),
    "selected": (190, 75, 200),
    "review": (0, 220, 255),
    "recognized": (0, 210, 0),
    "unknown": (0, 0, 255),
}
FACE_BOX_LABELS = {
    "selected": "已选",
    "review": "复核",
    "recognized": "已知",
    "unknown": "未知",
}

GROUP_REVIEW_KEYBOARD = st.components.v2.component(
    "seiyuu_faceid_group_review_keyboard",
    html='<span id="keyboard-hint" aria-live="polite"></span>',
    css="""
        :host { display: block; height: 0; overflow: hidden; }
        #keyboard-hint { display: none; }
    """,
    js="""
        export default function(component) {
            const { data, parentElement, setTriggerValue } = component;
            const hint = parentElement.querySelector('#keyboard-hint');
            if (hint) hint.textContent = '群像审核快捷键已启用';
            let digits = '';
            let timer = null;
            const send = (key) => setTriggerValue('key', { key, source: data.source });
            const flushDigits = () => {
                if (digits) send(digits);
                digits = '';
                if (timer) window.clearTimeout(timer);
                timer = null;
            };
            const onKeyDown = (event) => {
                if (
                    event.defaultPrevented || event.repeat || event.metaKey ||
                    event.ctrlKey || event.altKey
                ) return;
                const key = event.key;
                if (/^[0-9]$/.test(key)) {
                    event.preventDefault();
                    event.stopImmediatePropagation();
                    digits += key;
                    if (data.faceCount < 10 || digits.length >= 2) {
                        flushDigits();
                    } else {
                        if (timer) window.clearTimeout(timer);
                        timer = window.setTimeout(flushDigits, 350);
                    }
                    return;
                }
                const normalized = key.toLowerCase();
                if (key === 'Enter' || normalized === 'a' || normalized === 'y') {
                    event.preventDefault();
                    event.stopImmediatePropagation();
                    digits = '';
                    if (timer) window.clearTimeout(timer);
                    timer = null;
                    send(key);
                }
            };
            window.addEventListener('keydown', onKeyDown, true);
            return () => {
                window.removeEventListener('keydown', onKeyDown, true);
                if (timer) window.clearTimeout(timer);
            };
        }
    """,
)


@st.cache_resource(show_spinner="正在加载本地 ONNX 模型和 Gallery…")
def load_engine(config_path: str) -> FaceIdEngine:
    """Cache one CPU model instance for the Streamlit process."""
    return FaceIdEngine(load_config(Path(config_path)), allow_missing_gallery=True)


def decode_image(data: bytes) -> np.ndarray | None:
    """Decode an uploaded image from memory without writing it to disk."""
    return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)


def parse_group_face_input(value: str, detected_face_count: int) -> list[int]:
    """Parse one-based group-review IDs without relying on hot-reloaded modules."""
    normalized = value.strip().casefold()
    if not normalized:
        return []
    if normalized in {"a", "all", "全部", "全选"}:
        return list(range(detected_face_count))
    tokens = [token for token in re.split(r"[\s,，]+", normalized) if token]
    if not tokens or any(not token.isdigit() for token in tokens):
        raise ValueError("请使用如 1,3 的 Face ID，或输入 A 表示全部。")
    indexes = [int(token) - 1 for token in tokens]
    if any(index < 0 or index >= detected_face_count for index in indexes):
        raise ValueError(f"Face ID 必须在 1 到 {detected_face_count} 之间。")
    if len(set(indexes)) != len(indexes):
        raise ValueError("Face ID 不能重复。")
    return sorted(indexes)


def annotate_detections(
    image_bgr: np.ndarray,
    detections: list[FaceDetection],
    selected_indexes: set[int] | None,
    recognition_statuses: dict[int, str] | None = None,
    recognition_names: dict[int, str] | None = None,
    coordinate_scale: float = 1.0,
) -> np.ndarray:
    """Draw neutral detection boxes and color only selected/result faces."""
    annotated = image_bgr.copy()
    selected = selected_indexes or set()
    statuses = recognition_statuses or {}
    names = recognition_names or {}
    for index, detection in enumerate(detections):
        number = index + 1
        x1, y1, x2, y2 = np.round(detection.bbox * coordinate_scale).astype(int)
        status = statuses.get(index)
        if index in selected:
            state = "selected"
        elif status == "recognized":
            state = "recognized"
        elif status == "unknown":
            state = "unknown"
        elif status == "review":
            state = "review"
        else:
            state = "detected"
        color = FACE_BOX_COLORS[state]
        thickness = 3 if state != "detected" else 2
        state_label = FACE_BOX_LABELS.get(state)
        labels = [f"Face {number}"]
        if name := names.get(index):
            labels.append(name)
        elif state_label:
            labels.append(state_label)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
        text_metrics = [
            cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1) for label in labels
        ]
        label_width = max(width for (width, _), _ in text_metrics)
        label_height = sum(height + baseline for (_, height), baseline in text_metrics) + 8
        label_top = max(0, y1 - label_height - 6)
        cv2.rectangle(
            annotated,
            (x1, label_top),
            (x1 + label_width + 8, label_top + label_height + 4),
            color,
            cv2.FILLED,
        )
        text_y = label_top + 4
        for label, ((_, height), baseline) in zip(labels, text_metrics, strict=True):
            text_y += height
            cv2.putText(
                annotated,
                label,
                (x1 + 4, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
            text_y += baseline + 2
    return annotated


def resize_for_display(
    image_bgr: np.ndarray, max_long_edge: int = MAX_DISPLAY_LONG_EDGE
) -> np.ndarray:
    """Limit preview pixels while keeping inference on the original upload."""
    height, width = image_bgr.shape[:2]
    long_edge = max(height, width)
    if long_edge <= max_long_edge:
        return image_bgr
    scale = max_long_edge / long_edge
    return cv2.resize(
        image_bgr,
        (round(width * scale), round(height * scale)),
        interpolation=cv2.INTER_AREA,
    )


def resize_for_height(image_bgr: np.ndarray, max_height: int) -> np.ndarray:
    """Constrain only display height; keep a landscape image's useful width."""
    height, width = image_bgr.shape[:2]
    if height <= max_height:
        return image_bgr
    scale = max_height / height
    return cv2.resize(
        image_bgr,
        (round(width * scale), max_height),
        interpolation=cv2.INTER_AREA,
    )


def detection_preview(
    image_bgr: np.ndarray,
    detections: list[FaceDetection],
    selected_indexes: set[int] | None,
    recognition_statuses: dict[int, str] | None = None,
    recognition_names: dict[int, str] | None = None,
    max_long_edge: int = MAX_DISPLAY_LONG_EDGE,
    max_height: int | None = None,
) -> np.ndarray:
    """Resize first, then draw boxes so their final display width stays readable."""
    display = (
        resize_for_height(image_bgr, max_height)
        if max_height is not None
        else resize_for_display(image_bgr, max_long_edge=max_long_edge)
    )
    scale = display.shape[1] / image_bgr.shape[1]
    return annotate_detections(
        display,
        detections,
        selected_indexes,
        recognition_statuses,
        recognition_names,
        coordinate_scale=scale,
    )


def render_group_review_image(image_bgr: np.ndarray) -> None:
    """Render a centred face-review image that fits the browser viewport."""
    success, encoded = cv2.imencode(
        ".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92]
    )
    if not success:
        st.error("无法生成审核预览。")
        return
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    st.markdown(
        f"""
        <style>
          .group-review-image {{
            align-items: center;
            display: flex;
            height: calc(100vh - 165px);
            justify-content: center;
            overflow: hidden;
          }}
          .group-review-image img {{
            height: auto;
            max-height: 100%;
            max-width: 100%;
            object-fit: contain;
            width: auto;
          }}
        </style>
        <div class="group-review-image">
          <img alt="群像审核预览" src="data:image/jpeg;base64,{payload}" />
        </div>
        """,
        unsafe_allow_html=True,
    )


def prepare_upload(
    engine: FaceIdEngine, filename: str, content: bytes
) -> tuple[np.ndarray | None, list[FaceDetection], dict[str, Any] | None]:
    """Decode an upload and detect faces before the user chooses one to recognize."""
    image = decode_image(content)
    if image is None:
        return None, [], {"image": filename, "error": "无法解码图片"}
    try:
        detections = engine.detect_array(image)
    except Exception as error:
        return None, [], {"image": filename, "error": str(error)}
    return image, detections, None


def identity_directories() -> list[Path]:
    if not RAW_ROOT.is_dir():
        return []
    return sorted(path for path in RAW_ROOT.iterdir() if path.is_dir())


def image_paths(identity_directory: Path) -> list[Path]:
    return sorted(
        path
        for path in identity_directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def query_image_paths(directory: Path) -> list[Path]:
    """Return query images recursively without modifying the user-selected directory."""
    if not directory.is_dir():
        raise FileNotFoundError(f"目录不存在：{directory}")
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


@st.cache_data(show_spinner=False)
def cached_thumbnail(path: str, size: int, mtime_ns: int) -> np.ndarray | None:
    """Decode a small display thumbnail without changing the source image."""
    del size, mtime_ns
    image = cv2.imread(path)
    if image is None:
        return None
    return cv2.cvtColor(resize_for_display(image, THUMBNAIL_LONG_EDGE), cv2.COLOR_BGR2RGB)


def image_thumbnail(path: Path) -> np.ndarray | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return cached_thumbnail(str(path), stat.st_size, stat.st_mtime_ns)


def paginated_items[T](
    items: list[T], page_key: str, *, page_size: int = GRID_PAGE_SIZE
) -> tuple[list[T], int, int]:
    """Return one stable page; render controls after the caller's image grid."""
    if not items:
        return [], 0, 0
    page_count = (len(items) + page_size - 1) // page_size
    page = int(st.session_state.get(page_key, 0))
    page = min(max(page, 0), page_count - 1)
    st.session_state[page_key] = page
    start = page * page_size
    return items[start : start + page_size], page, page_count


def render_pagination_controls(page_key: str, page: int, page_count: int, total: int) -> None:
    """Place page navigation directly below the images it changes."""
    if page_count < 1:
        return
    previous_column, label_column, next_column = st.columns((1, 2, 1))
    with previous_column:
        if st.button("← 上一页", key=f"{page_key}-previous", disabled=page == 0):
            st.session_state[page_key] = page - 1
            st.rerun()
    with next_column:
        if st.button("下一页 →", key=f"{page_key}-next", disabled=page == page_count - 1):
            st.session_state[page_key] = page + 1
            st.rerun()
    with label_column:
        st.caption(f"第 {page + 1} / {page_count} 页 · 共 {total} 项")


def render_raw_thumbnail_grid(directory: Path, paths: list[Path]) -> None:
    """Show raw images as an orderly paginated image-host style grid."""
    selected_key = f"selected-raw-image-{sha256(str(directory).encode()).hexdigest()[:12]}"
    selected = st.session_state.get(selected_key)
    relative_paths = {path.relative_to(directory).as_posix(): path for path in paths}
    if selected not in relative_paths:
        selected = paths[0].relative_to(directory).as_posix() if paths else None
        st.session_state[selected_key] = selected
    page_key = f"raw-grid-page-{sha256(str(directory).encode()).hexdigest()[:12]}"
    page_paths, page, page_count = paginated_items(
        paths,
        page_key,
    )
    for offset in range(0, len(page_paths), GRID_COLUMNS):
        columns = st.columns(GRID_COLUMNS)
        for column, path in zip(columns, page_paths[offset : offset + GRID_COLUMNS], strict=False):
            relative_path = path.relative_to(directory).as_posix()
            with column:
                with st.container(border=True):
                    thumbnail = image_thumbnail(path)
                    if thumbnail is not None:
                        st.image(thumbnail, width="stretch")
                    else:
                        st.warning("无法预览")
                    st.caption(relative_path)
                    if st.button(
                        "查看详情",
                        key=f"raw-card-{sha256(str(path).encode()).hexdigest()[:16]}",
                        use_container_width=True,
                    ):
                        st.session_state[selected_key] = relative_path
                        st.session_state["raw-detail-image"] = str(path)
    render_pagination_controls(page_key, page, page_count, len(paths))


def raw_file_signature() -> tuple[tuple[str, int, int], ...]:
    """Return a cheap change token without reading each image's contents."""
    files = [
        path
        for directory in identity_directories()
        for path in image_paths(directory)
    ]
    return tuple(
        (
            path.relative_to(RAW_ROOT).as_posix(),
            path.stat().st_size,
            path.stat().st_mtime_ns,
        )
        for path in files
    )


@st.cache_data(show_spinner=False)
def cached_raw_sources(
    raw_root: str, signature: tuple[tuple[str, int, int], ...]
) -> list[RawImage]:
    """Hash raw files once per unchanged dataset, not once per review answer."""
    del signature
    return scan_raw_dataset(Path(raw_root))


@st.cache_data(show_spinner=False)
def cached_detection_manifest(
    cache_path: str, cache_size: int, cache_mtime_ns: int
) -> dict[str, DetectionRecord]:
    """Parse the immutable detection cache once until the scan output changes."""
    del cache_size, cache_mtime_ns
    return load_detection_records(Path(cache_path))


def is_valid_identity(value: str) -> bool:
    candidate = Path(value)
    return bool(value) and candidate.name == value and value not in {".", ".."}


def raw_image_and_detections(
    engine: FaceIdEngine, image_path: Path
) -> tuple[np.ndarray | None, list[FaceDetection]]:
    """Load one raw image and retain its detections for this browser session."""
    try:
        stat = image_path.stat()
    except OSError:
        return None, []
    cache_key = f"{image_path}:{stat.st_mtime_ns}:{stat.st_size}"
    cache = st.session_state.setdefault("raw_detection_cache", {})
    image = cv2.imread(str(image_path))
    if image is None:
        return None, []
    detections = cache.get(cache_key)
    if detections is None:
        detections = engine.detect_array(image)
        cache[cache_key] = detections
    return image, detections


def raw_recognition_results(
    engine: FaceIdEngine,
    image_path: Path,
    image: np.ndarray,
    detections: list[FaceDetection],
) -> dict[int, Any]:
    """Recognize cached raw detections once per raw image and Gallery version."""
    if engine.gallery is None:
        return {}
    image_stat = image_path.stat()
    vectors_stat = engine.config.gallery_vectors_path.stat()
    metadata_stat = engine.config.gallery_metadata_path.stat()
    cache_key = (
        f"{image_path}:{image_stat.st_mtime_ns}:{image_stat.st_size}:"
        f"{vectors_stat.st_mtime_ns}:{vectors_stat.st_size}:"
        f"{metadata_stat.st_mtime_ns}:{metadata_stat.st_size}"
    )
    cache = st.session_state.setdefault("raw_recognition_cache", {})
    results = cache.get(cache_key)
    if results is None:
        results = {
            index: engine.recognize_detection(image, detection)
            for index, detection in enumerate(detections)
        }
        cache[cache_key] = results
    return results


def reviewed_face_selections() -> dict[str, list[int]]:
    """Load once per session; incomplete command-line templates are harmless here."""
    key = "reviewed_face_selections"
    if key not in st.session_state:
        st.session_state[key] = (
            load_face_selections(FACE_SELECTIONS_PATH, allow_incomplete=True)
            if FACE_SELECTIONS_PATH.is_file()
            else {}
        )
    return st.session_state[key]


def set_face_selection(source_image: str, face_indexes: list[int]) -> None:
    selections = reviewed_face_selections()
    selections[source_image] = sorted(set(face_indexes))
    save_face_selections(FACE_SELECTIONS_PATH, selections)


def render_raw_image_review(engine: FaceIdEngine, image_path: Path) -> None:
    image, detections = raw_image_and_detections(engine, image_path)
    source_image = image_path.relative_to(RAW_ROOT).as_posix()
    if image is None:
        st.error("无法解码这张原始图片。")
        return
    selected_indexes = reviewed_face_selections().get(source_image, [])
    selected_indexes = [
        index
        for index in selected_indexes
        if isinstance(index, int) and 0 <= index < len(detections)
    ]
    recognition_results: dict[int, Any] = {}
    if engine.gallery is not None and detections:
        try:
            recognition_results = raw_recognition_results(engine, image_path, image, detections)
        except Exception as error:
            st.error(f"自动识别失败：{error}")
    statuses = {
        index: result.decision.status for index, result in recognition_results.items()
    }
    names = {
        index: result.decision.identity or "Unknown"
        for index, result in recognition_results.items()
    }
    st.image(
        cv2.cvtColor(
            detection_preview(
                image,
                detections,
                set(selected_indexes),
                recognition_statuses=statuses,
                recognition_names=names,
            ),
            cv2.COLOR_BGR2RGB,
        ),
        width="content",
        caption=f"{source_image}：检测到 {len(detections)} 张人脸",
    )
    if recognition_results:
        st.caption("灰色＝未选；紫色＝已选；黄色＝复核；绿色＝已知；红色＝Unknown。")
        st.dataframe(
            result_rows_from_record(
                {
                    "faces": [
                        serialize_result(recognition_results[index])
                        for index in sorted(recognition_results)
                    ]
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
    elif engine.gallery is None:
        st.caption("尚未建立 Gallery：当前仅显示检测结果。")
    if len(detections) > 1:
        st.info("请使用下方“审核群像”逐题选择属于该身份的所有人脸。")
    elif len(detections) == 1:
        st.success("单脸图片会在建库时自动使用。")
    else:
        st.warning("未检测到人脸；建库时会把它记录为拒绝项。")


@st.dialog("图片详情与自动识别", width="large")
def render_raw_image_dialog(engine: FaceIdEngine, image_path: Path) -> None:
    """Open raw image details immediately instead of placing them below a grid."""
    render_raw_image_review(engine, image_path)
    close_key = f"close-raw-detail-{sha256(str(image_path).encode()).hexdigest()[:12]}"
    if st.button("关闭", key=close_key):
        st.session_state.pop("raw-detail-image", None)
        st.rerun()


def render_query_image_review(engine: FaceIdEngine, image_path: Path) -> None:
    """Render a query image with the same annotated recognition detail as raw images."""
    image, detections = raw_image_and_detections(engine, image_path)
    if image is None:
        st.error("无法解码这张图片。")
        return
    recognition_results: dict[int, Any] = {}
    if engine.gallery is not None and detections:
        try:
            recognition_results = raw_recognition_results(engine, image_path, image, detections)
        except Exception as error:
            st.error(f"自动识别失败：{error}")
    statuses = {
        index: result.decision.status for index, result in recognition_results.items()
    }
    names = {
        index: result.decision.identity or "Unknown"
        for index, result in recognition_results.items()
    }
    st.image(
        cv2.cvtColor(
            detection_preview(
                image,
                detections,
                set(),
                recognition_statuses=statuses,
                recognition_names=names,
            ),
            cv2.COLOR_BGR2RGB,
        ),
        width="content",
        caption=f"{image_path.name}：检测到 {len(detections)} 张人脸",
    )
    if recognition_results:
        st.caption("灰色＝未选择；黄色＝复核；绿色＝已知；红色＝Unknown。")
        st.dataframe(
            result_rows_from_record(
                {
                    "faces": [
                        serialize_result(recognition_results[index])
                        for index in sorted(recognition_results)
                    ]
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
    elif engine.gallery is None:
        st.warning("尚未建立 Gallery，无法识别。")
    else:
        st.info("未检测到人脸。")


@st.dialog("批量识别详情", width="large")
def render_query_image_dialog(engine: FaceIdEngine, image_path: Path) -> None:
    """Show one batch-result image at a readable size with annotated face results."""
    render_query_image_review(engine, image_path)
    close_key = f"close-query-detail-{sha256(str(image_path).encode()).hexdigest()[:12]}"
    if st.button("关闭", key=close_key):
        st.session_state.pop("folder-detail-image", None)
        st.rerun()


def render_raw_upload(selected_identity: str | None) -> None:
    with st.expander("通过网页添加原始图片", expanded=False):
        new_identity = st.text_input("新身份文件夹名称", key="new-raw-identity")
        if st.button("新建身份文件夹", key="create-raw-identity"):
            if not is_valid_identity(new_identity):
                st.error("文件夹名称不能为空，且不能包含路径分隔符。")
            else:
                (RAW_ROOT / new_identity).mkdir(parents=True, exist_ok=True)
                st.session_state["selected_raw_identity"] = new_identity
                st.rerun()
        if selected_identity is None:
            st.caption("先新建或选择一个身份文件夹，再上传图片。")
            return
        uploads = st.file_uploader(
            f"上传到 raw/{selected_identity}/",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            accept_multiple_files=True,
            key=f"raw-upload-{selected_identity}",
        )
        if uploads and st.button("保存上传图片", key=f"save-raw-upload-{selected_identity}"):
            target_directory = RAW_ROOT / selected_identity
            target_directory.mkdir(parents=True, exist_ok=True)
            saved = 0
            skipped: list[str] = []
            for upload in uploads:
                filename = Path(upload.name).name
                target = target_directory / filename
                if not filename or target.exists():
                    skipped.append(upload.name)
                    continue
                target.write_bytes(upload.getvalue())
                saved += 1
            st.session_state.pop("raw_detection_cache", None)
            st.session_state.pop("raw_recognition_cache", None)
            if saved:
                st.success(f"已保存 {saved} 张图片。")
            if skipped:
                st.warning("以下文件已存在，未覆盖：" + "、".join(skipped))


def raw_review_items(
    sources: list[RawImage], detection_records: dict[str, DetectionRecord]
) -> list[tuple[RawImage, list[FaceDetection]]]:
    """Find cached group-photo detections that still need a human selection."""
    selections = reviewed_face_selections()
    pending: list[tuple[RawImage, list[FaceDetection]]] = []
    for source in sources:
        record = detection_records[source.relative_path.as_posix()]
        detections = list(record.detections)
        if record.error is not None or len(detections) < 2:
            continue
        selected_indexes = selections.get(source.relative_path.as_posix())
        if selected_indexes is None or any(
            not isinstance(index, int) or not 0 <= index < len(detections)
            for index in selected_indexes
        ):
            pending.append((source, detections))
    return pending


def current_raw_detection_cache() -> tuple[list[RawImage], dict[str, DetectionRecord] | None]:
    """Load detections only when they match every current raw source image."""
    sources = cached_raw_sources(str(RAW_ROOT), raw_file_signature())
    if not DETECTION_CACHE_PATH.is_file():
        return sources, None
    try:
        stat = DETECTION_CACHE_PATH.stat()
        records = cached_detection_manifest(
            str(DETECTION_CACHE_PATH), stat.st_size, stat.st_mtime_ns
        )
    except ValueError:
        return sources, None
    return sources, current_detection_records(sources, records)


def scan_all_raw_images(engine: FaceIdEngine) -> tuple[list[RawImage], dict[str, DetectionRecord]]:
    """Run detector once for all raw images and persist the reusable results."""
    progress_bar = st.progress(0, text="准备扫描原始图片…")

    def report(number: int, total: int, source: RawImage) -> None:
        fraction = number / total if total else 1.0
        progress_bar.progress(fraction, text=f"扫描 {number}/{total}：{source.relative_path}")

    sources, records = scan_raw_detections(RAW_ROOT, engine.detector, progress=report)
    progress_bar.empty()
    save_detection_records(DETECTION_CACHE_PATH, records)
    return sources, records


def rebuild_gallery(
    engine: FaceIdEngine, detection_records: dict[str, DetectionRecord]
) -> tuple[int, int, int]:
    """Run the same local alignment and Gallery build stages as the command-line tool."""
    selections = reviewed_face_selections()
    accepted, rejected = build_aligned_dataset(
        RAW_ROOT,
        ROOT / "data/aligned",
        ROOT / "data/rejected",
        detector=None,
        min_face_size=engine.config.min_face_size,
        min_sharpness=engine.config.min_sharpness,
        face_selections=selections,
        detection_records=detection_records,
    )
    if not accepted:
        return accepted, rejected, 0
    records = read_jsonl(ROOT / "data/aligned/metadata.jsonl")
    gallery = build_embedding_gallery(records, ROOT / "data/aligned", engine.recognizer)
    gallery.save(engine.config.gallery_vectors_path, engine.config.gallery_metadata_path)
    engine.gallery = gallery
    return accepted, rejected, len(gallery.prototype_identities)


def render_group_review(
    engine: FaceIdEngine,
    source: RawImage,
    detections: list[FaceDetection],
) -> None:
    """Present one group photo at a time and accept any number of selected faces."""
    source_image = source.relative_path.as_posix()
    image = cv2.imread(str(source.path))
    if image is None:
        st.error("无法解码这张原始图片；请修复或移除文件后重新扫描。")
        return
    current = group_review_draft(
        source_image, reviewed_face_selections().get(source_image, [])
    )
    keyboard_event = GROUP_REVIEW_KEYBOARD(
        key=f"group-review-keys-{sha256(source_image.encode()).hexdigest()[:12]}",
        data={"source": source_image, "faceCount": len(detections)},
        on_key_change=lambda: None,
        width=1,
        height=1,
    )
    key_event = keyboard_event.get("key")
    pressed_key = (
        key_event.get("key")
        if isinstance(key_event, dict) and key_event.get("source") == source_image
        else None
    )
    if isinstance(pressed_key, str):
        normalized_key = pressed_key.casefold()
        if pressed_key.isdigit():
            face_index = int(pressed_key) - 1
            if 0 <= face_index < len(detections):
                updated = set(current)
                if face_index in updated:
                    updated.remove(face_index)
                else:
                    updated.add(face_index)
                st.session_state["group_review_draft_indexes"] = sorted(updated)
                st.rerun()
        elif normalized_key == "a":
            st.session_state["group_review_draft_indexes"] = list(range(len(detections)))
            st.rerun()
        elif normalized_key == "y":
            # Suggestions are calculated below. Defer handling until they are available.
            st.session_state["group_review_apply_suggestion"] = source_image
        elif pressed_key == "Enter":
            set_face_selection(source_image, current)
            reset_group_review_draft()
            st.rerun()
    recognition_results: dict[int, Any] = {}
    if engine.gallery is not None:
        try:
            recognition_results = raw_recognition_results(engine, source.path, image, detections)
        except Exception as error:
            st.warning(f"自动建议暂不可用：{error}")
    statuses = {
        index: result.decision.status for index, result in recognition_results.items()
    }
    names = {
        index: result.decision.identity or "Unknown"
        for index, result in recognition_results.items()
    }
    suggested_indexes = [
        index
        for index, result in recognition_results.items()
        if result.decision.status == "recognized" and result.decision.identity == source.identity
    ]
    if st.session_state.pop("group_review_apply_suggestion", None) == source_image:
        if suggested_indexes:
            set_face_selection(source_image, suggested_indexes)
            reset_group_review_draft()
            st.rerun()
        st.warning("当前图片没有可信建议，请改用数字键选择。")
    preview = detection_preview(
        image,
        detections,
        set(current),
        recognition_statuses=statuses,
        recognition_names=names,
        max_height=GROUP_REVIEW_RENDER_HEIGHT,
    )
    image_column, control_column = st.columns((6, 2), gap="large", vertical_alignment="center")
    with image_column:
        render_group_review_image(preview)
    with control_column:
        with st.container(border=True):
            st.subheader("群像审核")
            st.caption(source.path.name)
            st.caption(f"身份：{source.identity} · 检测到 {len(detections)} 张人脸")
            st.divider()
            selected_numbers = ", ".join(str(index + 1) for index in current) or "无"
            st.markdown(f"**当前选择：** {selected_numbers}")
            st.caption("数字：切换 Face ID · A：全选 · Y：采用建议 · Enter：确认下一张")
            st.caption("颜色：灰色＝未选 · 紫色＝已选 · 黄色＝复核 · 绿色＝已知 · 红色＝未知")
            if suggested_indexes:
                numbers = ", ".join(str(index + 1) for index in suggested_indexes)
                st.success(f"可信建议：Face {numbers}")
            else:
                st.info("当前没有可信建议，请用数字键选择。")
            st.divider()
            confirmed = st.button("Enter · 确认并下一张", type="primary", use_container_width=True)
            use_suggestions = st.button(
                "Y · 采用可信建议",
                disabled=not suggested_indexes,
                use_container_width=True,
            )
            select_all = st.button("A · 全选", use_container_width=True)
            later = st.button("稍后再审核", use_container_width=True, key=f"later-{source_image}")
    if confirmed:
        set_face_selection(source_image, current)
        reset_group_review_draft()
        st.rerun()
    if use_suggestions:
        set_face_selection(source_image, suggested_indexes)
        reset_group_review_draft()
        st.rerun()
    if select_all:
        st.session_state["group_review_draft_indexes"] = list(range(len(detections)))
        st.rerun()
    if later:
        reset_group_review_draft()
        st.session_state["group_review_active"] = False
        st.session_state["group_review_started"] = False
        st.rerun()


def run_gallery_build(engine: FaceIdEngine, detection_records: dict[str, DetectionRecord]) -> None:
    """Build aligned faces and Gallery after the review queue is complete."""
    st.session_state["group_review_started"] = False
    with st.spinner("正在对齐人脸、提取向量并建立本地 Gallery…"):
        try:
            accepted, rejected, identities = rebuild_gallery(engine, detection_records)
        except Exception as error:
            st.error(f"建库失败：{error}")
        else:
            if accepted:
                st.success(f"建库完成：{accepted} 张脸，{identities} 个身份；拒绝 {rejected} 项。")
            else:
                st.warning(f"没有可入库的人脸；拒绝 {rejected} 项。")


def group_review_draft(source_image: str, existing: list[int]) -> list[int]:
    """Keep number-key toggles local until the user confirms this photo."""
    if st.session_state.get("group_review_draft_source") != source_image:
        st.session_state["group_review_draft_source"] = source_image
        st.session_state["group_review_draft_indexes"] = sorted(set(existing))
    return list(st.session_state["group_review_draft_indexes"])


def reset_group_review_draft() -> None:
    st.session_state.pop("group_review_draft_source", None)
    st.session_state.pop("group_review_draft_indexes", None)


def render_gallery_build(engine: FaceIdEngine) -> None:
    sources, detection_records = current_raw_detection_cache()
    if detection_records is None:
        st.divider()
        st.subheader("建库")
        st.caption("先完整扫描一次，再逐题审核群像；审核与建库都会复用扫描结果。")
        st.info("尚未扫描，或 raw 图片已变更。扫描仅检测人脸，不会生成对齐脸或修改原图。")
        if st.button("扫描全部原始图片", type="primary", key="scan-all-raw-images"):
            with st.spinner("正在运行人脸检测…"):
                sources, detection_records = scan_all_raw_images(engine)
            group_count = sum(
                1 for record in detection_records.values() if len(record.detections) > 1
            )
            st.success(f"扫描完成：{len(sources)} 张图片，其中 {group_count} 张群像待审核。")
            st.rerun()
        return
    group_count = sum(1 for record in detection_records.values() if len(record.detections) > 1)
    pending = raw_review_items(sources, detection_records)
    if st.session_state.get("group_review_active"):
        if pending:
            render_group_review(engine, *pending[0])
        else:
            st.session_state["group_review_active"] = False
            st.session_state["build_after_group_review"] = True
            st.rerun()
        return
    st.divider()
    st.subheader("建库")
    st.caption("先完整扫描一次，再逐题审核群像；审核与建库都会复用扫描结果。")
    st.caption(
        f"扫描缓存有效：{len(sources)} 张图片；群像 {group_count} 张，待审核 {len(pending)} 张。"
    )
    if not st.session_state.get("group_review_started"):
        if pending and st.button("开始逐题审核群像", type="primary", key="begin-group-review"):
            st.session_state["group_review_started"] = True
            st.session_state["group_review_active"] = True
            st.rerun()
        elif not pending:
            st.success("没有待审核的群像，可以开始建库。")
    if pending:
        st.warning(f"有 {len(pending)} 张群像尚未审核。")
        if st.button("继续逐题审核", type="primary", key="continue-group-review"):
            st.session_state["group_review_active"] = True
            st.rerun()
        return
    st.success("所有群像均已审核，可以开始建库。")
    if st.session_state.pop("build_after_group_review", False):
        run_gallery_build(engine, detection_records)
    elif st.button("开始重建 Gallery", type="primary", key="confirm-gallery-build"):
        run_gallery_build(engine, detection_records)


def render_gallery_management(engine: FaceIdEngine) -> None:
    st.header("图库管理")
    st.caption("浏览或上传 data/raw 中的身份图片。扫描、群像审核和建库位于“建库与审核”页面。")
    directories = identity_directories()
    if directories:
        directory_by_name = {directory.name: directory for directory in directories}
        names = list(directory_by_name)
        current = st.session_state.get("selected_raw_identity")
        if current not in directory_by_name:
            current = names[0]
        selected_identity = st.selectbox(
            "身份文件夹",
            options=names,
            index=names.index(current),
            format_func=lambda name: f"{name}（{len(image_paths(directory_by_name[name]))} 张）",
            key="selected-raw-identity",
        )
    else:
        st.info("data/raw 还没有身份文件夹。请新建一个，或手动创建后刷新页面。")
        selected_identity = None
    render_raw_upload(selected_identity)
    if selected_identity is not None:
        directory = RAW_ROOT / selected_identity
        paths = image_paths(directory)
        st.subheader(f"raw/{selected_identity}")
        if paths:
            render_raw_thumbnail_grid(directory, paths)
            detail_path = st.session_state.get("raw-detail-image")
            if isinstance(detail_path, str):
                candidate = Path(detail_path)
                if candidate.is_file() and candidate.is_relative_to(directory):
                    render_raw_image_dialog(engine, candidate)
                else:
                    st.session_state.pop("raw-detail-image", None)
        else:
            st.info("该文件夹暂时没有支持的图片格式。")


def recognize_query_folder(
    engine: FaceIdEngine, directory: Path, paths: list[Path]
) -> list[dict[str, Any]]:
    """Recognize every supported image in a local directory without writing to it."""
    records: list[dict[str, Any]] = []
    progress = st.progress(0, text="准备批量识别…")
    for number, path in enumerate(paths, start=1):
        record: dict[str, Any] = {
            "image": str(path),
            "relative_path": path.relative_to(directory).as_posix(),
            "faces": [],
        }
        image = cv2.imread(str(path))
        if image is None:
            record["error"] = "无法解码图片"
        else:
            try:
                results = engine.recognize_array(image)
            except Exception as error:
                record["error"] = str(error)
            else:
                record["detected_face_count"] = len(results)
                record["faces"] = [serialize_result(result) for result in results]
        records.append(record)
        progress.progress(number / len(paths), text=f"识别 {number}/{len(paths)}：{path.name}")
    progress.empty()
    return records


def render_folder_result_card(record: dict[str, Any]) -> None:
    """Render one query image and all of its face decisions in a compact card."""
    path_value = record.get("image")
    path = Path(path_value) if isinstance(path_value, str) else None
    if path is not None:
        thumbnail = image_thumbnail(path)
        if thumbnail is not None:
            st.image(thumbnail, width="stretch")
    st.caption(str(record.get("relative_path") or record.get("image") or "图片"))
    error = record.get("error")
    if isinstance(error, str):
        st.error(error)
    else:
        faces = record.get("faces")
        if not isinstance(faces, list) or not faces:
            st.info("未检测到人脸")
        else:
            for number, face in enumerate(faces, start=1):
                if not isinstance(face, dict):
                    continue
                status = face.get("status")
                identity = face.get("identity") or "Unknown"
                candidates = face.get("candidates")
                top_score = None
                if isinstance(candidates, list) and candidates and isinstance(candidates[0], dict):
                    top_score = candidates[0].get("score")
                detail = f"Face {number} · {identity} · {status}"
                if isinstance(top_score, (int, float)):
                    detail += f" · {top_score:.3f}"
                if status == "recognized":
                    st.success(detail)
                elif status == "unknown":
                    st.error(detail)
                else:
                    st.warning(detail)
    if path is not None and path.is_file():
        if st.button(
            "查看详情",
            key=f"query-card-{sha256(str(path).encode()).hexdigest()[:16]}",
            width="stretch",
        ):
            st.session_state["folder-detail-image"] = str(path)


def render_folder_recognition(engine: FaceIdEngine) -> None:
    """Run explicit, paginated local-folder recognition for large query sets."""
    st.subheader("本地文件夹批量识别")
    st.caption(
        "只读取目标目录。点击后会递归识别目录及所有子文件夹中的支持格式图片，"
        "并以缩略图网格显示结果。"
    )
    default_directory = str(ROOT / "data/query")
    directory_value = st.text_input(
        "目标图片文件夹",
        value=st.session_state.get("query-directory", default_directory),
        placeholder="例如：/Users/name/Pictures/to-identify",
        key="query-directory",
    )
    if st.button("识别整个文件夹", type="primary", key="recognize-query-directory"):
        if engine.gallery is None:
            st.error("请先在“建库与审核”页面建立 Gallery。")
        else:
            try:
                directory = Path(directory_value).expanduser().resolve(strict=True)
                paths = query_image_paths(directory)
            except (OSError, ValueError) as error:
                st.error(f"无法读取目标文件夹：{error}")
            else:
                if not paths:
                    st.warning("目标文件夹中没有支持的图片。")
                else:
                    records = recognize_query_folder(engine, directory, paths)
                    st.session_state["folder-recognition-results"] = {
                        "directory": str(directory),
                        "records": records,
                    }
                    st.session_state["folder-results-page"] = 0
                    st.success(f"已完成 {len(records)} 张图片的识别。")

    stored = st.session_state.get("folder-recognition-results")
    if not isinstance(stored, dict):
        return
    records = stored.get("records")
    directory = stored.get("directory")
    if not isinstance(records, list) or not isinstance(directory, str):
        return
    st.divider()
    st.caption(f"结果目录：{directory} · 共 {len(records)} 张图片")
    page_records, page, page_count = paginated_items(records, "folder-results-page")
    for offset in range(0, len(page_records), GRID_COLUMNS):
        columns = st.columns(GRID_COLUMNS)
        for column, record in zip(
            columns, page_records[offset : offset + GRID_COLUMNS], strict=False
        ):
            if not isinstance(record, dict):
                continue
            with column:
                with st.container(border=True):
                    render_folder_result_card(record)
    render_pagination_controls("folder-results-page", page, page_count, len(records))
    detail_path = st.session_state.get("folder-detail-image")
    if isinstance(detail_path, str):
        candidate = Path(detail_path)
        result_directory = Path(directory)
        if candidate.is_file() and candidate.is_relative_to(result_directory):
            render_query_image_dialog(engine, candidate)
        else:
            st.session_state.pop("folder-detail-image", None)
    jsonl = "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n"
    st.download_button(
        "下载本文件夹识别结果（JSONL）",
        data=jsonl,
        file_name="folder_recognition_results.jsonl",
        mime="application/x-ndjson",
    )


def render_recognition(engine: FaceIdEngine) -> None:
    st.header("识别图片")
    render_folder_recognition(engine)
    st.divider()
    st.subheader("临时上传识别")
    st.caption("默认识别图片中的所有脸；可多选需要保留的 Face。图片不会保存或上传。")
    uploads = st.file_uploader(
        "上传一张或多张待识别图片",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        accept_multiple_files=True,
        help=f"单次最多处理 {MAX_UPLOADS} 张图片。",
        key="recognition-uploads",
    )
    if not uploads:
        st.info("选择图片后开始检测。请先在“图库管理”中建立 Gallery。")
        return
    if len(uploads) > MAX_UPLOADS:
        st.warning(f"只处理前 {MAX_UPLOADS} 张图片。")
        uploads = uploads[:MAX_UPLOADS]

    exported: list[dict[str, Any]] = []
    progress = st.progress(0, text="准备识别…")
    for index, upload in enumerate(uploads, start=1):
        content = upload.getvalue()
        image, detections, error_record = prepare_upload(engine, upload.name, content)
        upload_id = sha256(content).hexdigest()[:16]
        selection_key = f"selected-query-face-{upload_id}"
        record: dict[str, Any] = {"image": upload.name, "faces": []}
        with st.container(border=True):
            st.subheader(upload.name)
            if error_record is not None:
                record = error_record
                st.error(str(record["error"]))
            elif image is not None:
                if not detections:
                    record = {"image": upload.name, "detected_face_count": 0, "faces": []}
                    st.warning("未检测到人脸。请换一张正脸更清晰的图片。")
                else:
                    selected_indexes = st.session_state.get(selection_key)
                    if not isinstance(selected_indexes, list) or any(
                        not isinstance(value, int) or not 0 <= value < len(detections)
                        for value in selected_indexes
                    ):
                        selected_indexes = list(range(len(detections)))
                        st.session_state[selection_key] = selected_indexes
                    preview_column, result_column = st.columns((3, 2))
                    with result_column:
                        st.metric("检测到的人脸", len(detections))
                        if len(detections) > 1:
                            selected_indexes = (
                                st.pills(
                                    "要识别的人脸（可多选）",
                                    options=list(range(len(detections))),
                                    format_func=lambda value: f"Face {value + 1}",
                                    selection_mode="multi",
                                    key=selection_key,
                                )
                                or []
                            )
                        else:
                            selected_indexes = [0]
                            st.success("自动识别 Face 1")
                        st.caption("灰色＝未选择；紫色＝已选；黄色＝复核；绿色＝已知；红色＝未知。")
                    results: list[tuple[int, Any]] = []
                    for selected_index in selected_indexes:
                        result_key = f"face-result-{upload_id}-{selected_index}"
                        result = st.session_state.get(result_key)
                        if result is None:
                            try:
                                result = engine.recognize_detection(
                                    image, detections[selected_index]
                                )
                            except Exception as error:
                                record = {"image": upload.name, "error": str(error)}
                                st.error(str(error))
                                break
                        if result is not None:
                            st.session_state[result_key] = result
                            results.append((selected_index, result))
                    statuses = {
                        selected_index: result.decision.status
                        for selected_index, result in results
                    }
                    names = {
                        selected_index: result.decision.identity or "Unknown"
                        for selected_index, result in results
                    }
                    with preview_column:
                        st.image(
                            cv2.cvtColor(
                                detection_preview(
                                    image,
                                    detections,
                                    set(selected_indexes),
                                    recognition_statuses=statuses,
                                    recognition_names=names,
                                ),
                                cv2.COLOR_BGR2RGB,
                            ),
                            width="content",
                            caption=f"预览最长边限制为 {MAX_DISPLAY_LONG_EDGE} px",
                        )
                    if len(detections) > 1:
                        st.caption("切换选择后，识别结果与框颜色会自动更新。")
                    if results:
                        record = {
                            "image": upload.name,
                            "detected_face_count": len(detections),
                            "selected_faces": [
                                selected_index + 1 for selected_index in selected_indexes
                            ],
                            "faces": [serialize_result(result) for _, result in results],
                        }
                        st.dataframe(
                            result_rows_from_record(record),
                            use_container_width=True,
                            hide_index=True,
                        )
                    elif not selected_indexes:
                        st.info("请选择至少一张 Face 后开始识别。")
            else:
                record = {"image": upload.name, "error": "无法准备图片"}
        exported.append(record)
        progress.progress(index / len(uploads), text=f"已处理 {index}/{len(uploads)} 张图片")
    progress.empty()
    jsonl = "\n".join(json.dumps(record, ensure_ascii=False) for record in exported) + "\n"
    st.download_button(
        "下载识别结果（JSONL）",
        data=jsonl,
        file_name="recognition_results.jsonl",
        mime="application/x-ndjson",
    )


def main() -> None:
    st.set_page_config(page_title="seiyuu-faceid", page_icon="👤", layout="wide")
    st.title("seiyuu-faceid")
    st.caption("本地运行：原始图库与上传图片均只保留在本机，不会上传到网络。")

    with st.sidebar:
        st.header("本地模型与 Gallery")
        st.code(str(CONFIG_PATH.relative_to(ROOT)))
        if st.button("重新加载模型和 Gallery"):
            st.cache_resource.clear()
            st.session_state.pop("raw_detection_cache", None)
            st.session_state.pop("raw_recognition_cache", None)
            st.rerun()
        st.caption("更换 ONNX 模型文件，或从命令行重建 Gallery 后，点击此按钮。")

    try:
        engine = load_engine(str(CONFIG_PATH))
    except Exception as error:
        st.error(f"无法加载本地模型或 Gallery：{error}")
        st.stop()

    gallery_tab, build_tab, recognition_tab = st.tabs(["图库管理", "建库与审核", "识别图片"])
    with gallery_tab:
        render_gallery_management(engine)
    with build_tab:
        render_gallery_build(engine)
    with recognition_tab:
        render_recognition(engine)


def result_rows_from_record(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Format serialized pipeline values, avoiding a second inference pass."""
    rows: list[dict[str, Any]] = []
    for number, face in enumerate(record.get("faces", []), start=1):
        if not isinstance(face, dict):
            continue
        candidates = face.get("candidates", [])
        top = " · ".join(
            f"{candidate['identity']} ({candidate['score']:.3f})"
            for candidate in candidates[:5]
            if isinstance(candidate, dict)
        )
        rows.append(
            {
                "face": number,
                "result": face.get("identity") or "Unknown",
                "status": face.get("status"),
                "detector_score": round(float(face.get("detector_score", 0)), 3),
                "quality_score": round(float(face.get("quality_score", 0)), 3),
                "quality_flags": ", ".join(face.get("quality_reasons", [])) or "-",
                "top_candidates": top or "-",
            }
        )
    return rows


if __name__ == "__main__":
    main()
