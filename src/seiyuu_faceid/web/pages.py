"""Local Streamlit interface for single and batch face identification."""

from __future__ import annotations

import json
import sys
from hashlib import sha256
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parents[3]
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
from seiyuu_faceid.dataset.scanner import IMAGE_SUFFIXES, RawImage, scan_raw_dataset  # noqa: E402
from seiyuu_faceid.dataset.selections import (  # noqa: E402
    load_face_selections,
    save_face_selections,
)
from seiyuu_faceid.runtime.config import load_config  # noqa: E402
from seiyuu_faceid.runtime.engine import FaceIdEngine  # noqa: E402
from seiyuu_faceid.runtime.pipeline import serialize_result  # noqa: E402
from seiyuu_faceid.vision.detector import FaceDetection  # noqa: E402
from seiyuu_faceid.web.components import (  # noqa: E402
    decode_image,
    detection_preview,
    render_group_review_image,
    resize_for_display,
)
from seiyuu_faceid.web.state import (  # noqa: E402
    consume_dialog_request,
    is_valid_identity,
    query_image_paths,
    result_rows_from_record,
)

CONFIG_PATH = ROOT / "configs/default.yaml"
MAX_UPLOADS = 50
MAX_DISPLAY_LONG_EDGE = 640
GROUP_REVIEW_RENDER_HEIGHT = 960
THUMBNAIL_LONG_EDGE = 240
GRID_COLUMNS = 4
GRID_PAGE_SIZE = 24
RAW_ROOT = ROOT / "data/raw"
EVALUATION_ROOT = ROOT / "data/gallery"
FACE_SELECTIONS_PATH = ROOT / "data/review/raw_faces/face_selections.jsonl"
EVALUATION_FACE_SELECTIONS_PATH = ROOT / "data/review/evaluation_face_selections.jsonl"
DETECTION_CACHE_PATH = ROOT / "data/runtime/detections.jsonl"

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
                const commands = new Set(data.commands || ['enter', 'a', 'y']);
                const command = key === 'Enter' ? 'enter' : normalized;
                if (commands.has(command)) {
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


@st.cache_resource(show_spinner="正在加载本地分类模型…")
def load_engine(config_path: str, config_version: int) -> FaceIdEngine:
    """Cache one CPU model instance for the Streamlit process."""
    return FaceIdEngine(load_config(Path(config_path)))


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


def evaluation_source_image_paths(identity_root: Path, source_directory: Path) -> list[Path]:
    """Return one non-overlapping evaluation source group at a time.

    ``data/gallery/<identity>/`` can contain both direct images and nested
    source folders. The root option deliberately includes only direct images,
    otherwise it would duplicate every nested folder in the review queue.
    """
    if source_directory != identity_root:
        return image_paths(source_directory)
    return sorted(
        path
        for path in identity_root.iterdir()
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
    files = [path for directory in identity_directories() for path in image_paths(directory)]
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
    """Classify cached detections once per raw image and loaded model."""
    image_stat = image_path.stat()
    cache_key = (
        f"{image_path}:{image_stat.st_mtime_ns}:{image_stat.st_size}:{engine.recognition_id}"
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


def evaluation_face_selections() -> dict[str, list[int]]:
    """Load explicit keep/exclude choices for potentially mixed evaluation folders."""
    key = "evaluation_face_selections"
    if key not in st.session_state:
        st.session_state[key] = (
            load_face_selections(EVALUATION_FACE_SELECTIONS_PATH, allow_incomplete=True)
            if EVALUATION_FACE_SELECTIONS_PATH.is_file()
            else {}
        )
    return st.session_state[key]


def set_evaluation_face_selection(source_image: str, face_indexes: list[int]) -> None:
    selections = evaluation_face_selections()
    selections[source_image] = sorted(set(face_indexes))
    save_face_selections(EVALUATION_FACE_SELECTIONS_PATH, selections)


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
    if detections:
        try:
            recognition_results = raw_recognition_results(engine, image_path, image, detections)
        except Exception as error:
            st.error(f"自动识别失败：{error}")
    statuses = {index: result.decision.status for index, result in recognition_results.items()}
    names = {
        index: result.decision.identity
        or ("Review" if result.decision.status == "review" else "Unknown")
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
    if detections:
        st.info("请使用下方“审核人脸”确认哪些 Face 属于该身份。")
    else:
        st.warning("未检测到人脸；整理样本时会把它记录为拒绝项。")


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
    if detections:
        try:
            recognition_results = raw_recognition_results(engine, image_path, image, detections)
        except Exception as error:
            st.error(f"自动识别失败：{error}")
    statuses = {index: result.decision.status for index, result in recognition_results.items()}
    names = {
        index: result.decision.identity
        or ("Review" if result.decision.status == "review" else "Unknown")
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
    """Find every cached detected face that still needs a human selection."""
    selections = reviewed_face_selections()
    pending: list[tuple[RawImage, list[FaceDetection]]] = []
    for source in sources:
        record = detection_records[source.relative_path.as_posix()]
        detections = list(record.detections)
        if record.error is not None or not detections:
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


def prepare_training_data(
    engine: FaceIdEngine, detection_records: dict[str, DetectionRecord]
) -> tuple[int, int]:
    """Prepare reviewed faces; this does not retrain or replace the selected model."""
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
    return accepted, rejected


def render_group_review(
    engine: FaceIdEngine,
    source: RawImage,
    detections: list[FaceDetection],
) -> None:
    """Present one source image at a time and accept any number of selected faces."""
    source_image = source.relative_path.as_posix()
    image = cv2.imread(str(source.path))
    if image is None:
        st.error("无法解码这张原始图片；请修复或移除文件后重新扫描。")
        return
    current = group_review_draft(source_image, reviewed_face_selections().get(source_image, []))
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
    if detections:
        try:
            recognition_results = raw_recognition_results(engine, source.path, image, detections)
        except Exception as error:
            st.warning(f"自动建议暂不可用：{error}")
    statuses = {index: result.decision.status for index, result in recognition_results.items()}
    names = {
        index: result.decision.identity
        or ("Review" if result.decision.status == "review" else "Unknown")
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
            st.caption("人脸审核 · 当前目标身份")
            st.markdown(f"## 🎯 {source.identity}")
            st.caption(source.path.name)
            st.caption(f"检测到 {len(detections)} 张人脸")
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


def run_dataset_build(engine: FaceIdEngine, detection_records: dict[str, DetectionRecord]) -> None:
    """Prepare reviewed training samples without changing the deployed classifier."""
    st.session_state["group_review_started"] = False
    with st.spinner("正在对齐已审核的人脸…"):
        try:
            accepted, rejected = prepare_training_data(engine, detection_records)
        except Exception as error:
            st.error(f"样本整理失败：{error}")
        else:
            st.success(f"样本整理完成：{accepted} 张脸；拒绝 {rejected} 项。")
            st.info("当前分类模型保持不变。新增样本需要重新训练后才能用于分类。")


def render_evaluation_review(engine: FaceIdEngine) -> None:
    """Review labelled holdout faces from the shared total-review page."""
    if not EVALUATION_ROOT.is_dir():
        st.info("data/gallery 不存在，暂无独立评测集可审核。")
        return
    identity_directories = sorted(path for path in EVALUATION_ROOT.iterdir() if path.is_dir())
    if not identity_directories:
        st.info("data/gallery 中暂无按身份整理的评测图片。")
        return
    selections = evaluation_face_selections()
    identity_key = "evaluation-review-identity"
    pending_identities = [
        directory
        for directory in identity_directories
        if any(
            path.relative_to(EVALUATION_ROOT).as_posix() not in selections
            for path in image_paths(directory)
        )
    ]
    current_identity = st.session_state.get(identity_key)
    if pending_identities and current_identity not in {path.name for path in pending_identities}:
        st.session_state[identity_key] = pending_identities[0].name
    image_column, control_column = st.columns((6, 2), gap="large", vertical_alignment="center")
    with control_column:
        with st.container(border=True):
            st.caption("总审核 · 评测集")
            identity = st.selectbox(
                "评测身份",
                [path.name for path in identity_directories],
                key=identity_key,
            )
    identity_root = EVALUATION_ROOT / identity
    root_images = evaluation_source_image_paths(identity_root, identity_root)
    source_directories = [
        *([identity_root] if root_images else []),
        *sorted(path for path in identity_root.iterdir() if path.is_dir() and image_paths(path)),
    ]
    if not source_directories:
        st.info("该身份下没有可审核的图片子目录。")
        return
    pending_by_source = {
        source: [
            path
            for path in evaluation_source_image_paths(identity_root, source)
            if path.relative_to(EVALUATION_ROOT).as_posix() not in selections
        ]
        for source in source_directories
    }
    source_key = f"evaluation-review-source-{identity}"
    available_sources = [source for source in source_directories if pending_by_source[source]]
    current_source = st.session_state.get(source_key)
    if available_sources and current_source not in available_sources:
        # Match the enrolment review flow: when one source is exhausted, move
        # straight to the next source that still needs a decision.
        st.session_state[source_key] = available_sources[0]
    with control_column:
        source_directory = st.selectbox(
            "图片来源目录",
            source_directories,
            format_func=lambda path: (
                f"根目录（{len(root_images)} 张）"
                if path == identity_root
                else f"{path.name}（{len(image_paths(path))} 张）"
            ),
            key=source_key,
        )
    identity_paths = image_paths(identity_root)
    identity_reviewed = sum(
        path.relative_to(EVALUATION_ROOT).as_posix() in selections for path in identity_paths
    )
    paths = evaluation_source_image_paths(identity_root, source_directory)
    pending = pending_by_source[source_directory]
    reviewed = len(paths) - len(pending)
    with control_column:
        if identity_paths:
            st.progress(
                identity_reviewed / len(identity_paths),
                text=f"{identity} 总审核进度：{identity_reviewed}/{len(identity_paths)}",
            )
        if paths:
            st.progress(
                reviewed / len(paths),
                text=f"当前来源审核进度：{reviewed}/{len(paths)}",
            )
        st.caption(f"已审核 {reviewed}/{len(paths)} 张；待审核 {len(pending)} 张。")
    if not pending:
        with control_column:
            st.success("该来源目录已全部完成审核。可重新运行独立评测。")
        return
    path = pending[0]
    source_image = path.relative_to(EVALUATION_ROOT).as_posix()
    image, detections = raw_image_and_detections(engine, path)
    if image is None:
        detections = []
    elif not detections:
        # A reviewed evaluation image without a detected face can never become
        # an identification sample. Treat it as an explicit "N" and move on.
        set_evaluation_face_selection(source_image, [])
        reset_evaluation_review_draft()
        st.rerun()
    current = evaluation_review_draft(
        source_image, evaluation_face_selections().get(source_image, [])
    )
    keyboard_source = f"evaluation:{source_image}"
    keyboard_event = GROUP_REVIEW_KEYBOARD(
        key=f"evaluation-review-keys-{sha256(source_image.encode()).hexdigest()[:12]}",
        data={"source": keyboard_source, "faceCount": len(detections)},
        on_key_change=lambda: None,
        width=1,
        height=1,
    )
    key_event = keyboard_event.get("key")
    pressed_key = (
        key_event.get("key")
        if isinstance(key_event, dict) and key_event.get("source") == keyboard_source
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
                st.session_state["evaluation_review_draft_indexes"] = sorted(updated)
                st.rerun()
        elif normalized_key == "a":
            st.session_state["evaluation_review_draft_indexes"] = list(range(len(detections)))
            st.rerun()
        elif normalized_key == "y":
            st.session_state["evaluation_review_apply_suggestion"] = source_image
        elif pressed_key == "Enter":
            set_evaluation_face_selection(source_image, current)
            reset_evaluation_review_draft()
            st.rerun()
    recognition_results: dict[int, Any] = {}
    if image is not None:
        try:
            recognition_results = raw_recognition_results(engine, path, image, detections)
        except Exception as error:
            st.warning(f"自动建议暂不可用：{error}")
    statuses = {index: result.decision.status for index, result in recognition_results.items()}
    names = {
        index: result.decision.identity
        or ("Review" if result.decision.status == "review" else "Unknown")
        for index, result in recognition_results.items()
    }
    suggested_indexes = [
        index
        for index, result in recognition_results.items()
        if result.decision.status == "recognized" and result.decision.identity == identity
    ]
    if st.session_state.pop("evaluation_review_apply_suggestion", None) == source_image:
        if suggested_indexes:
            set_evaluation_face_selection(source_image, suggested_indexes)
            reset_evaluation_review_draft()
            st.rerun()
        st.warning("当前图片没有可信建议，请改用数字键选择。")
    if image is not None:
        preview = detection_preview(
            image,
            detections,
            set(current),
            recognition_statuses=statuses,
            recognition_names=names,
            max_height=GROUP_REVIEW_RENDER_HEIGHT,
        )
        with image_column:
            render_group_review_image(preview)
    else:
        with image_column:
            st.info("预览不可用。请选择排除，或修复图片后重新审核。")
    with control_column:
        with st.container(border=True):
            st.caption("评测集审核 · 当前目标身份")
            st.markdown(f"## 🎯 {identity}")
            st.caption(source_image)
            st.caption(f"检测到 {len(detections)} 张人脸")
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
            later = st.button(
                "稍后再审核", use_container_width=True, key=f"evaluation-later-{source_image}"
            )
    if confirmed:
        set_evaluation_face_selection(source_image, current)
        reset_evaluation_review_draft()
        st.rerun()
    if use_suggestions:
        set_evaluation_face_selection(source_image, suggested_indexes)
        reset_evaluation_review_draft()
        st.rerun()
    if select_all:
        st.session_state["evaluation_review_draft_indexes"] = list(range(len(detections)))
        st.rerun()
    if later:
        reset_evaluation_review_draft()
        st.rerun()


def group_review_draft(source_image: str, existing: list[int]) -> list[int]:
    """Keep number-key toggles local until the user confirms this photo."""
    if st.session_state.get("group_review_draft_source") != source_image:
        st.session_state["group_review_draft_source"] = source_image
        st.session_state["group_review_draft_indexes"] = sorted(set(existing))
    return list(st.session_state["group_review_draft_indexes"])


def reset_group_review_draft() -> None:
    st.session_state.pop("group_review_draft_source", None)
    st.session_state.pop("group_review_draft_indexes", None)


def evaluation_review_draft(source_image: str, existing: list[int]) -> list[int]:
    """Keep evaluation keyboard choices separate from the enrollment review draft."""
    if st.session_state.get("evaluation_review_draft_source") != source_image:
        st.session_state["evaluation_review_draft_source"] = source_image
        st.session_state["evaluation_review_draft_indexes"] = sorted(set(existing))
    return list(st.session_state["evaluation_review_draft_indexes"])


def reset_evaluation_review_draft() -> None:
    """Discard a transient evaluation choice after leaving the current image."""
    st.session_state.pop("evaluation_review_draft_source", None)
    st.session_state.pop("evaluation_review_draft_indexes", None)


def render_dataset_review(engine: FaceIdEngine) -> None:
    st.header("训练样本审核")
    review_scope = st.segmented_control(
        "审核对象",
        options=["训练集（data/raw）", "评测集（data/gallery）"],
        default="训练集（data/raw）",
        key="review-scope",
    )
    if review_scope == "评测集（data/gallery）":
        render_evaluation_review(engine)
        return
    sources, detection_records = current_raw_detection_cache()
    if detection_records is None:
        st.divider()
        st.subheader("整理样本")
        st.caption("先完整扫描一次，再逐题审核所有检测到的人脸；审核与整理样本都会复用扫描结果。")
        st.info("尚未扫描，或 raw 图片已变更。扫描仅检测人脸，不会生成对齐脸或修改原图。")
        if st.button("扫描全部原始图片", type="primary", key="scan-all-raw-images"):
            with st.spinner("正在运行人脸检测…"):
                sources, detection_records = scan_all_raw_images(engine)
            reviewable_count = sum(1 for record in detection_records.values() if record.detections)
            st.success(
                f"扫描完成：{len(sources)} 张图片，其中 {reviewable_count} 张含人脸图片待审核。"
            )
            st.rerun()
        return
    reviewable_count = sum(1 for record in detection_records.values() if record.detections)
    pending = raw_review_items(sources, detection_records)
    reviewed_count = reviewable_count - len(pending)
    if st.session_state.get("group_review_active"):
        if pending:
            st.progress(
                reviewed_count / reviewable_count if reviewable_count else 1.0,
                text=f"训练集总审核进度：{reviewed_count}/{reviewable_count}",
            )
            render_group_review(engine, *pending[0])
        else:
            st.session_state["group_review_active"] = False
            st.session_state["build_after_group_review"] = True
            st.rerun()
        return
    st.divider()
    st.subheader("整理样本")
    st.caption("先完整扫描一次，再逐题审核所有检测到的人脸；审核与整理样本都会复用扫描结果。")
    st.caption(
        f"扫描缓存有效：{len(sources)} 张图片；含人脸图片 {reviewable_count} 张，"
        f"待审核 {len(pending)} 张。"
    )
    if reviewable_count:
        st.progress(
            reviewed_count / reviewable_count,
            text=f"训练集总审核进度：{reviewed_count}/{reviewable_count}",
        )
    if not st.session_state.get("group_review_started"):
        if pending and st.button("开始逐题审核人脸", type="primary", key="begin-group-review"):
            st.session_state["group_review_started"] = True
            st.session_state["group_review_active"] = True
            st.rerun()
        elif not pending:
            st.success("没有待审核的人脸，可以开始整理样本。")
    if pending:
        st.warning(f"有 {len(pending)} 张含人脸图片尚未审核。")
        if st.button("继续逐题审核人脸", type="primary", key="continue-group-review"):
            st.session_state["group_review_active"] = True
            st.rerun()
        return
    st.success("所有检测到的人脸均已审核，可以开始整理样本。")
    if st.session_state.pop("build_after_group_review", False):
        run_dataset_build(engine, detection_records)
    elif st.button("生成已审核的训练样本", type="primary", key="confirm-dataset-build"):
        run_dataset_build(engine, detection_records)


def render_dataset_management(engine: FaceIdEngine) -> None:
    st.header("训练样本")
    st.caption("浏览或上传 data/raw 中的身份图片。扫描、群像审核和整理样本位于“样本审核”页面。")
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
            detail_path = consume_dialog_request(st.session_state, "raw-detail-image")
            if isinstance(detail_path, str):
                candidate = Path(detail_path)
                if candidate.is_file() and candidate.is_relative_to(directory):
                    render_raw_image_dialog(engine, candidate)
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
                    "model_id": engine.recognition_id,
                    "records": records,
                }
                st.session_state["folder-results-page"] = 0
                st.success(f"已完成 {len(records)} 张图片的识别。")

    stored = st.session_state.get("folder-recognition-results")
    if not isinstance(stored, dict) or stored.get("model_id") != engine.recognition_id:
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
    detail_path = consume_dialog_request(st.session_state, "folder-detail-image")
    if isinstance(detail_path, str):
        candidate = Path(detail_path)
        result_directory = Path(directory)
        if candidate.is_file() and candidate.is_relative_to(result_directory):
            render_query_image_dialog(engine, candidate)
    jsonl = "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n"
    st.download_button(
        "下载本文件夹识别结果（JSONL）",
        data=jsonl,
        file_name="folder_recognition_results.jsonl",
        mime="application/x-ndjson",
    )


def render_recognition(engine: FaceIdEngine) -> None:
    st.header("识别图片")
    st.caption(
        f"接受阈值：{engine.config.probability_threshold:.2f}；低于阈值转人工复核。"
        "类别之外的人物和角色图需人工判断。候选分数为类别概率。"
    )
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
        st.info("选择图片后，使用已训练的模型分类。")
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
                        result_key = (
                            f"face-result-{engine.recognition_id}-{upload_id}-{selected_index}"
                        )
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
                        selected_index: result.decision.status for selected_index, result in results
                    }
                    names = {
                        selected_index: result.decision.identity
                        or ("Review" if result.decision.status == "review" else "Unknown")
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
        st.header("当前分类模型")
        st.code(str(CONFIG_PATH.relative_to(ROOT)))
        if st.button("重新加载模型"):
            st.cache_resource.clear()
            st.session_state.pop("raw_detection_cache", None)
            st.session_state.pop("raw_recognition_cache", None)
            st.rerun()
        st.caption("AdaFace + R50 · 50 个身份 · 留出集准确率 98.48%。")

    try:
        engine = load_engine(str(CONFIG_PATH), CONFIG_PATH.stat().st_mtime_ns)
    except Exception as error:
        st.error(f"无法加载本地模型：{error}")
        st.stop()

    dataset_tab, build_tab, recognition_tab = st.tabs(["训练样本", "样本审核", "识别图片"])
    with dataset_tab:
        render_dataset_management(engine)
    with build_tab:
        render_dataset_review(engine)
    with recognition_tab:
        render_recognition(engine)


if __name__ == "__main__":
    main()
