from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_streamlit_app_renders_without_an_unhandled_exception() -> None:
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py")
    app.run(timeout=30)
    assert not app.exception
    assert app.title[0].value == "seiyuu-faceid"
    root = Path(__file__).resolve().parents[1]
    if (root / "models/selected/manifest.json").is_file():
        assert len(app.tabs) == 3
        assert not app.error
    else:
        # Private model weights are absent in CI: show the load error visibly.
        assert any("无法加载本地模型" in error.value for error in app.error)
