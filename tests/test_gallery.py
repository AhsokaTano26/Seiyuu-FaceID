from __future__ import annotations

import numpy as np

from seiyuu_faceid.gallery.index import Gallery


def test_gallery_round_trip(tmp_path) -> None:
    gallery = Gallery.from_references(
        np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32),
        ["a", "b"],
        [{"source_image": "a.jpg"}, {"source_image": "b.jpg"}],
    )
    vectors = tmp_path / "gallery.npz"
    metadata = tmp_path / "gallery.json"
    gallery.save(vectors, metadata)
    restored = Gallery.load(vectors, metadata)
    assert restored.reference_identities == ("a", "b")
    assert np.allclose(restored.references, gallery.references)
    assert restored.prototype_identities == ("a", "b")
