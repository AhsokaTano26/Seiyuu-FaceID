# Local data layout

- `raw/<identity>/`: user-provided reference images. You may add files there
  manually or through the web UI; tools never alter existing source images.
- `aligned/`: accepted 112×112 face crops and `metadata.jsonl`.
- `rejected/`: JSONL records describing skipped faces/images.
- `gallery/<identity>/`: user-provided labelled holdout images used only for
  independent evaluation. This directory is never written by build tools.
- `runtime/`: disposable generated artifacts: the production embedding gallery,
  detection cache and evaluation reports.
- `query/`: optional local input directory for batch recognition. Its images are
  read-only query inputs and are never added to the Gallery.
- `cache/`: legacy disposable cache location; new artifacts use `runtime/`.
- `review/raw_faces/`: local numbered previews and the manual selection manifest
  for multi-face source images.

All of these directories are git-ignored except their `.gitkeep` placeholders.
