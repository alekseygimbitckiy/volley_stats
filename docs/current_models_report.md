# Current Models And Tracking Report

This report describes the models currently used by the volleyball analysis pipeline and how they interact.

## Overview

The current pipeline combines:

- Court layout from manual marking.
- YOLO person detection.
- vball-net ball tracking.
- PaddleOCR jersey number reading.
- ByteTrack player tracking.
- OSNet/SoccerNet ReID embeddings.
- Roster-based label correction and prediction.

The best current player marking command is:

```bash
./venv/bin/python tools/test_track_video.py data/game/volleydzen.mp4 \
  --embeddings data/processed/volleydzen_jersey_embeddings/player_embeddings.json \
  --ball-track data/processed/vball_net_raw/volleydzen.csv \
  --team-filter court-nearest-6 \
  --fill-roster-labels \
  --predict-missing-players \
  --max-player-prediction-gap 45 \
  --tracker bytetrack \
  --reid auto \
  --ocr auto \
  --ocr-backend paddleocr \
  --roster data/config/volleydzen_roster.json \
  --ocr-languages en \
  --ocr-min-confidence 0.85 \
  --ocr-relabel-min-confidence 0.92 \
  --ocr-skip-overlap-iou 0.25 \
  --reid-relabel-max-center-jump 100 \
  --ocr-every-n-frames 15 \
  --match-threshold 0 \
  --output-dir data/processed/volleydzen_labeled
```

## Court Marking

Script:

```text
tools/field_marking_ui.py
```

Output:

```text
data/processed/calibrations/field_layout.json
```

This is not a model. It is a manually marked near-side court polygon. The pipeline uses it to restrict player detections to the near side of the court.

Manual court segmentation is essential. If you do not mark the court for the current camera/video setup, the pipeline will use the existing saved layout in `data/processed/calibrations/field_layout.json`. If that file comes from another video orientation or resolution, player filtering can select the wrong boxes.

The important filter is:

```bash
--team-filter court-nearest-6
```

It keeps player boxes that intersect the marked court and chooses the six nearest to the near-side/front court line.

## Player Detection

Model:

```text
YOLO, default yolov8n.pt
```

Used by:

```text
tools/test_track_video.py
tools/bootstrap_jersey_players.py
```

Purpose:

- Detect person bounding boxes in each frame.
- Provide the raw player boxes passed into court filtering, OCR, ReID, and tracking.

YOLO does not know volleyball player identity. It only detects people.

## Ball Tracking

Model:

```text
vball-net ONNX
external/vball-net/vb-models/VballNetFastV1_155_h288_w512.onnx
```

Wrapper:

```text
tools/run_vball_net.py
```

Output:

```text
data/processed/vball_net_raw/<video_stem>.csv
data/processed/vball_net_raw/<video_stem>.json
```

Purpose:

- Detect the volleyball position frame by frame.
- Provide the ball trace used for annotated videos.
- Provide movement data for receive/contact moment detection.

## OCR

Current backend:

```text
PaddleOCR
```

Code:

```text
tools/jersey_ocr.py
```

Important flags:

```bash
--ocr auto
--ocr-backend paddleocr
--ocr-languages en
--ocr-min-confidence 0.85
--ocr-relabel-min-confidence 0.92
--ocr-skip-overlap-iou 0.25
--ocr-every-n-frames 15
```

How it works:

- Crops each detected player.
- Reads shirt number, and optionally shirt name if configured.
- Ignores OCR below `--ocr-min-confidence`.
- Does not OCR a player box if it overlaps another player by `--ocr-skip-overlap-iou`.
- Can correct an existing track label only above `--ocr-relabel-min-confidence`.

In the current setup, OCR is trusted only when it is confident. It is not used when players overlap because partial numbers are often misread.

## Player Tracking

Current tracker:

```text
ByteTrack
```

Flag:

```bash
--tracker bytetrack
```

Purpose:

- Connect person detections across frames.
- Keep the same track ID while a player moves.
- Reduce label flicker compared with frame-by-frame detection.

ByteTrack still can create a new track when a player is occluded, partially visible, or the detector misses them. The pipeline handles that with roster fill and prediction.

## ReID Embeddings

Current ReID model:

```text
OSNet with SoccerNet/SportsReID checkpoint
```

Embedding file:

```text
data/processed/volleydzen_jersey_embeddings/player_embeddings.json
```

Snapshot folder:

```text
data/processed/volleydzen_jersey_embeddings/snapshots/
```

Purpose:

- Build a visual embedding for each roster player from OCR-confirmed snapshots.
- Help fill labels when OCR cannot read a number in the current frame.

Important flags:

```bash
--reid auto
--match-threshold 0
--reid-relabel-max-center-jump 100
```

`--reid-relabel-max-center-jump 100` prevents ReID from assigning a missing roster label to a box that is too far from that player's predicted smooth location. This was added because unconstrained ReID could jump labels across the court.

## Roster Label Logic

Roster file:

```text
data/config/volleydzen_roster.json
```

Important flags:

```bash
--roster data/config/volleydzen_roster.json
--fill-roster-labels
```

How it works:

- OCR can assign a player if the parsed number matches the roster.
- Tracking keeps the label once it is attached to a track.
- ReID can fill missing labels from the roster.
- Duplicate labels are resolved so only one visible detection uses each roster player.

## Missing Player Prediction

Important flags:

```bash
--predict-missing-players
--max-player-prediction-gap 45
```

How it works:

- If a roster-labeled player is not detected, the code estimates velocity from recent boxes.
- It predicts a temporary box for that player.
- This keeps labels moving smoothly instead of disappearing immediately.

This prediction is only useful after a player has already been detected and labeled at least once.

## Current Best Behavior

The best current marking result comes from combining:

- Court filter: `court-nearest-6`
- Tracker: `bytetrack`
- OCR: `paddleocr`
- ReID: `auto`
- Roster fill: enabled
- Missing-player prediction: enabled
- ReID center-jump guard: `100`

This setup lets OCR correct labels when it is confident, uses ByteTrack for frame-to-frame continuity, uses ReID only as a fallback, and prevents ReID from making large identity jumps.

## Known Weak Points

- If ByteTrack loses a player and creates a new track, the label may briefly disappear or show a track ID before OCR/ReID/roster fill recovers it.
- If a player never receives a confident OCR or ReID label, prediction cannot know who that player is.
- OCR is unreliable when players overlap or turn sideways.
- ReID can confuse similar uniforms, so it must be constrained by smooth position checks.
- The court polygon must match the video camera orientation and resolution.
