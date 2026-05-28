# Volleyball Analysis Pipeline Runbook

This document shows the exact command sequence for the current pipeline.

The best current player marking command was tested on:

```bash
VIDEO=data/game/volleydzen.mp4
STEM=volleydzen
ROSTER=data/config/volleydzen_roster.json
EMBEDDINGS=data/processed/volleydzen_jersey_embeddings/player_embeddings.json
SNAPSHOTS=data/processed/volleydzen_jersey_embeddings/snapshots
FAST_BALL_TRACK=data/processed/fast_vball_raw/volleydzen.csv
BALL_TRACK="$FAST_BALL_TRACK"
```

## 1. Mark Court

Run the court marking UI:

```bash
./venv/bin/python tools/field_marking_ui.py --host 127.0.0.1 --port 8765
```

Open:

```text
http://127.0.0.1:8765/
```

Mark the near-side court polygon and save. The UI writes:

```text
data/processed/calibrations/field_layout.json
data/processed/calibrations/field_layout_annotated.png
data/processed/calibrations/field_layout_source_frame.png
```

This step is essential. If you skip it, the scripts will use the existing saved layout in `data/processed/calibrations/field_layout.json`. That is only valid if it was marked for the same camera position, video orientation, and resolution.

If port `8765` is busy, use another port:

```bash
./venv/bin/python tools/field_marking_ui.py --host 127.0.0.1 --port 8766
```

## 2. Prepare Roster

Create or edit a roster file with the real players. For the example above, use:

```text
$ROSTER
```

The file should contain one player per jersey number. Example format:

```json
{
  "players": [
    {"player_id": "jersey_10", "jersey_number": "10", "names": []},
    {"player_id": "jersey_11", "jersey_number": "11", "names": []}
  ]
}
```

## 3. Bootstrap Jersey Snapshots And Embeddings

This samples random frames, detects the six near-side players, reads jersey numbers with OCR, saves confident crops, and builds OSNet embeddings.

For the selected video:

```bash
./venv/bin/python tools/bootstrap_jersey_players.py "$VIDEO" \
  --frames 200 \
  --team-filter court-nearest-6 \
  --identity-types number \
  --roster "$ROSTER" \
  --ocr-backend paddleocr \
  --ocr-languages en \
  --ocr-min-confidence 0.85 \
  --max-samples-per-identity 8 \
  --embedding-backend soccernet-osnet \
  --embedding-device cpu \
  --ocr-device cpu \
  --output "$EMBEDDINGS" \
  --snapshot-dir "$SNAPSHOTS" \
  --fresh
```

Outputs:

```text
$EMBEDDINGS
$SNAPSHOTS/
```

## 3b. Optional Manual Player Snapshot Fixes

If automatic bootstrapping misses a player or saves weak crops, you can add or remove snapshots manually:

```bash
./venv/bin/python tools/player_embedding_ui.py \
  --video "$VIDEO" \
  --output "$EMBEDDINGS" \
  --snapshot-dir "$SNAPSHOTS" \
  --host 127.0.0.1 \
  --port 8767
```

Open:

```text
http://127.0.0.1:8767/
```

Use player IDs from the roster, for example `jersey_10`, `jersey_11`, `jersey_12`.
After manual changes, rebuild OSNet embeddings before running video labeling:

```bash
./venv/bin/python tools/rebuild_player_embeddings_osnet.py \
  --input "$EMBEDDINGS" \
  --output "$EMBEDDINGS" \
  --backend soccernet-osnet \
  --device cpu
```

## 4. Run Ball Tracking

Run the fast volleyball tracker and save the normalized ball track:

```bash
./venv/bin/python tools/run_fast_vball.py "$VIDEO" \
  --backend onnx \
  --model-path external/fast-volleyball-tracking-inference/models/VballNetV1_seq9_grayscale_148_h288_w512.onnx \
  --track-length 32 \
  --output-dir data/processed/fast_vball_raw
```

Outputs:

```text
$FAST_BALL_TRACK
data/processed/fast_vball_raw/$STEM.json
data/processed/fast_vball_raw/$STEM/ball.csv
```

Set the active ball track for the rest of the pipeline:

```bash
BALL_TRACK="$FAST_BALL_TRACK"
```

To create a ball-only debug video without player labels:

```bash
./venv/bin/python tools/test_track_video.py "$VIDEO" \
  --players off \
  --ball-track "$BALL_TRACK" \
  --ball-source vball-net \
  --max-ball-gap 0 \
  --ball-max-jump 100 \
  --ball-reacquire-gap 5 \
  --ball-reacquire-max-jump 1000 \
  --team-filter none \
  --tracker iou \
  --ocr off \
  --reid off \
  --output-dir "data/processed/${STEM}_ball_only"
```

Output:

```text
data/processed/${STEM}_ball_only/${STEM}_annotated.mp4
```

## 5. Label Video

This is the best current command format for player marking:

```bash
./venv/bin/python tools/test_track_video.py "$VIDEO" \
  --embeddings "$EMBEDDINGS" \
  --ball-track "$BALL_TRACK" \
  --team-filter court-nearest-6 \
  --fill-roster-labels \
  --predict-missing-players \
  --max-player-prediction-gap 45 \
  --tracker bytetrack \
  --reid auto \
  --ocr auto \
  --ocr-backend paddleocr \
  --roster "$ROSTER" \
  --ocr-languages en \
  --ocr-min-confidence 0.85 \
  --ocr-relabel-min-confidence 0.92 \
  --ocr-skip-overlap-iou 0.25 \
  --reid-relabel-max-center-jump 100 \
  --ocr-every-n-frames 15 \
  --match-threshold 0 \
  --output-dir "data/processed/${STEM}_labeled"
```

Outputs:

```text
data/processed/${STEM}_labeled/${STEM}_annotated.mp4
data/processed/${STEM}_labeled/${STEM}_test_tracking.json
data/processed/${STEM}_labeled/${STEM}_test_tracking.csv
```

## 6. Label Receive Moments

This labels likely receive/contact moments from the vball-net ball trajectory:

```bash
./venv/bin/python tools/label_receive_from_ball.py "$VIDEO" \
  --ball-track "$BALL_TRACK" \
  --output-dir data/processed/receive_from_ball \
  --label-hold-sec 2 \
  --trail-length 45
```

Outputs:

```text
data/processed/receive_from_ball/${STEM}_receive_from_ball_annotated.mp4
data/processed/receive_from_ball/${STEM}_receive_from_ball.json
```

## Command Notes

- `--team-filter court-nearest-6` keeps the six detected players nearest to the marked near-side court line.
- `--tracker bytetrack` gives more stable player boxes than raw YOLO detections.
- `--ocr-backend paddleocr` uses PaddleOCR for jersey number reading.
- `--ocr-min-confidence 0.85` ignores OCR reads below this confidence.
- `--ocr-relabel-min-confidence 0.92` requires stronger OCR before changing an existing tracked label.
- `--ocr-skip-overlap-iou 0.25` disables OCR for boxes overlapping another player by at least this IoU.
- `--ocr-every-n-frames 15` runs OCR every 15 frames to reduce CPU cost. Use `--ocr-every-n-frames 1` to try OCR on every frame, but PaddleOCR is heavy and this will be much slower.
- `--reid auto` allows OSNet/SoccerNet ReID to help fill missing roster labels.
- `--reid-relabel-max-center-jump 100` rejects ReID roster-fill labels that would jump too far from the player's predicted smooth position.
- `--predict-missing-players` keeps labels moving with estimated velocity when detection temporarily misses a player.
- Avoid `--uniform-color-filter` for the current setup; it was less stable than court filtering plus OCR/ReID/tracking.
- `--ball-max-jump 100` rejects a detected ball if it is more than 100 pixels from the predicted next ball position during normal continuous tracking. This removes single-frame noisy jumps.
- `--ball-reacquire-gap 5` starts relaxed reacquisition after 5 consecutive frames without an accepted real ball detection.
- `--ball-reacquire-max-jump 1000` allows the first real detection after that gap to be up to 1000 pixels from the old prediction. After that detection is accepted, the normal `--ball-max-jump 100` limit is used again.
