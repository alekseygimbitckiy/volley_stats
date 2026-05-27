# Volleyball Analysis Pipeline Runbook

This document shows the exact command sequence for the current pipeline.

The best current player marking command was tested on:

```bash
VIDEO=data/game/volleydzen.mp4
STEM=volleydzen
ROSTER=data/config/volleydzen_roster.json
EMBEDDINGS=data/processed/volleydzen_jersey_embeddings/player_embeddings.json
SNAPSHOTS=data/processed/volleydzen_jersey_embeddings/snapshots
BALL_TRACK=data/processed/vball_net_raw/volleydzen.csv
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

Create or edit a roster file with the real players. For `volleydzen.mp4`, use:

```text
data/config/volleydzen_roster.json
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

For `volleydzen.mp4`:

```bash
./venv/bin/python tools/bootstrap_jersey_players.py data/game/volleydzen.mp4 \
  --frames 200 \
  --team-filter court-nearest-6 \
  --identity-types number \
  --roster data/config/volleydzen_roster.json \
  --ocr-backend paddleocr \
  --ocr-languages en \
  --ocr-min-confidence 0.85 \
  --max-samples-per-identity 8 \
  --embedding-backend soccernet-osnet \
  --embedding-device cpu \
  --ocr-device cpu \
  --output data/processed/volleydzen_jersey_embeddings/player_embeddings.json \
  --snapshot-dir data/processed/volleydzen_jersey_embeddings/snapshots \
  --fresh
```

Outputs:

```text
data/processed/volleydzen_jersey_embeddings/player_embeddings.json
data/processed/volleydzen_jersey_embeddings/snapshots/
```

## 4. Run Ball Tracking

Run vball-net and save the ball track:

```bash
./venv/bin/python tools/run_vball_net.py data/game/volleydzen.mp4 \
  --model-path external/vball-net/vb-models/VballNetFastV1_155_h288_w512.onnx \
  --output-dir data/processed/vball_net_raw
```

Outputs:

```text
data/processed/vball_net_raw/volleydzen.csv
data/processed/vball_net_raw/volleydzen.json
```

## 5. Label Video

This is the best current command for player marking on `volleydzen.mp4`:

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

Outputs:

```text
data/processed/volleydzen_labeled/volleydzen_annotated.mp4
data/processed/volleydzen_labeled/volleydzen_test_tracking.json
data/processed/volleydzen_labeled/volleydzen_test_tracking.csv
```

## 6. Label Receive Moments

This labels likely receive/contact moments from the vball-net ball trajectory:

```bash
./venv/bin/python tools/label_receive_from_ball.py data/game/volleydzen.mp4 \
  --ball-track data/processed/vball_net_raw/volleydzen.csv \
  --output-dir data/processed/receive_from_ball \
  --label-hold-sec 2 \
  --trail-length 45
```

Outputs:

```text
data/processed/receive_from_ball/volleydzen_receive_from_ball_annotated.mp4
data/processed/receive_from_ball/volleydzen_receive_from_ball.json
```

## Command Notes

- `--team-filter court-nearest-6` keeps the six detected players nearest to the marked near-side court line.
- `--tracker bytetrack` gives more stable player boxes than raw YOLO detections.
- `--ocr-backend paddleocr` uses PaddleOCR for jersey number reading.
- `--ocr-min-confidence 0.85` ignores OCR reads below this confidence.
- `--ocr-relabel-min-confidence 0.92` requires stronger OCR before changing an existing tracked label.
- `--ocr-skip-overlap-iou 0.25` disables OCR for boxes overlapping another player by at least this IoU.
- `--reid auto` allows OSNet/SoccerNet ReID to help fill missing roster labels.
- `--reid-relabel-max-center-jump 100` rejects ReID roster-fill labels that would jump too far from the player's predicted smooth position.
- `--predict-missing-players` keeps labels moving with estimated velocity when detection temporarily misses a player.
- Avoid `--uniform-color-filter` for the current setup; it was less stable than court filtering plus OCR/ReID/tracking.
