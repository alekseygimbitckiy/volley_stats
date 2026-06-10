# Volleyball Analysis Pipeline Runbook

This document shows the current command sequence for processing one rally video.

The pipeline is:

1. Mark the near-side court.
2. Prepare roster and player embeddings.
3. Run fast volleyball ball tracking.
4. Clean the ball track.
5. Label the six nearest-team players.
6. Classify serve, receive, and technical actions with the pose SVM.

## 0. Set Variables

Set these variables first. Example for `video5224478607757318598_fxWedEdS.mp4`:

```bash
VIDEO=data/game/video5224478607757318598_fxWedEdS.mp4
STEM=video5224478607757318598_fxWedEdS
ROSTER=data/config/player_roster.json
EMBEDDINGS=data/processed/auto_jersey_embeddings/player_embeddings.json
SNAPSHOTS=data/processed/auto_jersey_embeddings/snapshots
FAST_BALL_TRACK=data/processed/fast_vball_raw/${STEM}.csv
BALL_TRACK="$FAST_BALL_TRACK"
CLEAN_OUT_DIR=data/processed/${STEM}_ball_only
CLEAN_BALL_TRACK=${CLEAN_OUT_DIR}/${STEM}_test_tracking.json
OUT_DIR=data/processed/${STEM}_labeled_fresh
TRACKING_JSON=${OUT_DIR}/${STEM}_test_tracking.json
POSE_SVM=data/processed/action_pose_dataset_batch/svm_model/pose_svm.joblib
POSE_MODEL=external/mediapipe/pose_landmarker_lite.task
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

Manual court segmentation is essential. If you skip this step, the existing saved layout is used. That is only valid when it was marked for the same camera position, orientation, and resolution.

If port `8765` is busy:

```bash
./venv/bin/python tools/field_marking_ui.py --host 127.0.0.1 --port 8766
```

## 2. Prepare Roster

Create or edit the roster file:

```text
$ROSTER
```

Example format:

```json
{
  "players": [
    {"player_id": "jersey_10", "jersey_number": "10", "names": []},
    {"player_id": "jersey_11", "jersey_number": "11", "names": []}
  ]
}
```

Use Cyrillic or Latin names if they are visible on shirts. OCR matching accepts similar names, but the jersey number must match the roster entry when both name and number are used.

## 3. Bootstrap Jersey Snapshots And Embeddings

This samples frames, detects near-side players, reads jersey numbers or names with OCR, saves confident crops, and builds OSNet/SoccerNet ReID embeddings.

```bash
./venv/bin/python tools/bootstrap_jersey_players.py "$VIDEO" \
  --frames 200 \
  --team-filter court-nearest-6 \
  --identity-types number \
  --roster "$ROSTER" \
  --ocr-backend paddleocr \
  --ocr-languages en,ru \
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

## 4. Optional Manual Player Snapshot Fixes

If automatic bootstrapping misses a player or saves weak crops, add or remove snapshots manually:

```bash
./venv/bin/python tools/player_embedding_ui.py \
  --host 127.0.0.1 \
  --port 8767
```

Open:

```text
http://127.0.0.1:8767/
```

After manual changes, rebuild embeddings:

```bash
./venv/bin/python tools/rebuild_player_embeddings_osnet.py \
  --input "$EMBEDDINGS" \
  --output "$EMBEDDINGS" \
  --backend soccernet-osnet \
  --device cpu
```

## 5. Run Fast Ball Tracking

Run fast volleyball ball tracking:

```bash
./venv/bin/python tools/run_fast_vball.py "$VIDEO" \
  --output-dir data/processed/fast_vball_raw
```

Expected output:

```text
$FAST_BALL_TRACK
```

## 6. Clean Ball Track

Create a cleaned ball-only debug video and cleaned JSON. This applies jump filtering and resets the ball state after missing or rejected detections.

```bash
./venv/bin/python tools/test_track_video.py "$VIDEO" \
  --players off \
  --ball-track "$BALL_TRACK" \
  --ball-source vball-net \
  --max-ball-gap 0 \
  --ball-max-jump 100 \
  --ball-reset-gap 5 \
  --team-filter none \
  --tracker iou \
  --ocr off \
  --reid off \
  --output-dir "$CLEAN_OUT_DIR"
```

Outputs:

```text
$CLEAN_OUT_DIR/${STEM}_annotated.mp4
$CLEAN_BALL_TRACK
$CLEAN_OUT_DIR/${STEM}_test_tracking.csv
```

Use `$CLEAN_BALL_TRACK` for the rest of the pipeline, not the raw `$BALL_TRACK`.

## 7. Label Nearest-Team Players

This labels the six near-side players using YOLO, ByteTrack, PaddleOCR, roster filling, missing-player prediction, and OSNet/SoccerNet ReID fallback.

```bash
./venv/bin/python tools/test_track_video.py "$VIDEO" \
  --embeddings "$EMBEDDINGS" \
  --ball-track "$CLEAN_BALL_TRACK" \
  --ball-source vball-net \
  --max-ball-gap 0 \
  --ball-max-jump 100 \
  --ball-reset-gap 5 \
  --team-filter court-nearest-6 \
  --fill-roster-labels \
  --predict-missing-players \
  --max-player-prediction-gap 45 \
  --tracker bytetrack \
  --frame-stride 1 \
  --device 0 \
  --reid auto \
  --embedding-device cpu \
  --ocr auto \
  --ocr-backend paddleocr \
  --ocr-device cpu \
  --roster "$ROSTER" \
  --ocr-languages en,ru \
  --ocr-min-confidence 0.85 \
  --ocr-relabel-min-confidence 0.92 \
  --ocr-skip-overlap-iou 0.25 \
  --reid-relabel-max-center-jump 100 \
  --ocr-every-n-frames 5 \
  --match-threshold 0 \
  --output-dir "$OUT_DIR"
```

Outputs:

```text
$OUT_DIR/${STEM}_annotated.mp4
$TRACKING_JSON
$OUT_DIR/${STEM}_test_tracking.csv
```

The default command uses `--ocr-languages en,ru` so PaddleOCR can read both Latin and Cyrillic shirt text.

## 8. Classify Serve, Receive, And Actions

This creates the final annotated rally video. It uses the cleaned ball track, the labeled player JSON, and the pose SVM classifier.

```bash
./venv/bin/python tools/classify_rally_serve_receive.py "$VIDEO" \
  --ball-track "$CLEAN_BALL_TRACK" \
  --tracking-json "$TRACKING_JSON" \
  --team-filter none \
  --output-dir data/processed/rally_classification \
  --pose-svm-model "$POSE_SVM" \
  --pose-model "$POSE_MODEL" \
  --pose-min-detection-confidence 0.20 \
  --receive-wait-prob-threshold 0.33 \
  --max-ball-gap 0 \
  --ball-max-jump 100 \
  --ball-reset-gap 5 \
  --serve-window 14 \
  --serve-min-speed 8 \
  --serve-min-distance 120 \
  --serve-max-mean-angle-change 38 \
  --reception-min-angle-change 90
```

Outputs:

```text
data/processed/rally_classification/${STEM}_serve_receive_annotated.mp4
data/processed/rally_classification/${STEM}_serve_receive.json
```

Use `--team-filter none` here because player filtering was already done in the player-labeling step.

## Notes

- `--ball-max-jump 100` rejects a ball detection if it jumps more than 100 pixels from the predicted position during continuous tracking.
- `--ball-reset-gap 5` resets ball state after 5 consecutive missing or rejected detections, allowing reacquisition.
- `--max-ball-gap 0` disables filling missing ball frames with predicted ball points.
- `--team-filter court-nearest-6` keeps six detected players near the marked near-side court.
- `--tracker bytetrack` gives more stable player tracks than frame-by-frame YOLO boxes.
- `--fill-roster-labels` tries to keep every roster player labeled.
- `--predict-missing-players` predicts a missing player's box from recent velocity.
- `--ocr-every-n-frames 5` runs OCR every 5 frames for better relabeling. Use larger values such as `15` or `30` when speed matters because PaddleOCR is heavy.
- `--ocr-min-confidence 0.85` filters weak OCR reads.
- `--ocr-relabel-min-confidence 0.92` requires stronger OCR before changing an existing player label.
- `--ocr-skip-overlap-iou 0.25` avoids OCR when player boxes overlap.
- `--reid auto` uses stored OSNet/SoccerNet embeddings as fallback.
- `--reid-relabel-max-center-jump 100` prevents ReID from moving a label too far from its smooth predicted position.
- Avoid `--uniform-color-filter` for the current setup; it was less stable than court filtering plus OCR/ReID/tracking.
- The final classifier marks action points at the ball direction-change point. If the ball is hidden during the change, it uses the midpoint between the last visible incoming frame and first visible outgoing frame for player-distance assignment.
