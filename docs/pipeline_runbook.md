# Volleyball Analysis Pipeline Runbook

This document shows the current command sequence for processing one rally video.

The pipeline is:

1. Mark the near-side court.
2. Mark reception scoring zones.
3. Prepare roster and player embeddings.
4. Run fast volleyball ball tracking.
5. Clean the ball track.
6. Track every player, classify tracks by team, and resolve identities from complete tracklets.
7. Classify serve, receive, and technical actions with the pose SVM.

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

## 2. Mark Reception Scoring Zones

Run the reception zone marking UI:

```bash
./venv/bin/python tools/reception_zone_marking_ui.py --host 127.0.0.1 --port 8768
```

Open:

```text
http://127.0.0.1:8768/
```

Draw the zones where the pass should originate after reception:

- score `1`: perfect reception zone
- score `0.5`: acceptable reception zone

The UI writes:

```text
data/processed/calibrations/reception_zones.json
data/processed/calibrations/reception_zones_annotated.png
```

The final classifier draws these zones on the annotated video and uses the passer's bottom-center court point for scoring.

## 3. Prepare Roster

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

## 4. Bootstrap Jersey Snapshots And Embeddings

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

## 5. Optional Manual Player Snapshot Fixes

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

## 6. Run Fast Ball Tracking

Run fast volleyball ball tracking:

```bash
./venv/bin/python tools/run_fast_vball.py "$VIDEO" \
  --output-dir data/processed/fast_vball_raw
```

Expected output:

```text
$FAST_BALL_TRACK
```

## 7. Clean Ball Track

Create a cleaned ball-only debug video and cleaned JSON. This applies jump filtering and resets the ball state after missing or rejected detections.

```bash
./venv/bin/python tools/test_track_video.py "$VIDEO" \
  --players off \
  --ball-track "$BALL_TRACK" \
  --ball-source vball-net \
  --max-ball-gap-sec 0 \
  --ball-max-jump 100 \
  --ball-reset-gap-sec 0.17 \
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

## 8. Track All Players, Classify Teams, And Resolve Tracklets

This detects and tracks every person before making a team decision. Team classification and identity resolution run offline over complete tracklets, so strong evidence found later can label earlier frames. Near-team players have blue frames, opposing players have red frames, and insufficient evidence remains `unknown` instead of receiving a forced roster label.

The `sportsmix` tracker is a lightweight implementation of the association ideas in [SportsMOT](https://openaccess.thecvf.com/content/ICCV2023/html/Cui_SportsMOT_A_Large_Multi-Object_Tracking_Dataset_in_Multiple_Sports_Scenes_ICCV_2023_paper.html) and [MixSort](https://github.com/MCG-NJU/MixSort): predicted motion and IoU are fused with OSNet appearance, appearance templates are not updated from overlapped crops, and tracks survive short occlusions. It does not claim to reproduce MixSort's learned MixFormer model.

```bash
./venv/bin/python tools/test_track_video.py "$VIDEO" \
  --embeddings "$EMBEDDINGS" \
  --ball-track "$CLEAN_BALL_TRACK" \
  --ball-source vball-net \
  --max-ball-gap-sec 0 \
  --ball-max-jump 100 \
  --ball-reset-gap-sec 0.17 \
  --team-filter none \
  --track-all-players \
  --tracklet-team-classification \
  --tracklet-identity \
  --split-tracklets-on-appearance-change \
  --tracklet-split-color-distance 60 \
  --tracklet-split-min-observations 6 \
  --interpolate-track-gaps \
  --interpolate-track-gap-sec 0.75 \
  --tracker sportsmix \
  --sportsmix-max-age-sec 0.75 \
  --sportsmix-match-threshold 0.92 \
  --sportsmix-motion-weight 0.55 \
  --sportsmix-appearance-weight 0.35 \
  --sportsmix-center-weight 0.10 \
  --sportsmix-template-overlap-iou 0.25 \
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
  --ocr-every-n-frames 5 \
  --match-threshold 1.05 \
  --tracklet-reid-threshold 0.92 \
  --tracklet-reid-margin 0.06 \
  --tracklet-reid-min-confidence 0.12 \
  --output-dir "$OUT_DIR"
```

Outputs:

```text
$OUT_DIR/${STEM}_annotated.mp4
$TRACKING_JSON
$OUT_DIR/${STEM}_test_tracking.csv
```

The default command uses `--ocr-languages en,ru` so PaddleOCR can read both Latin and Cyrillic shirt text.

## 9. Classify Serve, Receive, And Actions

This creates the final annotated rally video. It uses the cleaned ball track, the labeled player JSON, and the pose SVM classifier.

```bash
./venv/bin/python tools/classify_rally_serve_receive.py "$VIDEO" \
  --ball-track "$CLEAN_BALL_TRACK" \
  --tracking-json "$TRACKING_JSON" \
  --reception-zones data/processed/calibrations/reception_zones.json \
  --team-filter classified-near \
  --output-dir data/processed/rally_classification \
  --pose-svm-model "$POSE_SVM" \
  --pose-model "$POSE_MODEL" \
  --pose-min-detection-confidence 0.20 \
  --receive-prob-threshold 0.33 \
  --receive-wait-prob-threshold 0.33 \
  --max-ball-gap-sec 0 \
  --ball-max-jump 100 \
  --ball-reset-gap-sec 0.17 \
  --serve-window-sec 0.47 \
  --reception-window-sec 0.13 \
  --reception-min-gap-sec 0.17 \
  --action-min-gap-sec 0.40 \
  --serve-min-speed 8 \
  --serve-min-distance 120 \
  --serve-max-mean-angle-change 38 \
  --reception-min-angle-change 90
```

Outputs:

```text
data/processed/rally_classification/${STEM}_serve_receive_annotated.mp4
data/processed/rally_classification/${STEM}_serve_receive.json
data/processed/rally_classification/${STEM}_reception_evaluation.json
```

`--team-filter classified-near` uses only tracklets classified as the near team for action-to-player assignment. The final video still draws opposing tracks in red.

## Volleydzen Tracking Ablation

The checked evaluation points are stored in `tests/fixtures/volleydzen_test_tracking_eval.json`: player 9's overhead reception at frame 40, player 8's set at frame 78, player 10's attack at frame 89, and three opposing-player boxes.

| Experiment | Correct action identities | Opponents marked red | Result |
| --- | ---: | ---: | --- |
| Legacy nearest-six + forced labels | 1/3 | 0/3 | Baseline |
| Track all people | 2/3 | 0/3 | Improved setter coverage |
| Then classify teams | 2/3 | 3/3 | Improved team separation |
| Then whole-track identity + interpolation | 2/3 | 3/3 | No additional contact-score gain; safer unknown/future propagation |
| SportsMOT-inspired association, no appearance split | 3/3 | 1/3 | Improved attack continuity, but one mixed tracklet remained |
| SportsMOT-inspired association + appearance change split | 3/3 | 3/3 | Best tested configuration |

Reproduce the scoring after running experiments:

```bash
./venv/bin/python tools/evaluate_tracking_experiment.py \
  final=data/processed/volleydzen_test_tracking_final/volleydzen_test_test_tracking.json \
  --config tests/fixtures/volleydzen_test_tracking_eval.json \
  --output data/processed/volleydzen_test_tracking_final/evaluation.json
```

## Notes

- `--ball-max-jump 100` rejects a ball detection if it jumps more than 100 pixels from the predicted position during continuous tracking.
- Temporal thresholds use seconds and are converted to frame counts from each video's measured FPS. This keeps equivalent behavior at 30, 60, and 120 FPS.
- `--ball-reset-gap-sec 0.17` resets ball state after approximately 0.17 seconds of missing or rejected detections, allowing reacquisition.
- `--max-ball-gap-sec 0` disables filling missing ball time with predicted ball points.
- `--reception-min-gap-sec 0.17` ignores trajectory changes immediately after the detected serve window. The default is about 5 frames at 30 FPS, 10 at 60 FPS, and 20 at 120 FPS.
- The old frame options such as `--ball-reset-gap`, `--max-ball-gap`, `--serve-window`, and `--reception-min-frame-gap` remain as deprecated explicit overrides for older commands.
- `--track-all-players` deliberately postpones team classification until after tracking.
- `--tracker sportsmix` fuses motion/IoU with OSNet appearance and freezes appearance-template updates during overlap.
- `--split-tracklets-on-appearance-change` cuts a track if association followed a different uniform through an overlap.
- `--tracklet-identity` assigns one identity from all clean OCR/ReID evidence in a tracklet and propagates it to earlier frames.
- Ambiguous tracklets remain `unknown`; roster labels are never filled by position alone.
- `--ocr-every-n-frames 5` runs OCR every 5 frames for better relabeling. Use larger values such as `15` or `30` when speed matters because PaddleOCR is heavy.
- `--ocr-min-confidence 0.85` filters weak OCR reads.
- `--ocr-relabel-min-confidence 0.92` requires stronger OCR before changing an existing player label.
- `--ocr-skip-overlap-iou 0.25` avoids OCR when player boxes overlap.
- `--reid auto` uses stored OSNet/SoccerNet embeddings as fallback.
- `--reid-relabel-max-center-jump 100` prevents ReID from moving a label too far from its smooth predicted position.
- Avoid `--uniform-color-filter` for the current setup; it was less stable than court filtering plus OCR/ReID/tracking.
- The final classifier marks action points at the ball direction-change point. If the ball is hidden during the change, it uses the midpoint between the last visible incoming frame and first visible outgoing frame for player-distance assignment.
- `--receive-prob-threshold 0.33` marks the first action with `receive_top` or `receive_bottom` probability above `0.33` as the reception.
- Reception score is saved to `${STEM}_reception_evaluation.json` and printed on the final video. A detected pass inside a score `1` zone gives `1`, inside a score `0.5` zone gives `0.5`, outside zones gives `0`, and missing pass gives `-1`.
