# Volleyball Reception Quality Analysis Plan

## Goal

Build a pipeline that analyzes short volleyball rally clips and reports, for each near-side player reception:

- which player received the serve
- when the reception happened
- where the ball was passed
- how close the pass was to the setter target
- a reception quality score focused on outcome, not technique
- confidence and evidence such as annotated video, frame numbers, and court coordinates

The first version should handle only the near side of the court, filmed from behind the end line.

## Annotation-Minimized Strategy

Use existing models and tools first. Manual annotation should be limited to:

- one-time court calibration for each distinct camera setup, only if automatic court-line detection is not reliable
- review of low-confidence detections and scoring decisions
- a small targeted fine-tuning set only if the existing ball tracker fails on this footage

The default pipeline should process every clip automatically and mark uncertain cases for review instead of requiring labels up front.

## Core Assumptions

- Each input video contains one rally.
- The camera is fixed or mostly fixed from behind the near-side baseline.
- Only near-side players need to be tracked and evaluated.
- Reception quality means "how useful the pass was for the setter", not whether the receiver used correct body mechanics.
- The initial target can be heuristic-based, with confidence-driven review, then improved only where the automatic pipeline fails.

## Recommended Output

For each detected reception, save one row in a CSV/JSON table:

```text
video_id
receiver_track_id
receiver_player_id
contact_frame
contact_time_sec
ball_contact_x_m
ball_contact_y_m
pass_target_x_m
pass_target_y_m
setter_contact_frame
setter_contact_x_m
setter_contact_y_m
target_error_m
quality_score
quality_label
confidence
notes
```

Suggested quality scale:

```text
3 = perfect/good pass: setter can run normal offense with minimal movement
2 = playable positive pass: setter can set, but movement or options are limited
1 = poor but playable pass: setter must chase, only limited attack/free ball likely
0 = error or near-error: ace, overpass, unplayable, or no controlled set possible
```

## Existing Tools Baseline

### Ball Tracking: `asigatchov/vball-net`

Use [`asigatchov/vball-net`](https://github.com/asigatchov/vball-net) as the default ball tracker. Its README describes volleyball-specific tracking models based on TrackNetV4, including `VballNetV1` and the lighter `VballNetFastV1`. It also supports pretrained model downloads and ONNX inference.

Preferred first integration:

```bash
uv run src/inference_onnx.py \
  --video_path data/game/example.mp4 \
  --model_path vb-models/VballNetFastV1_155_h288_w512.onnx \
  --output_dir data/processed/vball_net_raw/
```

Implementation note: wrap this as a local adapter instead of mixing `vball-net` internals into the main analysis code. The adapter should convert `vball-net` predictions into the project ball-track schema.

Use this order:

1. Try `VballNetFastV1` first for speed.
2. Fall back to `VballNetV1` if the fast model misses too many contact frames.
3. Use the repository's frame annotation script only for targeted correction/fine-tuning, not for the full dataset.

### Player Tracking: Pretrained Person Detector + Multi-Object Tracker

Use a pretrained person detector first, such as Ultralytics YOLO person detection, then track detections with ByteTrack or BoT-SORT. This should require no custom player annotation for the first version.

The important constraint is the near-side court mask: generic person detectors will see far-side players and background people, so filter detections with the court homography before tracking or immediately after detection.

### Court Calibration: Reuse by Camera Setup

Do not manually calibrate every clip unless the camera moves. Prefer this order:

1. Reuse a saved homography for clips from the same camera position.
2. Attempt automatic line detection on a representative frame.
3. If needed, manually click court points once for the camera setup and save the result.

### Optional Pose Estimation

Pose models such as YOLO pose, RTMPose, or MediaPipe can help estimate which player contacted the ball, but they are optional. Do not use pose to judge reception technique; use it only as supporting evidence for contact assignment.

## Step-by-Step Implementation Plan

### 1. Inventory and Normalize the Video Dataset

1. Scan `data/game/*.mp4` and create a manifest with video path, duration, FPS, width, height, and frame count.
2. Assign a stable `video_id` to every clip.
3. Extract a few representative frames per video for visual inspection.
4. Store metadata in `data/processed/video_manifest.csv`.

Why this matters: later tracking and event detection should reference stable frame numbers and timestamps.

### 2. Define Court Coordinates

Use a real volleyball court coordinate system:

```text
x = court width, left-to-right from near-side camera view, 0 to 9 meters
y = near baseline to net, 0 to 9 meters for the near side
```

For the near side:

- near baseline: `y = 0`
- attack line: `y = 6`
- net: `y = 9`
- left sideline: `x = 0`
- right sideline: `x = 9`

Create a reusable calibration file per camera setup. If the camera position changes between clips, create a separate calibration entry for that video or setup:

```text
video_id
camera_setup_id
image_points: baseline corners, sideline/net corners, attack line points if visible
court_points: corresponding real-world court coordinates
homography_matrix
calibration_confidence
```

Avoid per-video manual point selection. If multiple clips share the same camera setup, calibrate one representative frame and reuse that homography. Add manual court clicks only as a fallback when automatic line detection or saved calibration is not good enough.

### 3. Mask to the Near Side Only

1. Use the court homography to define the near-side polygon in image space.
2. Ignore detections outside the near-side court plus a small margin.
3. Keep far-side players, spectators, and background objects out of the tracking pipeline.

This reduces false player assignments and simplifies the first version.

### 4. Detect and Track Near-Side Players

1. Run a pretrained person detector on every frame or every Nth frame.
2. Filter detections to the near-side court area with the court homography.
3. Track players through time using ByteTrack or BoT-SORT.
4. Store each track as bounding boxes plus court-projected foot positions.

Track record format:

```text
video_id
frame
track_id
bbox_x1
bbox_y1
bbox_x2
bbox_y2
foot_x_m
foot_y_m
confidence
```

For early code, `track_id` is enough. For player identity across clips, add jersey-number OCR or a small roster mapping later. Avoid manual player labeling until the per-clip analysis is working.

### 5. Detect and Track the Ball

Ball tracking is the hardest part because the ball is small, fast, blurred, and often occluded.

Recommended tool-first approach:

1. Run `vball-net` inference on every clip.
2. Convert its heatmap/prediction output into frame-level ball coordinates.
3. Link detections across frames into a ball trajectory.
4. Use motion smoothing or a Kalman filter to bridge short missed detections.
5. Reject impossible jumps using speed and direction constraints in court coordinates.
6. Save low-confidence and missing-contact windows for optional review.

Ball record format:

```text
video_id
frame
ball_x_px
ball_y_px
ball_x_m
ball_y_m
confidence
is_interpolated
source_model
source_confidence
```

Do not label ball positions up front. If `vball-net` is weak on this camera angle, collect only the failure windows around expected reception and setter contact, then fine-tune on that targeted subset.

### 6. Detect the Serve-Reception Event

For each rally:

1. Determine whether the near side is receiving. The reception candidate exists when the ball travels from far side toward near side and crosses the net.
2. Identify the first controlled near-side contact after the serve crosses into the near court.
3. Estimate contact frame using:
   - ball trajectory direction change
   - ball speed change
   - proximity between ball and near-side player bounding boxes
   - ball height proxy from image position and court location
4. Assign the receiver as the near-side player whose body/arms are nearest to the ball at the contact frame.

Fallback: export low-confidence cases for correction of `contact_frame` and `receiver_track_id`. High-confidence cases should flow through without manual review.

### 7. Detect the Setter Target or Setter Contact

Reception quality should be based on the pass outcome. Use two target definitions:

1. **Actual setter contact point**: where the setter contacts the received ball.
2. **Expected setter target zone**: a configurable ideal pass target if setter contact cannot be detected.

Recommended first-pass target:

```text
target_x_m = configurable by team/system, default around the setter zone
target_y_m = configurable, usually near the net but not too tight
```

Do not hard-code this forever. Different teams may want a different ideal location.

Detect setter contact by finding the next near-side ball contact after reception:

1. Search frames after reception contact.
2. Find another ball trajectory direction/speed change near a near-side player.
3. Assign that player as the setter candidate.
4. Store the setter contact court coordinate.

If setter contact is not reliable, use the ball location at the point where the pass reaches the setter zone, or use the configured target zone and score against it.

### 8. Engineer Reception Quality Features

Compute outcome features after reception:

```text
target_error_m = distance from setter contact/pass endpoint to ideal target
setter_movement_m = distance setter moved from ready position to contact
pass_tightness_m = distance from ball/setter contact to net
pass_width_error_m = left-right deviation from target
pass_depth_error_m = too short/too deep from target
time_to_setter_sec = time from reception contact to setter contact
overpass_flag = ball crosses back over net immediately after reception
unplayable_flag = no second near-side contact after reception
out_of_system_flag = setter contact far outside target zone
confidence = combined detector, tracker, and event confidence
```

These features measure pass usefulness without judging platform angle, footwork, or body posture.

### 9. Convert Features to a Quality Score

Start with a transparent rule-based scorer:

```text
score = 3 when:
  target_error_m <= 1.0
  pass is not too tight to the net
  setter movement is small
  second contact is controlled

score = 2 when:
  target_error_m <= 2.5
  setter can play the ball but must move
  attack options are somewhat limited

score = 1 when:
  ball is playable but far from target
  setter or another player must chase
  likely only free ball or predictable set

score = 0 when:
  reception error, ace, overpass, no controlled second contact, or ball is unplayable
```

Tune thresholds using a small reviewed subset and spot checks. Once enough reviewed examples accumulate naturally from low-confidence cases, train a small supervised model such as logistic regression, random forest, or gradient boosting over the engineered features.

### 10. Create a Confidence-Driven Review Loop

Automated labels will be imperfect, but the workflow should not require reviewing every clip.

For each clip, generate:

- annotated video with player tracks, ball track, contact frame, receiver, setter target, and score
- a small JSON/CSV file with all detected events
- optional still frames around contact: `contact_frame - 10` to `contact_frame + 10`

Automatically send only these cases to review:

- missing or low-confidence ball trajectory near expected reception
- multiple players close to the ball at the contact frame
- no detected second near-side contact after reception
- score near a threshold boundary
- overpass/error decisions
- random audit sample, for example 5-10% of high-confidence clips

Reviewer actions:

- confirm or correct receiver
- confirm or correct contact frame
- confirm or correct setter contact frame
- assign human quality score
- add notes for difficult cases

This review data becomes training data for later model improvement, but annotation volume stays proportional to uncertainty instead of total video count.

### 11. Evaluate Accuracy

Track separate metrics for each stage:

```text
player_detection_precision
player_detection_recall
player_track_id_switches
ball_detection_precision
ball_detection_recall
receiver_assignment_accuracy
contact_frame_error_frames
setter_contact_error_frames
quality_score_accuracy
quality_score_within_1_accuracy
```

A useful MVP target:

- receiver assignment accuracy above 85%
- reception contact within plus/minus 5 frames
- quality score within 1 point above 85%

Exact-score quality accuracy will be lower at first because volleyball pass ratings can be subjective.

### 12. Suggested Project Structure

```text
data/
  game/
    *.mp4
  processed/
    video_manifest.csv
    calibrations/
    tracks/
    ball_tracks/
    vball_net_raw/
    events/
    annotations/
    review_queue/

src/
  video_manifest.py
  court_calibration.py
  court_mask.py
  vball_net_adapter.py
  player_tracking.py
  ball_tracking.py
  event_detection.py
  reception_scoring.py
  review_export.py
  visualize.py

configs/
  court.yaml
  scoring.yaml
  model_paths.yaml
  tool_paths.yaml

reports/
  reception_quality_summary.csv
```

### 13. Implementation Phases

#### Phase 1: Tool-First MVP

1. Build video manifest generation.
2. Add camera-setup court calibration with reuse across clips.
3. Run `vball-net` ball tracking through `vball_net_adapter.py`.
4. Run pretrained person detection plus ByteTrack/BoT-SORT for near-side players.
5. Detect reception and setter contacts from ball trajectory changes plus player proximity.
6. Score receptions with rule-based thresholds.
7. Export CSV, JSON events, confidence values, and annotated clips.

Deliverable: a fully automatic first pass over all clips, with low-confidence cases flagged.

#### Phase 2: Confidence-Driven QA

1. Review only flagged cases and a small random audit sample.
2. Correct receiver, reception contact, setter contact, and quality label where needed.
3. Measure failure modes by video, camera setup, and event type.
4. Tune event-detection thresholds and scoring thresholds.

Deliverable: measurable automatic accuracy with a small review workload.

#### Phase 3: Targeted Fine-Tuning Only If Needed

1. If `vball-net` misses important contact frames, collect only those failure windows.
2. Use `vball-net` annotation tooling to label a compact fine-tuning set.
3. Fine-tune or swap between `VballNetFastV1` and `VballNetV1`.
4. Re-run the same evaluation set before adopting the tuned model.

Deliverable: improved ball tracking only if the pretrained model is not sufficient.

#### Phase 4: Player Identity Across Clips

1. Add jersey-number OCR or manual jersey mapping.
2. Merge per-rally track IDs into stable player IDs.
3. Produce per-player reception summaries.

Deliverable: player-level statistics across all clips.

#### Phase 5: Learned Quality Model

1. Use reviewed low-confidence cases plus audit samples as the labeled dataset.
2. Train a classifier/regressor using pass outcome features.
3. Calibrate model probabilities.
4. Keep the rule-based scorer as a fallback and sanity check.

Deliverable: quality score predictions that better match the coach/reviewer standard.

## Practical Scoring Configuration

Put scoring thresholds in a config file instead of code:

```yaml
ideal_target:
  x_m: 5.8
  y_m: 7.6

score_thresholds:
  perfect_target_error_m: 1.0
  positive_target_error_m: 2.5
  max_tight_to_net_m: 0.6
  max_setter_movement_for_perfect_m: 1.0
  max_contact_frame_search_gap: 90
```

These values are starting points only. Tune them after reviewing real clips.

## Main Risks and Mitigations

```text
Risk: Ball is too small or blurred.
Mitigation: use `vball-net`, full-resolution source frames where possible, trajectory smoothing, and targeted fine-tuning only on failure windows.

Risk: Camera perspective makes distances inaccurate.
Mitigation: use court homography, save calibration per camera setup, and reuse it across clips.

Risk: Player tracks swap during clustering or occlusion.
Mitigation: restrict to near side, use temporal smoothing, add jersey identity later.

Risk: Quality labels are subjective.
Mitigation: define a fixed 0-3 rubric, review threshold cases, and track scorer agreement on the audit sample.

Risk: Setter target differs by team rotation or system.
Mitigation: make target configurable and support per-rally override only when review identifies a real exception.
```

## First Code to Build

Build the first version in this order:

1. `video_manifest.py`: scan clips and write metadata.
2. `vball_net_adapter.py`: run/ingest `vball-net` predictions and write normalized ball tracks.
3. `court_calibration.py` and `court_mask.py`: create/reuse camera-setup homography and near-side mask.
4. `player_tracking.py`: run pretrained person detection plus ByteTrack/BoT-SORT and output near-side tracks.
5. `event_detection.py`: infer reception contact, receiver, setter contact, and confidence.
6. `reception_scoring.py`: score receptions from pass outcome features.
7. `visualize.py`: draw court, tracks, ball trajectory, contacts, and frame numbers on clips.
8. `review_export.py`: export only low-confidence cases and audit samples.

This order gives useful outputs early while keeping manual annotation as a fallback instead of a prerequisite.

## Final MVP Definition

The MVP is complete when a command can process the clips and produce:

```text
reports/reception_quality_summary.csv
data/processed/events/*.json
data/processed/annotations/*_annotated.mp4
```

Each reception row should include receiver, contact time, target error, quality score, and confidence. Low-confidence cases should be easy to review and correct.

## Implemented: Field Marking UI

A dependency-free local calibration UI is available at:

```bash
./venv/bin/python tools/field_marking_ui.py --host 127.0.0.1 --port 8765
```

Open `http://127.0.0.1:8765/`, select the MP4 clips from `data/game`, and click `Random Frame`. The browser chooses a random video and random timestamp, draws the frame on a canvas, and lets you mark the four near-side court boundary corners in this order:

```text
1. near-left boundary corner
2. near-right boundary corner
3. far-right boundary corner near net
4. far-left boundary corner near net
```

Click `Save Layout` to write:

```text
data/processed/calibrations/field_layout.json
data/processed/calibrations/field_layout_annotated.png
data/processed/calibrations/field_layout_source_frame.png
```

`field_layout.json` contains the reusable near-side boundary polygon in source-frame pixel coordinates. Because all clips come from the same stationary camera, this polygon should be used as the tracking mask for every video.

## Implemented: Player Snapshot Embedding UI

When jerseys do not have visible numbers, use a small manually curated identity set:

```bash
./venv/bin/python tools/player_embedding_ui.py --host 127.0.0.1 --port 8767
```

Open `http://127.0.0.1:8767/`, select all MP4 clips from `data/game`, choose how many random frames to sample from each video, and click `Build Queue`. For each sampled frame:

1. Add/select the current player ID.
2. Drag a box around that player.
3. Repeat until each player has several clear snapshots from different clips.
4. Click `Save Embeddings`.

The UI saves:

```text
data/processed/player_embeddings/player_embeddings.json
data/processed/player_embeddings/snapshots/<player_id>/*.png
```

The current embedding is a dependency-free browser descriptor built from crop color histograms and low-resolution luminance. It is intended as a bootstrap identity descriptor. Later, this can be replaced by a neural person ReID model while keeping the same `player_id`, snapshot, and sample metadata structure.

## Implemented: Video Analysis UI

The video analysis script combines:

- `vball-net` ball-track output
- YOLO person detection
- the saved near-side court mask
- the saved player embeddings

Run the browser interface with:

```bash
python3 tools/video_analysis_ui.py --host 127.0.0.1 --port 8768
```

Open `http://127.0.0.1:8768/`.

The underlying CLI is:

```bash
python3 tools/analyze_videos.py \
  --videos-dir data/game \
  --layout data/processed/calibrations/field_layout.json \
  --embeddings data/processed/player_embeddings/player_embeddings.json \
  --ball-tracks-dir data/processed/vball_net_raw \
  --output-dir data/processed/analysis \
  --yolo-model yolov8n.pt \
  --frame-stride 5
```

Required Python packages for automatic player detection:

```bash
python3 -m pip install opencv-python numpy ultralytics
```

Ball positions must either already exist under `data/processed/vball_net_raw`, or the UI/CLI must be given a `vball-net` command template that writes per-video ball tracks. The analyzer accepts ball-track JSON/CSV files named by video stem, for example:

```text
data/processed/vball_net_raw/video5224478607757318573.json
data/processed/vball_net_raw/video5224478607757318573.csv
data/processed/vball_net_raw/video5224478607757318573/ball_positions.json
data/processed/vball_net_raw/video5224478607757318573/ball_positions.csv
```

### vball-net Integration

The upstream repository is cloned at:

```text
external/vball-net
```

The project wrapper is:

```bash
python3 tools/run_vball_net.py data/game/video5224478607757318596_09cJizIV.mp4 \
  --model-path external/vball-net/vb-models/VballNetFastV1_155_h288_w512.onnx
```

It runs `external/vball-net/src/inference_onnx.py`, then normalizes the result into:

```text
data/processed/vball_net_raw/<video_id>.csv
data/processed/vball_net_raw/<video_id>.json
```

Required inference packages:

```bash
python3 -m pip install onnx onnxruntime pandas tqdm
```

The cloned `vball-net` repository does not include pretrained `.onnx` weights. Download one of the pretrained models listed in `external/vball-net/README.md` and place it here:

```text
external/vball-net/vb-models/VballNetFastV1_155_h288_w512.onnx
```

To debug one video using vball-net ball positions plus YOLO player labels:

```bash
python3 tools/test_track_video.py data/game/video5224478607757318596_09cJizIV.mp4 \
  --vball-model-path external/vball-net/vb-models/VballNetFastV1_155_h288_w512.onnx \
  --team-filter largest \
  --trail-length 300
```

To process all videos with the analyzer, set the video analysis UI field `Optional vball-net ONNX model path or command template` to:

```text
external/vball-net/vb-models/VballNetFastV1_155_h288_w512.onnx
```

Outputs:

```text
data/processed/analysis/detections_summary.csv
data/processed/analysis/per_video/<video_id>.json
```

## References

- [`asigatchov/vball-net`](https://github.com/asigatchov/vball-net): volleyball ball tracking baseline and pretrained model workflow.
