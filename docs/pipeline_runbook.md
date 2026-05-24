# Volleyball Analysis Pipeline Runbook

This is the short command list for the current pipeline.

Example video used below:

```bash
VIDEO=data/game/video5224478607757318596_09cJizIV.mp4
STEM=video5224478607757318596_09cJizIV
```

## 1. Mark Court

Start the court marking UI:

```bash
./venv/bin/python tools/field_marking_ui.py --host 127.0.0.1 --port 8765
```

Open:

```text
http://127.0.0.1:8765/
```

Save output is written to:

```text
data/processed/calibrations/field_layout.json
data/processed/calibrations/field_layout_annotated.png
data/processed/calibrations/field_layout_source_frame.png
```

If port `8765` is busy, use another port, for example `8766`.

## 2. Make Player Snapshots

Start the player snapshot UI:

```bash
./venv/bin/python tools/player_embedding_ui.py --host 127.0.0.1 --port 8767
```

Open:

```text
http://127.0.0.1:8767/
```

Manually select player crops and assign player names, for example `player_1`, `player_2`.

Save output is written to:

```text
data/processed/player_embeddings/player_embeddings.json
data/processed/player_embeddings/snapshots/
```

## 3. Build OSNet Player Embeddings

After saving snapshots, rebuild embeddings using Torchreid OSNet:

```bash
./venv/bin/python tools/rebuild_player_embeddings_osnet.py --device cpu
```

This updates:

```text
data/processed/player_embeddings/player_embeddings.json
```

## 4. Run Ball Model

Run vball-net ball tracking:

```bash
./venv/bin/python tools/run_vball_net.py "$VIDEO" \
  --model-path external/vball-net/vb-models/VballNetFastV1_155_h288_w512.onnx \
  --output-dir data/processed/vball_net_raw
```

Output:

```text
data/processed/vball_net_raw/$STEM.csv
data/processed/vball_net_raw/$STEM.json
```

## 5. Label Video

Create a labeled video with ball trace and OSNet player labels:

```bash
./venv/bin/python tools/test_track_video.py "$VIDEO" \
  --ball-track "data/processed/vball_net_raw/$STEM.csv" \
  --team-filter largest \
  --embedding-device cpu \
  --match-threshold 0 \
  --output-dir data/processed/test_tracking_osnet
```

Output labeled video:

```text
data/processed/test_tracking_osnet/$STEM_annotated.mp4
```

Output data files:

```text
data/processed/test_tracking_osnet/$STEM_test_tracking.json
data/processed/test_tracking_osnet/$STEM_test_tracking.csv
```

## 6. Label Receive Moments From Ball Trace

To create a video with likely receive/contact moments from vball-net ball movement:

```bash
./venv/bin/python tools/label_receive_from_ball.py "$VIDEO" \
  --ball-track "data/processed/vball_net_raw/$STEM.csv" \
  --output-dir data/processed/receive_from_ball \
  --label-hold-sec 2 \
  --trail-length 45
```

Output labeled video:

```text
data/processed/receive_from_ball/$STEM_receive_from_ball_annotated.mp4
```

Output receive moments JSON:

```text
data/processed/receive_from_ball/$STEM_receive_from_ball.json
```

## Notes

- Use `--match-threshold 0` to force each stored player label onto the nearest detected player.
- Use `--match-threshold 1.05` to avoid labels when OSNet confidence is weak.
- Use `--team-filter largest` when the saved court polygon does not match the video size or orientation.
- If a command says a port is already in use, rerun it with a different `--port`.
