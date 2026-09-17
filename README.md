# Smoke Detection and Growth Quantification Using YOLO

## Overview

This project implements a video-based smoke detection and growth analysis pipeline built upon the YOLO (You Only Look Once) object detection framework. The system processes an input video frame by frame, detects regions of smoke, quantifies the aggregate detected smoke area over time, and produces two primary outputs: an annotated video with stylized bounding boxes and a live growth overlay, and a post-processed analytical chart depicting the temporal evolution of smoke area with a fitted trend line.

The objective of this pipeline is to provide both a qualitative visual assessment (via annotated video) and a quantitative growth analysis (via the exported chart) suitable for research, monitoring, or early-warning applications related to fire and smoke development.

## Methodology

### Detection

Smoke detection is performed using a YOLO model (via the `ultralytics` library) loaded from a user-supplied weights file. For each video frame, the model generates bounding box predictions, which are filtered according to a configurable confidence threshold and a minimum area threshold to reduce the influence of spurious, low-confidence, or negligible detections.

### Area Quantification

For each frame, the total detected smoke area is computed as the sum of the pixel areas of all bounding boxes exceeding the minimum area threshold. This aggregate value is recorded across all frames to construct a time series representing the growth of detected smoke over the duration of the video.

### Visualization

Two complementary visualizations are produced:

1. **Real-time overlay**: Each output frame includes stylized bounding boxes (rendered with corner-bracket, heads-up-display styling) around detected smoke regions, along with a live line graph embedded in the frame that displays recent smoke area values, drawn using OpenCV primitives.
2. **Post-processing chart**: Following completion of video processing, a chart is generated using Plotly, displaying the raw per-frame smoke area alongside a smoothed trend line. The trend line is obtained via Support Vector Regression (SVR) with a radial basis function (RBF) kernel, fitted to the standardized time series to characterize the underlying growth trend independent of frame-to-frame noise.

## Repository Structure

```
.
├── main.py     # Primary detection, annotation, and analysis script
└── README.md
```

## Requirements

The following Python packages are required:

- `ultralytics`
- `opencv-python`
- `plotly`
- `kaleido`
- `scikit-learn`
- `numpy`

These dependencies may be installed via:

```bash
pip install ultralytics opencv-python plotly kaleido scikit-learn numpy
```

## Configuration

Pipeline parameters are defined in the configuration section at the top of `main.py` and may be modified as needed:

| Parameter | Description |
|---|---|
| `MODEL_PATH` | Path to the trained YOLO weights file (e.g., `best.pt`) |
| `INPUT_VIDEO` | Path to the source video for analysis |
| `OUTPUT_VIDEO` | Destination path for the annotated output video |
| `OUTPUT_CHART_HTML` | Destination path for the interactive HTML growth chart |
| `OUTPUT_CHART_PNG` | Destination path for the static PNG growth chart |
| `CONF_THRESHOLD` | Minimum detection confidence considered valid |
| `IOU_THRESHOLD` | Intersection-over-union threshold used for non-maximum suppression |
| `MIN_BOX_AREA` | Minimum bounding box area (in pixels squared) considered valid |
| `BOX_COLOR`, `CORNER_LEN`, `BOX_THICKNESS` | Visual parameters for bounding box rendering |
| `GRAPH_PANEL_W`, `GRAPH_PANEL_H`, `GRAPH_MARGIN`, `GRAPH_MAX_POINTS` | Visual parameters for the live overlay graph |
| `SVR_KERNEL`, `SVR_C`, `SVR_EPSILON` | Parameters governing the Support Vector Regression trend fit |

## Usage

1. Place the trained YOLO weights file and the target input video in the working directory, or update `MODEL_PATH` and `INPUT_VIDEO` accordingly.
2. Execute the script:

```bash
python main.py
```

3. Upon completion, the following outputs will be generated:
   - An annotated video file (`demo_fire_annotated.mp4` by default) containing bounding box overlays and a live growth graph.
   - An interactive HTML chart (`smoke_growth_chart.html`) and a static PNG chart (`smoke_growth_chart.png`) depicting smoke area growth over time with a fitted trend line.

Alternatively, the pipeline may be invoked programmatically:

```python
from main import run
run()
```

## Output Description

### Annotated Video

Each frame of the output video contains:

- Red, HUD-style bounding boxes around detected smoke regions, labeled with class name and confidence score.
- A semi-transparent panel in the lower-right corner displaying a live line graph of recent smoke area measurements.
- A frame counter in the upper-left corner.

### Growth Chart

The exported chart presents:

- The raw, per-frame detected smoke area as a function of time, rendered as a shaded area plot.
- A smoothed trend line obtained via SVR regression, illustrating the underlying growth pattern independent of frame-level noise.

## Notes and Limitations

- Detection accuracy is contingent upon the quality and domain relevance of the supplied YOLO weights.
- The SVR trend line is computed only when a minimum of five data points is available; videos of insufficient length or frame count will produce the raw area plot without a fitted trend.
- Static PNG export via Kaleido requires a functioning Kaleido installation; failure to export the PNG does not interrupt generation of the HTML chart.

## License

No license has been specified for this repository. Users should consult the repository owner regarding permitted use, modification, and distribution of this code.
