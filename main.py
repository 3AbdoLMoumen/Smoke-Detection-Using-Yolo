"""
Smoke Growth Detector
----------------------
Reads a video, runs YOLO smoke detection frame-by-frame, draws stylized red
bounding boxes, tracks the total detected smoke area per frame to quantify
growth, overlays a live growth graph (drawn with cv2.line) on the video,
saves the annotated video, and exports a final styled Plotly growth chart
with a fitted SVR trend line.

Requirements:
    pip install ultralytics opencv-python plotly kaleido scikit-learn

Usage:
    python smoke_growth_detector.py
    (edit the CONFIG section below, or import and call run())
"""

import cv2
import numpy as np
from ultralytics import YOLO
import plotly.graph_objects as go
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler


# ============================== CONFIG ==============================
MODEL_PATH = "best.pt"
INPUT_VIDEO = "video.mp4"
OUTPUT_VIDEO = "demo_fire_annotated.mp4"
OUTPUT_CHART_HTML = "smoke_growth_chart.html"
OUTPUT_CHART_PNG = "smoke_growth_chart.png"

CONF_THRESHOLD = 0.07
IOU_THRESHOLD = 0.5          # NMS overlap threshold (lower = merges/suppresses more)
MIN_BOX_AREA = 200           # px^2, ignore detections smaller than this (noise filter)

BOX_COLOR = (0, 0, 255)      # red, BGR
CORNER_LEN = 18              # length of HUD-style corner brackets (px)
BOX_THICKNESS = 2

GRAPH_PANEL_W = 320
GRAPH_PANEL_H = 140
GRAPH_MARGIN = 16
GRAPH_MAX_POINTS = 150       # how many recent points to show in the live overlay

# SVR fit parameters (final chart trend line)
SVR_KERNEL = "rbf"
SVR_C = 10
SVR_EPSILON = 0.05
# ======================================================================


def draw_stylized_box(frame, x1, y1, x2, y2, label=None, conf=None):
    """Draw a red sci-fi HUD style bounding box: thin rect + bold corner brackets."""
    color = BOX_COLOR
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1, lineType=cv2.LINE_AA)

    c = CORNER_LEN
    t = BOX_THICKNESS + 1
    corners = [
        ((x1, y1), (1, 1)),
        ((x2, y1), (-1, 1)),
        ((x1, y2), (1, -1)),
        ((x2, y2), (-1, -1)),
    ]
    for (cx, cy), (dx, dy) in corners:
        cv2.line(frame, (cx, cy), (cx + dx * c, cy), color, t, lineType=cv2.LINE_AA)
        cv2.line(frame, (cx, cy), (cx, cy + dy * c), color, t, lineType=cv2.LINE_AA)

    if label is not None:
        text = f"{label} {conf:.2f}" if conf is not None else label
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = y1 - 8 if y1 - 8 > th else y1 + th + 8
        cv2.rectangle(frame, (x1, ty - th - 6), (x1 + tw + 8, ty + 4), color, -1)
        cv2.putText(frame, text, (x1 + 4, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)


def draw_growth_graph(frame, area_history):
    """Draw a live line graph of area growth in the bottom-right corner using cv2.line."""
    h, w = frame.shape[:2]
    px0 = w - GRAPH_PANEL_W - GRAPH_MARGIN
    py0 = h - GRAPH_PANEL_H - GRAPH_MARGIN
    px1 = w - GRAPH_MARGIN
    py1 = h - GRAPH_MARGIN

    # semi-transparent background panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (px0, py0), (px1, py1), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.rectangle(frame, (px0, py0), (px1, py1), (200, 200, 200), 1, cv2.LINE_AA)

    cv2.putText(frame, "Smoke Area Growth", (px0 + 8, py0 + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    pts = area_history[-GRAPH_MAX_POINTS:]
    if len(pts) < 2:
        return

    max_val = max(max(pts), 1)  # avoid div by zero
    inner_x0, inner_y0 = px0 + 10, py0 + 28
    inner_x1, inner_y1 = px1 - 10, py1 - 10
    inner_w = inner_x1 - inner_x0
    inner_h = inner_y1 - inner_y0

    coords = []
    n = len(pts)
    for i, val in enumerate(pts):
        x = inner_x0 + int((i / max(n - 1, 1)) * inner_w)
        y = inner_y1 - int((val / max_val) * inner_h)
        coords.append((x, y))

    for i in range(1, len(coords)):
        cv2.line(frame, coords[i - 1], coords[i], BOX_COLOR, 2, cv2.LINE_AA)

    # current value readout
    cv2.putText(frame, f"{pts[-1]:.0f} px^2", (inner_x1 - 90, inner_y0 + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)


def run():
    model = YOLO(MODEL_PATH)
    print("#### MODEL LOADED")

    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {INPUT_VIDEO}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

    area_history = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)[0]

        total_area = 0.0
        if results.boxes is not None and len(results.boxes) > 0:
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = model.names[cls_id]

                area = max(0, x2 - x1) * max(0, y2 - y1)
                if area > MIN_BOX_AREA:
                    total_area += area
                    draw_stylized_box(frame, x1, y1, x2, y2, label=label, conf=conf)

        area_history.append(total_area)
        draw_growth_graph(frame, area_history)

        cv2.putText(frame, f"Frame {frame_idx}", (12, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"Annotated video saved to: {OUTPUT_VIDEO}")
    print(f"Processed {frame_idx} frames.")

    save_growth_chart(area_history, fps)
    print("#### VIDEO PROCESSED")


def save_growth_chart(area_history, fps):
    """Build and save a styled Plotly chart of smoke bounding-box area over time,
    with a fitted SVR trend line to show the underlying growth trend."""
    time_seconds = np.array([i / fps for i in range(len(area_history))])
    areas = np.array(area_history, dtype=float)

    fig = go.Figure()

    # raw detected area (noisy, per-frame)
    fig.add_trace(go.Scatter(
        x=time_seconds,
        y=areas,
        mode="lines",
        line=dict(color="red", width=1.5),
        fill="tozeroy",
        fillcolor="rgba(255,0,0,0.12)",
        name="Raw smoke bbox area",
        opacity=0.6,
    ))

    # --- fitted SVR trend line ---
    if len(time_seconds) >= 5:  # need a handful of points for a meaningful fit
        X = time_seconds.reshape(-1, 1)
        y = areas

        # SVR (RBF kernel) is scale-sensitive, so standardize both X and y before fitting
        x_scaler = StandardScaler()
        y_scaler = StandardScaler()
        X_scaled = x_scaler.fit_transform(X)
        y_scaled = y_scaler.fit_transform(y.reshape(-1, 1)).ravel()

        svr = SVR(kernel=SVR_KERNEL, C=SVR_C, gamma="scale", epsilon=SVR_EPSILON)
        svr.fit(X_scaled, y_scaled)

        # predict on a smooth, dense grid for a clean curve
        x_dense = np.linspace(time_seconds.min(), time_seconds.max(), 300).reshape(-1, 1)
        x_dense_scaled = x_scaler.transform(x_dense)
        y_pred_scaled = svr.predict(x_dense_scaled)
        y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

        fig.add_trace(go.Scatter(
            x=x_dense.ravel(),
            y=y_pred,
            mode="lines",
            line=dict(color="cyan", width=3),
            name="SVR fitted trend",
        ))
    else:
        print("Not enough points to fit SVR trend line, skipping.")

    fig.update_layout(
        title="Smoke Detection Area Growth Over Time",
        xaxis_title="Time (seconds)",
        yaxis_title="Total detected smoke area (px\u00b2)",
        template="plotly_dark",
        font=dict(family="Arial", size=14),
        plot_bgcolor="rgb(15,15,15)",
        paper_bgcolor="rgb(15,15,15)",
        title_font=dict(size=20, color="white"),
        margin=dict(l=60, r=40, t=70, b=50),
        legend=dict(bgcolor="rgba(0,0,0,0.3)"),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.1)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.1)")

    fig.write_html(OUTPUT_CHART_HTML)
    try:
        fig.write_image(OUTPUT_CHART_PNG, width=1200, height=600, scale=2)
    except Exception as e:
        print(f"Could not save static PNG (kaleido may be missing): {e}")

    print(f"Growth chart saved to: {OUTPUT_CHART_HTML}")


if __name__ == "__main__":
    run()
