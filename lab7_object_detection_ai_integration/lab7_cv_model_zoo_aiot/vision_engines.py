"""Vision engines for Lab 7 Model Zoo.

This file intentionally keeps all computer vision task logic in one place so that
students can compare different model families without jumping through many files.
Each function returns the same structure:
    annotated_frame, records, event

The code tries to use real libraries when installed. If a heavy optional package is
missing, it uses a lightweight fallback so the lab still runs and students can see
what kind of output each model family is expected to produce.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


# -----------------------------
# Shared helpers
# -----------------------------
@dataclass
class ZooState:
    previous_frame_gray: Optional[np.ndarray] = None
    tracking_objects: Dict[int, Tuple[int, int]] = field(default_factory=dict)
    tracking_next_id: int = 1
    count_in: int = 0
    count_out: int = 0
    last_centroid_y: Dict[int, int] = field(default_factory=dict)


COCO_DEMO_CLASSES = {
    0: "person",
    39: "bottle",
    63: "laptop",
    67: "cell phone",
    56: "chair",
    2: "car",
}


def resize_keep_width(frame: np.ndarray, width: int = 640) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= width:
        return frame.copy()
    ratio = width / float(w)
    return cv2.resize(frame, (width, int(h * ratio)))


def put_label(img: np.ndarray, text: str, org: Tuple[int, int], color=(0, 255, 0)) -> None:
    x, y = org
    cv2.rectangle(img, (x, y - 22), (x + max(80, len(text) * 8), y + 4), color, -1)
    cv2.putText(img, text, (x + 4, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1, cv2.LINE_AA)


def add_header(img: np.ndarray, title: str, subtitle: str = "") -> np.ndarray:
    canvas = img.copy()
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 34), (30, 30, 30), -1)
    cv2.putText(canvas, title, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    if subtitle:
        cv2.putText(canvas, subtitle, (min(canvas.shape[1] - 280, 260), 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    return canvas


def contour_boxes(frame: np.ndarray, min_area: int = 1200) -> List[Tuple[int, int, int, int, float]]:
    """Lightweight fallback detector using edge/contour information."""
    resized = resize_keep_width(frame, 640)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 60, 150)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w < 25 or h < 25:
            continue
        conf = min(0.85, 0.35 + area / (resized.shape[0] * resized.shape[1]))
        boxes.append((x, y, x + w, y + h, float(conf)))
    boxes = sorted(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)[:5]
    if not boxes:
        h, w = resized.shape[:2]
        boxes = [(w // 3, h // 4, 2 * w // 3, 3 * h // 4, 0.50)]
    return boxes


def event_from_records(task: str, records: List[Dict[str, Any]], severity: str = "NORMAL") -> Dict[str, Any]:
    return {
        "task": task,
        "event_type": f"{task.upper()}_RESULT",
        "severity": severity,
        "num_records": len(records),
        "explanation": f"{task} produced {len(records)} record(s).",
        "action_hint": "Display result on dashboard and review quality before using for automatic control.",
    }


# -----------------------------
# Task 1: YOLO / object detection
# -----------------------------
_YOLO_MODEL = None
_YOLO_MODEL_NAME = None


def _try_load_yolo(model_path: str = "yolov8n.pt"):
    global _YOLO_MODEL, _YOLO_MODEL_NAME
    if _YOLO_MODEL is not None and _YOLO_MODEL_NAME == model_path:
        return _YOLO_MODEL
    try:
        from ultralytics import YOLO  # type: ignore
        _YOLO_MODEL = YOLO(model_path)
        _YOLO_MODEL_NAME = model_path
        return _YOLO_MODEL
    except Exception:
        return None


def run_yolo_detection(frame: np.ndarray, conf: float = 0.35, classes: str = "", model_path: str = "yolov8n.pt"):
    start = time.perf_counter()
    annotated = resize_keep_width(frame, 640)
    model = _try_load_yolo(model_path)
    class_filter = [c.strip().lower() for c in classes.split(',') if c.strip()]
    records: List[Dict[str, Any]] = []
    engine_status = "real_yolo" if model is not None else "fallback_contour_detector"

    if model is not None:
        results = model.predict(source=annotated, conf=conf, imgsz=416, verbose=False)
        names = getattr(model, 'names', {})
        if results:
            for box in results[0].boxes:
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                score = float(box.conf[0].item())
                cls_id = int(box.cls[0].item())
                cls_name = str(names.get(cls_id, f"class_{cls_id}"))
                if class_filter and cls_name.lower() not in class_filter:
                    continue
                if score < conf:
                    continue
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 220, 0), 2)
                put_label(annotated, f"{cls_name} {score:.2f}", (x1, max(24, y1)), (0, 220, 0))
                records.append({
                    "task": "detection", "class_name": cls_name, "confidence": round(score, 3),
                    "bbox": [x1, y1, x2, y2], "engine": engine_status
                })
    else:
        demo_names = ["person", "bottle", "cell phone", "laptop", "object_demo"]
        for i, (x1, y1, x2, y2, score) in enumerate(contour_boxes(annotated)):
            cls_name = demo_names[i % len(demo_names)]
            if class_filter and cls_name.lower() not in class_filter:
                continue
            if score < conf:
                continue
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 220, 0), 2)
            put_label(annotated, f"{cls_name} {score:.2f}", (x1, max(24, y1)), (0, 220, 0))
            records.append({
                "task": "detection", "class_name": cls_name, "confidence": round(score, 3),
                "bbox": [x1, y1, x2, y2], "engine": engine_status
            })
    elapsed_ms = (time.perf_counter() - start) * 1000
    annotated = add_header(annotated, "Object Detection", f"{engine_status} | {elapsed_ms:.1f} ms")
    event = event_from_records("detection", records, "WARNING" if records else "NORMAL")
    return annotated, records, event


# -----------------------------
# Task 2: Tracking / counting
# -----------------------------
def _assign_tracking_ids(boxes: List[Tuple[int, int, int, int, float]], state: ZooState) -> List[Dict[str, Any]]:
    records = []
    for x1, y1, x2, y2, score in boxes[:4]:
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        best_id = None
        best_dist = 10**9
        for obj_id, (px, py) in state.tracking_objects.items():
            dist = math.hypot(cx - px, cy - py)
            if dist < best_dist and dist < 90:
                best_id, best_dist = obj_id, dist
        if best_id is None:
            best_id = state.tracking_next_id
            state.tracking_next_id += 1
        state.tracking_objects[best_id] = (cx, cy)
        records.append({"track_id": best_id, "bbox": [x1, y1, x2, y2], "centroid": [cx, cy], "confidence": round(score, 3)})
    return records


def run_tracking_counting(frame: np.ndarray, state: ZooState, line_ratio: float = 0.55):
    start = time.perf_counter()
    annotated = resize_keep_width(frame, 640)
    boxes = contour_boxes(annotated, min_area=1000)
    records = _assign_tracking_ids(boxes, state)
    h, w = annotated.shape[:2]
    line_y = int(h * line_ratio)
    cv2.line(annotated, (0, line_y), (w, line_y), (0, 180, 255), 2)
    for rec in records:
        x1, y1, x2, y2 = rec["bbox"]
        cx, cy = rec["centroid"]
        tid = rec["track_id"]
        prev_y = state.last_centroid_y.get(tid)
        if prev_y is not None:
            if prev_y < line_y <= cy:
                state.count_in += 1
            elif prev_y > line_y >= cy:
                state.count_out += 1
        state.last_centroid_y[tid] = cy
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 180, 0), 2)
        cv2.circle(annotated, (cx, cy), 4, (0, 0, 255), -1)
        put_label(annotated, f"ID {tid}", (x1, max(24, y1)), (255, 180, 0))
    cv2.putText(annotated, f"Count down: {state.count_in} | up: {state.count_out}", (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 255), 2)
    elapsed_ms = (time.perf_counter() - start) * 1000
    annotated = add_header(annotated, "Tracking / Counting", f"centroid demo | {elapsed_ms:.1f} ms")
    for rec in records:
        rec.update({"task": "tracking_counting", "count_in": state.count_in, "count_out": state.count_out, "engine": "centroid_tracker_demo"})
    event = event_from_records("tracking_counting", records, "WARNING" if (state.count_in + state.count_out) else "NORMAL")
    return annotated, records, event


# -----------------------------
# Task 3: Pose landmark
# -----------------------------
def run_pose_landmark(frame: np.ndarray, min_conf: float = 0.5):
    start = time.perf_counter()
    annotated = resize_keep_width(frame, 640)
    records: List[Dict[str, Any]] = []
    h, w = annotated.shape[:2]
    # Lightweight demo skeleton: enough to illustrate landmark output without forcing MediaPipe install.
    pts = {
        "nose": (w//2, h//5), "left_shoulder": (w//2-70, h//3), "right_shoulder": (w//2+70, h//3),
        "left_elbow": (w//2-115, h//2), "right_elbow": (w//2+115, h//2),
        "left_wrist": (w//2-145, h//2+65), "right_wrist": (w//2+145, h//2+65),
        "left_hip": (w//2-45, 2*h//3), "right_hip": (w//2+45, 2*h//3),
        "left_knee": (w//2-50, 5*h//6), "right_knee": (w//2+50, 5*h//6),
    }
    edges = [("nose","left_shoulder"),("nose","right_shoulder"),("left_shoulder","right_shoulder"),("left_shoulder","left_elbow"),("left_elbow","left_wrist"),("right_shoulder","right_elbow"),("right_elbow","right_wrist"),("left_shoulder","left_hip"),("right_shoulder","right_hip"),("left_hip","right_hip"),("left_hip","left_knee"),("right_hip","right_knee")]
    for a,b in edges:
        cv2.line(annotated, pts[a], pts[b], (0, 255, 255), 2)
    for name, p in pts.items():
        cv2.circle(annotated, p, 5, (0, 0, 255), -1)
        records.append({"task": "pose", "landmark": name, "x": p[0], "y": p[1], "confidence": min_conf, "engine": "mediapipe_pose_fallback_demo"})
    elapsed_ms = (time.perf_counter() - start) * 1000
    annotated = add_header(annotated, "Pose Landmark", f"fallback landmarks | {elapsed_ms:.1f} ms")
    event = event_from_records("pose", records, "NORMAL")
    return annotated, records, event


# -----------------------------
# Task 4: Hand / gesture
# -----------------------------
def run_hand_gesture(frame: np.ndarray, min_conf: float = 0.5):
    start = time.perf_counter()
    annotated = resize_keep_width(frame, 640)
    h, w = annotated.shape[:2]
    center = (w//2, h//2)
    records = []
    # Demo hand: palm + five fingers landmarks.
    palm = [(center[0]-35, center[1]+20), (center[0]+35, center[1]+20), (center[0]+45, center[1]-25), (center[0]-45, center[1]-25)]
    cv2.polylines(annotated, [np.array(palm, np.int32)], True, (255, 0, 255), 2)
    fingertips = [(center[0]-55, center[1]-95), (center[0]-25, center[1]-120), (center[0]+5, center[1]-130), (center[0]+35, center[1]-115), (center[0]+65, center[1]-85)]
    for i, tip in enumerate(fingertips):
        cv2.line(annotated, center, tip, (255, 0, 255), 2)
        cv2.circle(annotated, tip, 5, (0, 255, 255), -1)
        records.append({"task": "hand_gesture", "landmark": f"finger_{i+1}", "x": tip[0], "y": tip[1], "confidence": min_conf, "engine": "mediapipe_hand_fallback_demo"})
    put_label(annotated, "open_palm_demo", (center[0]-75, center[1]+75), (255, 0, 255))
    records.append({"task": "hand_gesture", "gesture": "open_palm_demo", "confidence": min_conf, "engine": "mediapipe_gesture_fallback_demo"})
    elapsed_ms = (time.perf_counter() - start) * 1000
    annotated = add_header(annotated, "Hand / Gesture", f"fallback gesture | {elapsed_ms:.1f} ms")
    event = event_from_records("hand_gesture", records, "NORMAL")
    return annotated, records, event


# -----------------------------
# Task 5: Face landmarks / attention cue
# -----------------------------
def run_face_landmark(frame: np.ndarray, min_conf: float = 0.5):
    start = time.perf_counter()
    annotated = resize_keep_width(frame, 640)
    h, w = annotated.shape[:2]
    cx, cy = w//2, h//3
    axes = (70, 90)
    cv2.ellipse(annotated, (cx, cy), axes, 0, 0, 360, (0, 180, 255), 2)
    points = {
        "left_eye": (cx-28, cy-22), "right_eye": (cx+28, cy-22), "nose": (cx, cy+10),
        "mouth_left": (cx-30, cy+48), "mouth_right": (cx+30, cy+48), "chin": (cx, cy+82)
    }
    records = []
    for name, p in points.items():
        cv2.circle(annotated, p, 4, (0, 0, 255), -1)
        records.append({"task": "face_landmark", "landmark": name, "x": p[0], "y": p[1], "confidence": min_conf, "engine": "face_landmark_fallback_demo"})
    put_label(annotated, "face_attention_demo", (cx-75, cy+120), (0, 180, 255))
    elapsed_ms = (time.perf_counter() - start) * 1000
    annotated = add_header(annotated, "Face Landmark", f"fallback face mesh | {elapsed_ms:.1f} ms")
    event = event_from_records("face_landmark", records, "NORMAL")
    return annotated, records, event


# -----------------------------
# Task 6: OCR
# -----------------------------
def run_ocr(frame: np.ndarray, text_conf: float = 0.5):
    start = time.perf_counter()
    annotated = resize_keep_width(frame, 640)
    records: List[Dict[str, Any]] = []
    engine = "fallback_ocr_demo"
    try:
        import easyocr  # type: ignore
        reader = easyocr.Reader(['en'], gpu=False)
        result = reader.readtext(annotated)
        engine = "easyocr"
        for bbox, text, score in result[:8]:
            pts = np.array(bbox, dtype=np.int32)
            cv2.polylines(annotated, [pts], True, (0, 255, 255), 2)
            x, y = int(pts[:,0].min()), int(pts[:,1].min())
            put_label(annotated, f"{text} {score:.2f}", (x, max(24,y)), (0,255,255))
            records.append({"task":"ocr", "text": text, "confidence": round(float(score),3), "bbox": pts.tolist(), "engine": engine})
    except Exception:
        h, w = annotated.shape[:2]
        x1, y1, x2, y2 = w//4, h//3, 3*w//4, h//3+70
        cv2.rectangle(annotated, (x1,y1), (x2,y2), (0,255,255), 2)
        cv2.putText(annotated, "AIOT LAB 7", (x1+12, y1+45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2, cv2.LINE_AA)
        put_label(annotated, "OCR demo text", (x1, y1), (0,255,255))
        records.append({"task":"ocr", "text": "AIOT LAB 7 DEMO", "confidence": text_conf, "bbox": [x1,y1,x2,y2], "engine": engine})
    elapsed_ms = (time.perf_counter() - start) * 1000
    annotated = add_header(annotated, "OCR", f"{engine} | {elapsed_ms:.1f} ms")
    event = event_from_records("ocr", records, "NORMAL")
    return annotated, records, event


# -----------------------------
# Task 7: Segmentation
# -----------------------------
def run_segmentation(frame: np.ndarray, alpha: float = 0.35):
    start = time.perf_counter()
    annotated = resize_keep_width(frame, 640)
    gray = cv2.cvtColor(annotated, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # Pick the larger of mask/inverse foreground area not covering whole frame.
    if np.mean(mask) > 160:
        mask = cv2.bitwise_not(mask)
    color = np.zeros_like(annotated)
    color[:,:,1] = 180
    overlay = annotated.copy()
    overlay[mask > 0] = cv2.addWeighted(annotated, 1-alpha, color, alpha, 0)[mask > 0]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    records=[]
    if contours:
        c = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(c))
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(overlay, (x,y), (x+w,y+h), (0,255,0), 2)
        put_label(overlay, f"mask area {int(area)}", (x, max(24,y)), (0,255,0))
        records.append({"task":"segmentation", "mask_area": int(area), "bbox":[x,y,x+w,y+h], "engine":"opencv_threshold_segmentation_demo"})
    elapsed_ms = (time.perf_counter() - start) * 1000
    annotated = add_header(overlay, "Segmentation", f"OpenCV demo mask | {elapsed_ms:.1f} ms")
    event = event_from_records("segmentation", records, "NORMAL")
    return annotated, records, event


# -----------------------------
# Task 8: Motion detection
# -----------------------------
def run_opencv_motion(frame: np.ndarray, state: ZooState, motion_threshold: int = 25, min_area: int = 800):
    start = time.perf_counter()
    annotated = resize_keep_width(frame, 640)
    gray = cv2.cvtColor(annotated, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9,9), 0)
    records=[]
    if state.previous_frame_gray is None:
        state.previous_frame_gray = gray
        event = {"task":"opencv_motion", "event_type":"MOTION_INITIALIZED", "severity":"NORMAL", "num_records":0, "explanation":"Background frame initialized.", "action_hint":"Move object in front of camera and observe motion mask."}
        return add_header(annotated, "OpenCV Motion", "background initialized"), records, event
    diff = cv2.absdiff(state.previous_frame_gray, gray)
    _, thresh = cv2.threshold(diff, motion_threshold, 255, cv2.THRESH_BINARY)
    thresh = cv2.dilate(thresh, np.ones((5,5),np.uint8), iterations=2)
    contours,_ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(annotated, (x,y), (x+w,y+h), (0,0,255), 2)
        records.append({"task":"opencv_motion", "motion_area": int(area), "bbox":[x,y,x+w,y+h], "engine":"frame_difference"})
    state.previous_frame_gray = gray
    elapsed_ms = (time.perf_counter()-start)*1000
    annotated = add_header(annotated, "OpenCV Motion", f"frame diff | {elapsed_ms:.1f} ms")
    event = event_from_records("opencv_motion", records, "WARNING" if records else "NORMAL")
    return annotated, records, event


TASKS = {
    "detection": run_yolo_detection,
    "tracking_counting": run_tracking_counting,
    "pose_landmark": run_pose_landmark,
    "hand_gesture": run_hand_gesture,
    "face_landmark": run_face_landmark,
    "ocr": run_ocr,
    "segmentation": run_segmentation,
    "opencv_motion": run_opencv_motion,
}


def run_task(task: str, frame: np.ndarray, state: Optional[ZooState] = None, **params):
    state = state or ZooState()
    if task == "detection":
        return run_yolo_detection(frame, conf=float(params.get('conf', 0.35)), classes=str(params.get('classes', '')), model_path=str(params.get('model_path', 'yolov8n.pt')))
    if task == "tracking_counting":
        return run_tracking_counting(frame, state, line_ratio=float(params.get('line_ratio', 0.55)))
    if task == "pose_landmark":
        return run_pose_landmark(frame, min_conf=float(params.get('min_conf', 0.5)))
    if task == "hand_gesture":
        return run_hand_gesture(frame, min_conf=float(params.get('min_conf', 0.5)))
    if task == "face_landmark":
        return run_face_landmark(frame, min_conf=float(params.get('min_conf', 0.5)))
    if task == "ocr":
        return run_ocr(frame, text_conf=float(params.get('text_conf', 0.5)))
    if task == "segmentation":
        return run_segmentation(frame, alpha=float(params.get('alpha', 0.35)))
    if task == "opencv_motion":
        return run_opencv_motion(frame, state, motion_threshold=int(params.get('motion_threshold', 25)), min_area=int(params.get('min_area', 800)))
    annotated = add_header(resize_keep_width(frame, 640), "Unknown task", task)
    return annotated, [], {"task": task, "event_type": "UNKNOWN_TASK", "severity": "ERROR", "num_records":0, "explanation":"Unknown task", "action_hint":"Choose a valid task."}
