import cv2
import json
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("runs/detect/parking_model4/weights/best.pt")
# Load parking zones
with open("parking_zones.json") as f:
    zones = json.load(f)["zones"]

def check_overlap(box, zone):
    """
    Check if a detected object overlaps with a parking zone.
    Uses center point of the bounding box.
    """
    cx = (box[0] + box[2]) / 2
    cy = (box[1] + box[3]) / 2
    return zone["x1"] < cx < zone["x2"] and zone["y1"] < cy < zone["y2"]

def check_overlap_area(box, zone):
    """
    Check overlap using AREA instead of center point.
    More accurate for larger objects.
    """
    # Find intersection
    ix1 = max(box[0], zone["x1"])
    iy1 = max(box[1], zone["y1"])
    ix2 = min(box[2], zone["x2"])
    iy2 = min(box[3], zone["y2"])

    if ix2 < ix1 or iy2 < iy1:
        return False

    intersection = (ix2 - ix1) * (iy2 - iy1)
    zone_area = (zone["x2"] - zone["x1"]) * (zone["y2"] - zone["y1"])

    # If more than 20% of zone is covered → occupied
    overlap_ratio = intersection / zone_area
    return overlap_ratio > 0.5

def detect_parking(frame):
    """
    Run YOLO on a frame.
    Detects ANY object (not just cars) for better demo flexibility.
    Returns: list of zone statuses + annotated frame
    """
    # Detect ALL objects (no class filter)
    results = model(frame, verbose=False, conf=0.4)

    detected_boxes = []
    for result in results:
        for box in result.boxes.xyxy.tolist():
            detected_boxes.append(box)

    zone_status = []
    for zone in zones:
        # Use area-based overlap for better accuracy
        occupied = any(check_overlap_area(box, zone) for box in detected_boxes)
        status = "occupied" if occupied else "available"

        zone_status.append({
            "id": zone["id"],
            "status": status,
            "x1": zone["x1"],
            "y1": zone["y1"],
            "x2": zone["x2"],
            "y2": zone["y2"]
        })

        # Draw zone box
        color = (0, 0, 255) if occupied else (0, 255, 0)  # Red or Green
        cv2.rectangle(frame,
                      (zone["x1"], zone["y1"]),
                      (zone["x2"], zone["y2"]),
                      color, 3)

        # Draw status label with background
        label = f"{zone['id']}: {'🔴 Occupied' if occupied else '🟢 Free'}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        label_w, label_h = label_size

        # Background rectangle for text
        cv2.rectangle(frame,
                      (zone["x1"], zone["y1"] - label_h - 10),
                      (zone["x1"] + label_w + 8, zone["y1"]),
                      color, -1)

        # White text on colored background
        
        # Draw status label
        label = f"{zone['id']}: {'Occupied' if occupied else 'Free'}"
        cv2.putText(frame, label,
            (zone["x1"] + 4, zone["y1"] - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (255, 255, 255), 2)

    # Draw detected object boxes in BLUE for reference
    for box in detected_boxes:
        cv2.rectangle(frame,
                      (int(box[0]), int(box[1])),
                      (int(box[2]), int(box[3])),
                      (255, 165, 0), 2)  # Orange box = detected object

    # Draw overall summary on frame
    total = len(zone_status)
    occupied_count = sum(1 for z in zone_status if z["status"] == "occupied")
    available_count = total - occupied_count

    summary_text = f"Available: {available_count}  |  Occupied: {occupied_count}  |  Total: {total}"

    # Summary background bar at top
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 35), (0, 0, 0), -1)
    cv2.putText(frame, summary_text,
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2)

    return zone_status, frame


def get_summary(zone_status):
    """Returns occupancy summary dictionary."""
    total = len(zone_status)
    occupied = sum(1 for z in zone_status if z["status"] == "occupied")
    available = total - occupied
    return {
        "total": total,
        "occupied": occupied,
        "available": available,
        "occupancy_rate": round((occupied / total) * 100, 1) if total > 0 else 0
    }