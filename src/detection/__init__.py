"""
Food Object Detection package (Phase 2).
"""
from src.detection.detector import Detection, FoodDetector
from src.detection.evaluate import box_iou, compute_detection_metrics, evaluate_detector
from src.detection.inference import run_detection_inference
from src.detection.visualize import draw_detections, save_annotated_image

__all__ = [
    "Detection",
    "FoodDetector",
    "box_iou",
    "compute_detection_metrics",
    "evaluate_detector",
    "run_detection_inference",
    "draw_detections",
    "save_annotated_image",
]
