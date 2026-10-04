"""
Food Image Segmentation package (Phase 3).
"""
from src.segmentation.dataset import FoodSegmentationDataset, create_segmentation_fixture
from src.segmentation.evaluate import (
    calculate_segmentation_metrics,
    evaluate_segmenter_on_samples,
    save_segmentation_metrics,
)
from src.segmentation.inference import run_detection_and_segmentation
from src.segmentation.models import DiceBCELoss, FoodUNet
from src.segmentation.sam_utils import MaskType, SAMSegmenter
from src.segmentation.segmenter import FoodSegmenter
from src.segmentation.visualize import (
    create_mask_overlay,
    render_4_panel_comparison,
    save_segmentation_visualization,
)

__all__ = [
    "FoodSegmenter",
    "SAMSegmenter",
    "FoodUNet",
    "DiceBCELoss",
    "MaskType",
    "FoodSegmentationDataset",
    "create_segmentation_fixture",
    "calculate_segmentation_metrics",
    "evaluate_segmenter_on_samples",
    "save_segmentation_metrics",
    "run_detection_and_segmentation",
    "create_mask_overlay",
    "render_4_panel_comparison",
    "save_segmentation_visualization",
]
