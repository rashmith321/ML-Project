"""
Food Classification package (Phase 4).
"""
from src.classification.classifier import ClassificationResult, FoodClassifier
from src.classification.dataset import FoodCropDataset, create_classification_fixture
from src.classification.evaluate import (
    compute_classification_metrics,
    evaluate_classifier_on_manifest,
    save_confusion_matrix_csv,
)
from src.classification.inference import run_full_pipeline
from src.classification.taxonomy import FOOD101_CLASSES, FoodTaxonomy
from src.classification.visualize import draw_classification_badge, plot_confusion_matrix

__all__ = [
    "ClassificationResult",
    "FoodClassifier",
    "FoodTaxonomy",
    "FOOD101_CLASSES",
    "FoodCropDataset",
    "create_classification_fixture",
    "compute_classification_metrics",
    "evaluate_classifier_on_manifest",
    "save_confusion_matrix_csv",
    "draw_classification_badge",
    "plot_confusion_matrix",
    "run_full_pipeline",
]
