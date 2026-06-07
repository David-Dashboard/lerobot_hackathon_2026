"""Perception: open-vocabulary detection of trash items on the table.

Uses OWL-ViT (HF transformers) so "can, bottle, wrapper" works with no custom
training -- runs LOCALLY (no hosted API / venue WiFi). transformers + torch are
imported lazily so the rest of the package imports without them, and so tests can
exercise the pure ROI/geometry helpers without the model.

`Detection.table_xy` is filled in later by localize.Localizer.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Detection:
    label: str
    confidence: float
    pixel_xy: tuple[float, float]                 # bbox center (u, v)
    bbox: tuple[float, float, float, float]       # x1, y1, x2, y2
    table_xy: tuple[float, float] | None = None   # set by Localizer


def center_of(bbox) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def filter_roi(detections, roi: dict | None):
    """Keep only detections whose center is inside the pixel ROI (or all if no ROI).
    Excludes the arm/bin regions from being treated as trash."""
    if not roi:
        return list(detections)
    return [
        d for d in detections
        if roi["x_min"] <= d.pixel_xy[0] <= roi["x_max"]
        and roi["y_min"] <= d.pixel_xy[1] <= roi["y_max"]
    ]


class OwlVitDetector:
    """Lazy-loaded OWL-ViT detector. First `detect()` downloads/loads the model;
    cache the weights offline beforehand so it doesn't depend on WiFi at run time."""

    def __init__(
        self,
        classes,
        confidence: float = 0.25,
        device: str = "cpu",
        model_id: str = "google/owlvit-base-patch32",
    ):
        self.classes = list(classes)
        self.confidence = confidence
        self.device = device
        self.model_id = model_id
        self._processor = None
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch  # noqa: F401
            from transformers import OwlViTForObjectDetection, OwlViTProcessor
        except ImportError as e:  # pragma: no cover - depends on optional dep
            raise ImportError(
                "perception needs 'transformers' (and torch). Install it:\n"
                "  .\\.venv\\Scripts\\python.exe -m pip install transformers\n"
                f"(original error: {e})"
            ) from e
        self._processor = OwlViTProcessor.from_pretrained(self.model_id)
        self._model = OwlViTForObjectDetection.from_pretrained(self.model_id).to(self.device)
        self._model.eval()

    def detect(self, image_rgb) -> list[Detection]:
        """Run detection on an RGB numpy image (HxWx3) -> list[Detection]."""
        self._ensure_loaded()
        import torch
        from PIL import Image

        pil = Image.fromarray(image_rgb)
        inputs = self._processor(text=[self.classes], images=pil, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self._model(**inputs)

        target_sizes = torch.tensor([pil.size[::-1]], device=self.device)  # (h, w)
        # transformers >=5 renamed this to post_process_grounded_object_detection
        # (older versions used post_process_object_detection). Support both.
        if hasattr(self._processor, "post_process_grounded_object_detection"):
            results = self._processor.post_process_grounded_object_detection(
                outputs, threshold=self.confidence, target_sizes=target_sizes,
                text_labels=[self.classes],
            )[0]
        else:  # pragma: no cover - legacy transformers
            results = self._processor.post_process_object_detection(
                outputs, threshold=self.confidence, target_sizes=target_sizes
            )[0]

        # Labels may come back as strings ("text_labels") or class indices ("labels").
        labels = results.get("text_labels")
        if labels is None:
            labels = [self.classes[int(i)] for i in results["labels"]]

        detections: list[Detection] = []
        for score, label, box in zip(results["scores"], labels, results["boxes"]):
            bbox = tuple(float(v) for v in box.tolist())
            detections.append(
                Detection(
                    label=str(label),
                    confidence=float(score),
                    pixel_xy=center_of(bbox),
                    bbox=bbox,
                )
            )
        return detections
