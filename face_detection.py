"""
Детекция лиц в кадре — MediaPipe BlazeFace (short range).

Файл модели models/blaze_face_short_range.tflite скачивается автоматически
при первом вызове load_face_detector() и в git не хранится.
"""

import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp

MODEL_PATH = Path(__file__).resolve().parent / "models" / "blaze_face_short_range.tflite"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)


def ensure_model() -> Path:
    """Скачивает модель BlazeFace (~225 КБ), если её ещё нет на диске."""
    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


def load_face_detector(min_detection_confidence: float = 0.5):
    """Создаёт детектор лиц BlazeFace."""
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    base_options = mp_python.BaseOptions(model_asset_path=str(ensure_model()))
    options = mp_vision.FaceDetectorOptions(
        base_options=base_options,
        min_detection_confidence=min_detection_confidence,
    )
    return mp_vision.FaceDetector.create_from_options(options)


def detect_all_faces(frame_bgr, detector) -> list[tuple[int, int, int, int]]:
    """
    Находит все лица на кадре (BGR).
    Возвращает список (x, y, w, h), отсортированный по площади:
    самое крупное лицо — первое (считается «главным»).
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = detector.detect(mp_image)

    if not result.detections:
        return []

    h_img, w_img = frame_bgr.shape[:2]
    faces = []
    for detection in result.detections:
        bbox = detection.bounding_box
        x = max(0, bbox.origin_x)
        y = max(0, bbox.origin_y)
        w = min(bbox.width, w_img - x)
        h = min(bbox.height, h_img - y)
        if w > 0 and h > 0:
            faces.append((x, y, w, h))

    faces.sort(key=lambda f: f[2] * f[3], reverse=True)
    return faces
