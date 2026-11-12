"""
Распознавание эмоций по изображению лица.

Единая точка входа для всех моделей проекта: get_emotions().
Поддерживаются две модели:
  - "deepface" — CNN из библиотеки DeepFace (обучена на FER2013);
  - "vit"      — Vision Transformer trpakov/vit-face-expression (HuggingFace).

Обе модели приводятся к одному набору из 6 классов (как в датасете CREMA-D):
angry, disgust, fear, happy, sad, neutral. Класс "surprise" отбрасывается,
а оставшиеся вероятности перенормируются так, чтобы в сумме давать 100.
"""

from functools import lru_cache
from time import perf_counter

import cv2
import numpy as np

# Порядок классов фиксирован — его используют графики, отчёты и метрики.
EMOTIONS = ("angry", "disgust", "fear", "happy", "sad", "neutral")
MODELS = ("deepface", "vit")

VIT_MODEL_ID = "trpakov/vit-face-expression"

# Возможные варианты написания меток у разных моделей -> наше название.
_LABEL_ALIASES = {
    "anger": "angry",
    "happiness": "happy",
    "sadness": "sad",
    "disgusted": "disgust",
    "fearful": "fear",
}


def _empty_result() -> dict:
    return {"emotions": {}, "dominant_emotion": "", "inference_time": 0.0}


def _normalize(raw: dict) -> dict:
    """
    Приводит ответ модели к 6 классам EMOTIONS в процентах (сумма = 100).
    Неизвестные метки и "surprise" отбрасываются.
    """
    scores = {e: 0.0 for e in EMOTIONS}
    for label, value in raw.items():
        key = str(label).strip().lower()
        key = _LABEL_ALIASES.get(key, key)
        if key in scores:
            scores[key] += float(value)

    total = sum(scores.values())
    if total <= 0:
        return {}
    return {e: scores[e] * 100.0 / total for e in EMOTIONS}


@lru_cache(maxsize=1)
def load_vit_model():
    """Загружает ViT-пайплайн один раз на процесс (~350 МБ при первом запуске)."""
    from transformers import pipeline
    return pipeline("image-classification", model=VIT_MODEL_ID)


def _run_deepface(image, is_face_crop: bool) -> dict:
    from deepface import DeepFace
    # Если лицо уже вырезано внешним детектором — не ищем его повторно.
    result = DeepFace.analyze(
        img_path=image,
        actions=["emotion"],
        enforce_detection=not is_face_crop,
        detector_backend="skip" if is_face_crop else "opencv",
        silent=True,
    )
    return result[0]["emotion"]


def _run_vit(image) -> dict:
    from PIL import Image
    if isinstance(image, str):
        bgr = cv2.imread(image)
        if bgr is None:
            raise ValueError(f"Не удалось прочитать изображение: {image}")
    else:
        bgr = image
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    # top_k=None — вернуть вероятности ВСЕХ классов (по умолчанию только топ-5).
    predictions = load_vit_model()(Image.fromarray(rgb), top_k=None)
    return {p["label"]: p["score"] for p in predictions}


def get_emotions(image: str | np.ndarray | None,
                 model: str = "deepface",
                 is_face_crop: bool = False) -> dict:
    """
    Функция распознает эмоции человека на изображении.

    Параметры:
        image        — путь к изображению (str) или кадр/вырезка лица
                       в формате numpy BGR (как отдаёт OpenCV).
        model        — "deepface" или "vit".
        is_face_crop — True, если на вход уже подано вырезанное лицо
                       (например, детектором BlazeFace). Тогда DeepFace
                       не ищет лицо повторно. ViT всегда классифицирует
                       изображение целиком, поэтому ему нужно вырезанное лицо.

    Возвращает словарь вида:
    {'emotions': {'angry': 1.2,
                  'disgust': 0.1,
                  'fear': 3.4,
                  'happy': 90.0,
                  'sad': 2.3,
                  'neutral': 3.0},          # проценты, сумма = 100
     'dominant_emotion': 'happy',
     'inference_time': 0.27134099999966566} - в случае успешного распознавания.

    В противном случае:
    {'emotions': {},
     'dominant_emotion': '',
     'inference_time': 0.0}
    """
    result = _empty_result()

    if model not in MODELS:
        raise ValueError(f"Неизвестная модель {model!r}, допустимо: {MODELS}")

    if isinstance(image, np.ndarray):
        if image.size == 0:
            return result
    elif not isinstance(image, str):
        return result

    start_time = perf_counter()
    try:
        raw = _run_deepface(image, is_face_crop) if model == "deepface" else _run_vit(image)
    except Exception:
        return result
    end_time = perf_counter()

    emotions = _normalize(raw)
    if not emotions:
        return result

    result["emotions"] = emotions
    result["dominant_emotion"] = max(emotions, key=emotions.get)
    result["inference_time"] = end_time - start_time
    return result


# Тестовая проверка
if __name__ == "__main__":
    for m in MODELS:
        print(m, get_emotions(".test_images/test.png", model=m))
        print(m, get_emotions(".test_images/test2.png", model=m))
