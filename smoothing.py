"""
Причинное (causal) временное сглаживание вероятностей эмоций.

Каждый сглаживатель работает онлайн: получает вектор вероятностей текущего кадра
и возвращает сглаженный вектор, используя ТОЛЬКО текущий и прошлые кадры.
Поэтому один и тот же код подходит и для офлайн-экспериментов, и для
видеопотока в реальном времени (один сглаживатель на одно отслеживаемое лицо).

Вектор вероятностей — numpy-массив длины 6 в порядке emotion_detection.EMOTIONS,
значения в долях (сумма = 1).

Режимы задаются строкой (см. make_smoother):
    "none"      — без сглаживания (покадровая классификация, базовая линия)
    "ma:N"      — скользящее среднее по последним N кадрам
    "ema:A"     — экспоненциальное сглаживание, A ∈ (0, 1] — вес нового кадра
                  (меньше A → сильнее сглаживание и больше задержка)
"""

from collections import deque

import numpy as np


class Smoother:
    """Базовый класс. update() принимает и возвращает вектор вероятностей."""

    name = "none"

    def reset(self) -> None:
        pass

    def update(self, probs: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class NoSmoothing(Smoother):
    name = "none"

    def update(self, probs):
        return np.asarray(probs, dtype=np.float64)


class MovingAverage(Smoother):
    """Среднее по последним `window` кадрам (в начале — по тем, что есть)."""

    def __init__(self, window: int):
        if window < 1:
            raise ValueError("Окно скользящего среднего должно быть >= 1")
        self.window = window
        self.name = f"ma:{window}"
        self._buf = deque(maxlen=window)

    def reset(self):
        self._buf.clear()

    def update(self, probs):
        self._buf.append(np.asarray(probs, dtype=np.float64))
        return np.mean(self._buf, axis=0)


class EMA(Smoother):
    """s_t = alpha * p_t + (1 - alpha) * s_{t-1};  s_0 = p_0."""

    def __init__(self, alpha: float):
        if not 0 < alpha <= 1:
            raise ValueError("alpha должен быть в (0, 1]")
        self.alpha = alpha
        self.name = f"ema:{alpha:g}"
        self._state = None

    def reset(self):
        self._state = None

    def update(self, probs):
        p = np.asarray(probs, dtype=np.float64)
        if self._state is None:
            self._state = p
        else:
            self._state = self.alpha * p + (1 - self.alpha) * self._state
        return self._state


def make_smoother(spec: str) -> Smoother:
    """Создаёт сглаживатель по строке: "none", "ma:5", "ema:0.3"."""
    spec = spec.strip().lower()
    if spec in ("none", "frame", ""):
        return NoSmoothing()
    kind, _, value = spec.partition(":")
    if kind == "ma":
        return MovingAverage(int(value))
    if kind == "ema":
        return EMA(float(value))
    raise ValueError(f"Неизвестный режим сглаживания: {spec!r} (ожидается none, ma:N, ema:A)")
