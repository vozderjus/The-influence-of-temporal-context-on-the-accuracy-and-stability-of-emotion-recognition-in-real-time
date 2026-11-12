"""
Метрики качества, стабильности и задержки реакции.

Метки — целые индексы классов в порядке emotion_detection.EMOTIONS.
"""

import numpy as np


def classification_report(y_true, y_pred, n_classes: int) -> dict:
    """
    Accuracy и macro-усреднённые Precision / Recall / F1 + матрица ошибок.
    Классы, которых нет ни в y_true, ни в y_pred, в macro-среднее не входят.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    cm = np.zeros((n_classes, n_classes), dtype=int)  # строки — истина, столбцы — прогноз
    np.add.at(cm, (y_true, y_pred), 1)

    tp = np.diag(cm).astype(float)
    pred_sum = cm.sum(axis=0).astype(float)
    true_sum = cm.sum(axis=1).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(pred_sum > 0, tp / pred_sum, 0.0)
        recall = np.where(true_sum > 0, tp / true_sum, 0.0)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)

    present = (true_sum > 0) | (pred_sum > 0)
    total = cm.sum()
    return {
        "accuracy": float(tp.sum() / total) if total else 0.0,
        "precision_macro": float(precision[present].mean()) if present.any() else 0.0,
        "recall_macro": float(recall[present].mean()) if present.any() else 0.0,
        "f1_macro": float(f1[present].mean()) if present.any() else 0.0,
        "per_class_f1": f1,
        "confusion": cm,
    }


def run_lengths(preds) -> list[int]:
    """Длины отрезков подряд идущих одинаковых предсказаний: [A,A,B,B,B] -> [2,3]."""
    preds = list(preds)
    if not preds:
        return []
    runs, cur = [], 1
    for prev, nxt in zip(preds, preds[1:]):
        if nxt == prev:
            cur += 1
        else:
            runs.append(cur)
            cur = 1
    runs.append(cur)
    return runs


def count_switches(preds) -> int:
    """Число смен предсказанного класса между соседними кадрами."""
    return max(len(run_lengths(preds)) - 1, 0)


def reaction_delay(preds, boundary: int, new_label: int, stable_frames: int) -> int | None:
    """
    Задержка реакции на смену эмоции (в кадрах).

    preds    — предсказания склеенной последовательности «клип A → клип B»;
    boundary — индекс первого кадра клипа B;
    Возвращает t - boundary для первого t >= boundary, начиная с которого
    `stable_frames` кадров подряд предсказан new_label. None — не дождались.
    """
    preds = list(preds)
    for t in range(boundary, len(preds) - stable_frames + 1):
        if all(p == new_label for p in preds[t:t + stable_frames]):
            return t - boundary
    return None
