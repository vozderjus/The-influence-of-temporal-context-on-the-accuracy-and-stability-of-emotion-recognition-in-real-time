"""
PDF-отчёт по обработанному видео: шапка, график эмоций по времени,
ключевые кадры.

results_log    — список {"time": сек, "angry": %, ..., "neutral": %}
screenshots_log — список {"time", "emotion", "score", "frame" (BGR)}
"""

import os
import tempfile
from collections import Counter
from datetime import datetime

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fpdf import FPDF
from PIL import Image

from emotion_detection import EMOTIONS
from visualization import SYSTEM_FONT_PATH, get_color, ru


def _dominant(entry: dict) -> str:
    scores = {k: entry.get(k, 0) for k in EMOTIONS}
    return max(scores, key=scores.get)


def generate_emotion_timeline_chart(results_log, output_path) -> bool:
    """
    График: по горизонтали — время, по вертикали — эмоции.
    Цветные полосы показывают доминантную эмоцию в каждый момент.
    """
    if not results_log:
        return False

    times = [r["time"] for r in results_log]
    dominant_emotions = [_dominant(r) for r in results_log]

    emotion_to_idx = {e: i for i, e in enumerate(EMOTIONS)}
    y_values = [emotion_to_idx[e] for e in dominant_emotions]
    colors_for_plot = [tuple(c / 255 for c in get_color(e)) for e in dominant_emotions]

    fig, ax = plt.subplots(figsize=(10, 4))

    for i in range(len(times) - 1):
        ax.barh(y_values[i], times[i + 1] - times[i], left=times[i],
                height=0.7, color=colors_for_plot[i], edgecolor="none")
    # Последняя точка — маленькая полоска
    bar_width = (times[-1] - times[0]) / len(times) if len(times) > 1 else 1
    ax.barh(y_values[-1], bar_width, left=times[-1],
            height=0.7, color=colors_for_plot[-1], edgecolor="none")

    ax.set_yticks(range(len(EMOTIONS)))
    ax.set_yticklabels([ru(e) for e in EMOTIONS], fontsize=12)
    ax.set_xlabel("Время (сек)", fontsize=12)
    ax.set_title("Эмоции по времени", fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlim(left=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def select_best_screenshots(screenshots_log, max_count=3):
    """
    Выбирает лучшие скриншоты — из середины самых длинных отрезков
    одной эмоции с наивысшей уверенностью.
    """
    if not screenshots_log:
        return []

    # Группируем последовательные кадры с одной эмоцией в сегменты
    segments = []
    current = {"emotion": screenshots_log[0]["emotion"], "entries": [screenshots_log[0]]}
    for entry in screenshots_log[1:]:
        if entry["emotion"] == current["emotion"]:
            current["entries"].append(entry)
        else:
            segments.append(current)
            current = {"emotion": entry["emotion"], "entries": [entry]}
    segments.append(current)

    # Качество сегмента = длина * средняя уверенность
    for seg in segments:
        scores = [e["score"] for e in seg["entries"]]
        seg["quality"] = len(scores) * (sum(scores) / len(scores))
    segments.sort(key=lambda s: s["quality"], reverse=True)

    # Сначала — по одному кадру на разные эмоции
    selected, used_emotions = [], set()
    for seg in segments:
        if len(selected) >= max_count:
            break
        if seg["emotion"] in used_emotions:
            continue
        selected.append(seg["entries"][len(seg["entries"]) // 2])
        used_emotions.add(seg["emotion"])

    # Если разных эмоций не хватило — добираем любыми
    for seg in segments:
        if len(selected) >= max_count:
            break
        candidate = seg["entries"][len(seg["entries"]) // 2]
        if candidate not in selected:
            selected.append(candidate)

    selected.sort(key=lambda e: e["time"])
    return selected


def generate_pdf_report(video_name, model_name, results_log, screenshots_log, duration) -> bytes:
    """Генерирует PDF-отчёт и возвращает его байты."""
    most_common_en = Counter(_dominant(r) for r in results_log).most_common(1)[0][0]

    chart_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
    chart_ok = generate_emotion_timeline_chart(results_log, chart_path)

    screenshot_paths = []
    for entry in select_best_screenshots(screenshots_log, max_count=3):
        img_rgb = cv2.cvtColor(entry["frame"], cv2.COLOR_BGR2RGB)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        tmp.close()
        Image.fromarray(img_rgb).save(tmp.name, quality=85)
        screenshot_paths.append({
            "path": tmp.name,
            "time": entry["time"],
            "emotion": ru(entry["emotion"]),
            "score": entry["score"],
        })

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Шрифт с поддержкой кириллицы
    if SYSTEM_FONT_PATH:
        pdf.add_font("CustomFont", "", SYSTEM_FONT_PATH)
        pdf.add_font("CustomFont", "B", SYSTEM_FONT_PATH)
        font_name = "CustomFont"
    else:
        font_name = "Helvetica"

    # --- Страница 1: шапка + график ---
    pdf.add_page()
    pdf.set_font(font_name, "B", 20)
    pdf.cell(0, 12, "Отчёт по анализу эмоций", ln=True, align="C")
    pdf.ln(8)

    pdf.set_font(font_name, "", 12)
    now = datetime.now().strftime("%d.%m.%Y %H:%M")
    pdf.cell(0, 8, f"Дата и время отчёта: {now}", ln=True)
    pdf.cell(0, 8, f"Видеофайл: {video_name}", ln=True)
    pdf.cell(0, 8, f"Длительность: {duration:.1f} сек", ln=True)
    pdf.cell(0, 8, f"Модель: {model_name}", ln=True)
    pdf.cell(0, 8, f"Преобладающая эмоция: {ru(most_common_en)}", ln=True)
    pdf.ln(6)

    page_width = pdf.w - pdf.l_margin - pdf.r_margin

    if chart_ok and os.path.exists(chart_path):
        pdf.set_font(font_name, "B", 14)
        pdf.cell(0, 10, "График эмоций по времени", ln=True)
        pdf.ln(2)
        pdf.image(chart_path, x=pdf.l_margin, w=page_width)
        pdf.ln(4)

    # --- Ключевые кадры ---
    if screenshot_paths:
        if pdf.get_y() > 200:
            pdf.add_page()
        pdf.set_font(font_name, "B", 14)
        pdf.cell(0, 10, "Ключевые кадры", ln=True)
        pdf.ln(2)

        for sc in screenshot_paths:
            if pdf.get_y() > 190:
                pdf.add_page()
            pdf.set_font(font_name, "", 11)
            pdf.cell(0, 7, f"Время: {sc['time']:.1f} сек  |  Эмоция: {sc['emotion']}  |  "
                           f"Уверенность: {sc['score']:.0f}%", ln=True)
            pdf.ln(1)
            if os.path.exists(sc["path"]):
                img_w = min(page_width, 160)
                pdf.image(sc["path"], x=pdf.l_margin + (page_width - img_w) / 2, w=img_w)
                pdf.ln(6)

    pdf_bytes = bytes(pdf.output())

    for path in [chart_path] + [sc["path"] for sc in screenshot_paths]:
        try:
            os.unlink(path)
        except OSError:
            pass

    return pdf_bytes
