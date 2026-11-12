"""
Визуализация: русские названия эмоций, цвета, шрифты, рамки с подписями.
"""

import os
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

EMOTION_RU = {
    "happy":   "радость",
    "sad":     "грусть",
    "angry":   "злость",
    "fear":    "страх",
    "disgust": "отвращение",
    "neutral": "нейтральное",
}

EMOTION_COLORS = {
    "happy":   (0, 200, 0),
    "sad":     (50, 100, 220),
    "angry":   (220, 40, 40),
    "fear":    (150, 50, 180),
    "disgust": (150, 130, 0),
    "neutral": (140, 140, 140),
}


def ru(emotion_en: str) -> str:
    return EMOTION_RU.get(emotion_en.lower(), emotion_en)


def get_color(emotion_key: str) -> tuple[int, int, int]:
    return EMOTION_COLORS.get(emotion_key.lower(), (255, 255, 255))


def color_hex(emotion_key: str) -> str:
    r, g, b = get_color(emotion_key)
    return f"#{r:02x}{g:02x}{b:02x}"


# ========== ПОИСК ШРИФТА (кроссплатформенный) ==========
_FONT_CANDIDATES = [
    # Windows
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/Arial.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
    # macOS
    "/Library/Fonts/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    # Linux — DejaVu (поддерживает кириллицу)
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    # Linux — Liberation (метрически совместим с Arial)
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/liberation-sans/LiberationSans-Regular.ttf",
    "/usr/share/fonts/liberation/LiberationSans-Regular.ttf",
    # Linux — FreeFonts
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    "/usr/share/fonts/gnu-free/FreeSans.ttf",
    # Linux — Ubuntu
    "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
    # Linux — Noto Sans
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/google-noto/NotoSans-Regular.ttf",
]


def find_system_font() -> str | None:
    """Ищет шрифт с поддержкой кириллицы на Windows, Linux и macOS."""
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            return path
    return None


SYSTEM_FONT_PATH = find_system_font()


@lru_cache(maxsize=32)
def load_font(size: int = 28):
    if SYSTEM_FONT_PATH:
        try:
            return ImageFont.truetype(SYSTEM_FONT_PATH, size)
        except Exception:
            pass
    return ImageFont.load_default()


def fit_font(draw, text, max_width, max_size=28, min_size=12):
    for size in range(max_size, min_size - 1, -1):
        font = load_font(size)
        bbox = draw.textbbox((0, 0), text, font=font)
        if bbox[2] - bbox[0] <= max_width:
            return font
    return load_font(min_size)


# ========== РИСОВАНИЕ РАМКИ И ПОДПИСИ ==========
def draw_overlay(frame_bgr, faces_with_emotions):
    """
    Рисует рамки и подписи «эмоция NN%» над лицами.
    faces_with_emotions — список ((x, y, w, h), emotion_en, score_percent).
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(pil_img)

    for face_box, emotion_en, score in faces_with_emotions:
        if emotion_en is None:
            continue

        x, y, w, h = face_box
        color = get_color(emotion_en)
        label = f"{ru(emotion_en)} {score:.0f}%"

        padding = 6
        max_text_width = max(30, w - padding * 2)
        font = fit_font(draw, label, max_text_width, max_size=26, min_size=12)

        draw.rectangle([x, y, x + w, y + h], outline=color, width=3)

        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]

        label_y_top = max(0, y - text_h - padding * 2 - 4)
        draw.rectangle(
            [x, label_y_top, x + text_w + padding * 2, label_y_top + text_h + padding * 2],
            fill=color,
        )
        draw.text((x + padding, label_y_top + padding - 2), label, fill=(255, 255, 255), font=font)

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
