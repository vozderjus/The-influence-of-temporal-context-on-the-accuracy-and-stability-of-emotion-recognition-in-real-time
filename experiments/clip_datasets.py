"""
Список видеоклипов датасетов с метками эмоций.

Метки берутся из имён файлов, поэтому каталоги CSV не обязательны.

RAVDESS: MM-VV-EE-II-SS-RR-AA.mp4
    MM модальность (01 видео+звук, 02 только видео, 03 только звук — пропускаем)
    VV канал (01 речь, 02 пение)
    EE эмоция (01 neutral, 02 calm, 03 happy, 04 sad, 05 angry, 06 fearful,
               07 disgust, 08 surprised); calm и surprised пропускаем
    II интенсивность (01 normal, 02 strong)
    SS фраза, RR повтор, AA актёр (нечётный — мужчина, чётный — женщина)

CREMA-D: AAAA_SSS_EEE_LL.flv
    AAAA актёр, SSS фраза, EEE эмоция (ANG DIS FEA HAP NEU SAD),
    LL интенсивность (LO MD HI XX). HI считаем strong, остальное normal.
    Видео лежат в Git LFS: файлы-заглушки (< 2 КБ) пропускаются.
"""

import os
import random
from dataclasses import dataclass, field
from pathlib import Path

LFS_STUB_MAX_BYTES = 2048

RAVDESS_EMOTIONS = {"01": "neutral", "03": "happy", "04": "sad",
                    "05": "angry", "06": "fear", "07": "disgust"}
CREMAD_EMOTIONS = {"ANG": "angry", "DIS": "disgust", "FEA": "fear",
                   "HAP": "happy", "NEU": "neutral", "SAD": "sad"}


@dataclass
class Clip:
    path: str
    dataset: str
    label: str          # одна из emotion_detection.EMOTIONS
    actor: str
    intensity: str      # "normal" | "strong"
    meta: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return Path(self.path).name

    @property
    def stem(self) -> str:
        return Path(self.path).stem


def _ravdess_clips(root: Path) -> list[Clip]:
    """
    Модальности 01 (видео+звук) и 02 (только видео) содержат ОДИНАКОВУЮ картинку,
    поэтому каждая запись берётся один раз: 01, если есть, иначе 02.
    """
    clips: dict[tuple, Clip] = {}
    for p in sorted(root.rglob("*.mp4")):
        parts = p.stem.split("-")
        if len(parts) != 7:
            continue
        modality, channel, emo, inten, statement, rep, actor = parts
        if modality not in ("01", "02") or emo not in RAVDESS_EMOTIONS:
            continue
        key = (channel, emo, inten, statement, rep, actor)
        if key in clips and clips[key].meta["modality"] == "01":
            continue
        clips[key] = Clip(
            path=str(p), dataset="ravdess", label=RAVDESS_EMOTIONS[emo], actor=actor,
            intensity="strong" if inten == "02" else "normal",
            meta={"modality": modality, "channel": channel, "statement": statement, "repetition": rep},
        )
    return list(clips.values())


def _cremad_clips(root: Path) -> list[Clip]:
    clips = []
    video_dir = root / "VideoFlash" if (root / "VideoFlash").is_dir() else root
    for p in video_dir.rglob("*.flv"):
        parts = p.stem.split("_")
        if len(parts) != 4 or parts[2] not in CREMAD_EMOTIONS:
            continue
        if os.path.getsize(p) < LFS_STUB_MAX_BYTES:
            continue  # заглушка Git LFS, видео не скачано
        actor, sentence, emo, level = parts
        clips.append(Clip(
            path=str(p), dataset="cremad", label=CREMAD_EMOTIONS[emo], actor=actor,
            intensity="strong" if level == "HI" else "normal",
            meta={"sentence": sentence, "level": level},
        ))
    return clips


def list_clips(dataset: str, root: str,
               actors: list[str] | None = None,
               emotions: list[str] | None = None,
               intensity: str = "all",
               per_label: int | None = None,
               limit: int | None = None,
               seed: int = 0) -> list[Clip]:
    """
    Возвращает отфильтрованный список клипов (отсортирован по имени файла).

    per_label — взять не более N клипов на каждую эмоцию (случайно, seed),
                удобно для быстрых сбалансированных прогонов.
    limit     — общее ограничение после всех фильтров.
    """
    root_p = Path(root)
    if dataset == "ravdess":
        clips = _ravdess_clips(root_p)
    elif dataset == "cremad":
        clips = _cremad_clips(root_p)
    else:
        raise ValueError(f"Неизвестный датасет {dataset!r}: ожидается ravdess или cremad")

    if actors:
        wanted = {a.zfill(2) if dataset == "ravdess" else a for a in actors}
        clips = [c for c in clips if c.actor in wanted]
    if emotions:
        clips = [c for c in clips if c.label in set(emotions)]
    if intensity != "all":
        # neutral в RAVDESS бывает только normal — не теряем его при intensity=strong
        clips = [c for c in clips if c.intensity == intensity or c.label == "neutral"]

    clips.sort(key=lambda c: c.name)

    if per_label:
        rng = random.Random(seed)
        by_label: dict[str, list[Clip]] = {}
        for c in clips:
            by_label.setdefault(c.label, []).append(c)
        clips = []
        for label in sorted(by_label):
            group = by_label[label]
            clips += rng.sample(group, min(per_label, len(group)))
        clips.sort(key=lambda c: c.name)

    if limit:
        clips = clips[:limit]
    return clips
