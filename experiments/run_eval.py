"""
Офлайн-оценка распознавания эмоций на видео-датасетах (RAVDESS, CREMA-D).

Два этапа:
  1. ПРЕДСКАЗАНИЕ (дорого). Каждый клип читается кадр за кадром (с шагом --stride),
     на каждом кадре BlazeFace ищет самое крупное лицо, модель (DeepFace / ViT)
     выдаёт вероятности 6 эмоций. Результат кэшируется в results/cache/ —
     повторный запуск с теми же клипами, моделью и шагом ничего не пересчитывает.
  2. ОЦЕНКА (дёшево). К покадровым вероятностям применяются режимы сглаживания
     (none / ma:N / ema:A) и считаются метрики. Сглаживание причинное, поэтому
     результат совпадает с тем, что дала бы обработка видеопотока в реальном времени.

Пример (быстрый прогон, ~1–3 мин на CPU):
  python experiments/run_eval.py --dataset ravdess --root "E:/.../ravdess_work/extracted" \
      --actors 01 02 --intensity strong --stride 3 --models deepface vit \
      --modes none ma:5 ma:15 ema:0.3 ema:0.1 --transitions 20 --out results/quick

Описание метрик и выходных файлов — experiments/README.md.
"""

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2  # noqa: E402

import face_detection  # noqa: E402
from emotion_detection import EMOTIONS, MODELS, get_emotions  # noqa: E402
from smoothing import make_smoother  # noqa: E402
from clip_datasets import Clip, list_clips  # noqa: E402
from eval_metrics import classification_report, count_switches, reaction_delay, run_lengths  # noqa: E402

LABEL_IDX = {e: i for i, e in enumerate(EMOTIONS)}
CACHE_DIR = PROJECT_ROOT / "results" / "cache"


# =====================================================================
# Этап 1: покадровые предсказания с кэшем
# =====================================================================
def cache_path(clip: Clip, model: str, stride: int) -> Path:
    return CACHE_DIR / clip.dataset / f"{model}_s{stride}" / f"{clip.stem}.npz"


def predict_clip(clip: Clip, model: str, stride: int, detector, force: bool = False) -> dict:
    """
    Возвращает dict:
      probs  — (T, 6) вероятности в долях; NaN, если лицо не найдено / не распознано
      det_ms, emo_ms — (T,) время детекции и распознавания на кадр, мс
      fps    — частота кадров исходного видео
    """
    path = cache_path(clip, model, stride)
    if path.exists() and not force:
        d = np.load(path)
        return {k: d[k] for k in d.files} | {"fps": float(d["fps"])}

    cap = cv2.VideoCapture(clip.path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    probs, det_ms, emo_ms = [], [], []
    idx = 0
    while True:
        if idx % stride:
            if not cap.grab():
                break
            idx += 1
            continue
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1

        t0 = time.perf_counter()
        faces = face_detection.detect_all_faces(frame, detector)
        det_ms.append((time.perf_counter() - t0) * 1000)

        row, e_ms = np.full(len(EMOTIONS), np.nan), np.nan
        if faces:
            x, y, w, h = faces[0]  # самое крупное лицо
            res = get_emotions(frame[y:y + h, x:x + w], model=model, is_face_crop=True)
            if res["emotions"]:
                row = np.array([res["emotions"][e] for e in EMOTIONS]) / 100.0
                e_ms = res["inference_time"] * 1000
        probs.append(row)
        emo_ms.append(e_ms)
    cap.release()

    out = {"probs": np.array(probs, dtype=np.float32).reshape(-1, len(EMOTIONS)),
           "det_ms": np.array(det_ms, dtype=np.float32),
           "emo_ms": np.array(emo_ms, dtype=np.float32),
           "fps": float(fps)}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)
    return out


# =====================================================================
# Этап 2: сглаживание и метрики
# =====================================================================
def smooth_predict(probs: np.ndarray, mode: str) -> tuple[np.ndarray, float]:
    """
    Прогоняет вероятности через сглаживатель. Кадры без лица (NaN) пропускаются:
    состояние сглаживателя не обновляется, предсказание = -1.
    Возвращает (предсказания (T,), среднее время сглаживания на кадр в мкс).
    """
    smoother = make_smoother(mode)
    preds = np.full(len(probs), -1, dtype=int)
    t0 = time.perf_counter()
    n = 0
    for i, p in enumerate(probs):
        if np.isnan(p).any():
            continue
        preds[i] = int(np.argmax(smoother.update(p)))
        n += 1
    us = (time.perf_counter() - t0) * 1e6 / max(n, 1)
    return preds, us


def evaluate(clips, preds_by_clip, modes, model, stride, transitions_pairs, stable_frames):
    summary_rows, clip_rows, confusions = [], [], {}

    for mode in modes:
        y_true, y_pred = [], []
        switches_total, valid_sec_total, all_runs_sec = 0, 0.0, []
        clip_correct_major, smooth_us = 0, []

        for clip in clips:
            d = preds_by_clip[clip.path]
            preds, us = smooth_predict(d["probs"], mode)
            smooth_us.append(us)
            valid = preds[preds >= 0]
            frame_sec = stride / d["fps"]
            true = LABEL_IDX[clip.label]

            y_true += [true] * len(valid)
            y_pred += valid.tolist()

            sw = count_switches(valid)
            runs = run_lengths(valid)
            dur = len(valid) * frame_sec
            switches_total += sw
            valid_sec_total += dur
            all_runs_sec += [r * frame_sec for r in runs]

            majority = int(np.bincount(valid, minlength=len(EMOTIONS)).argmax()) if len(valid) else -1
            clip_correct_major += int(majority == true)

            clip_rows.append({
                "model": model, "mode": mode, "file": clip.name, "actor": clip.actor,
                "label": clip.label, "intensity": clip.intensity,
                "frames": len(preds), "face_frames": len(valid),
                "frame_acc": float((valid == true).mean()) if len(valid) else np.nan,
                "majority_pred": EMOTIONS[majority] if majority >= 0 else "",
                "switches": sw,
                "switches_per_sec": sw / dur if dur else np.nan,
                "mean_stable_sec": float(np.mean(runs) * frame_sec) if runs else np.nan,
            })

        rep = classification_report(y_true, y_pred, len(EMOTIONS))
        confusions[mode] = rep["confusion"]

        all_det = np.concatenate([preds_by_clip[c.path]["det_ms"] for c in clips])
        all_emo = np.concatenate([preds_by_clip[c.path]["emo_ms"] for c in clips])
        all_emo = all_emo[~np.isnan(all_emo)]
        mean_total_ms = float(np.mean(all_det) + (np.mean(all_emo) if len(all_emo) else 0))
        total_frames = sum(len(preds_by_clip[c.path]["probs"]) for c in clips)

        row = {
            "model": model, "mode": mode, "clips": len(clips),
            "frames": total_frames, "face_frames": len(y_true),
            "face_rate": len(y_true) / total_frames if total_frames else 0,
            "accuracy": rep["accuracy"],
            "precision_macro": rep["precision_macro"],
            "recall_macro": rep["recall_macro"],
            "f1_macro": rep["f1_macro"],
            **{f"f1_{e}": float(rep["per_class_f1"][i]) for i, e in enumerate(EMOTIONS)},
            "clip_acc_majority": clip_correct_major / len(clips) if clips else 0,
            "switches_per_clip": switches_total / len(clips) if clips else 0,
            "switches_per_sec": switches_total / valid_sec_total if valid_sec_total else np.nan,
            "mean_stable_sec": float(np.mean(all_runs_sec)) if all_runs_sec else np.nan,
            "det_ms": float(np.mean(all_det)) if len(all_det) else np.nan,
            "emo_ms": float(np.mean(all_emo)) if len(all_emo) else np.nan,
            "smooth_us": float(np.mean(smooth_us)),
            "fps_pipeline": 1000.0 / mean_total_ms if mean_total_ms else np.nan,
        }

        # --- задержка реакции на склеенных парах клипов A → B ---
        if transitions_pairs:
            delays_sec, misses = [], 0
            for a, b in transitions_pairs:
                pa, pb = preds_by_clip[a.path]["probs"], preds_by_clip[b.path]["probs"]
                preds, _ = smooth_predict(np.vstack([pa, pb]), mode)
                valid_a = int((preds[:len(pa)] >= 0).sum())
                valid = preds[preds >= 0]
                delay = reaction_delay(valid, valid_a, LABEL_IDX[b.label], stable_frames)
                if delay is None:
                    misses += 1
                else:
                    delays_sec.append(delay * stride / preds_by_clip[b.path]["fps"])
            row["transitions"] = len(transitions_pairs)
            row["delay_sec_mean"] = float(np.mean(delays_sec)) if delays_sec else np.nan
            row["delay_sec_median"] = float(np.median(delays_sec)) if delays_sec else np.nan
            row["delay_miss_rate"] = misses / len(transitions_pairs)

        summary_rows.append(row)

    return summary_rows, clip_rows, confusions


def make_transition_pairs(clips, n_pairs, seed):
    """Случайные пары клипов одного актёра с разными эмоциями (A → B)."""
    if n_pairs <= 0:
        return []
    rng = random.Random(seed)
    by_actor: dict[str, list[Clip]] = {}
    for c in clips:
        by_actor.setdefault(c.actor, []).append(c)
    candidates = [(a, b) for group in by_actor.values() for a in group for b in group
                  if a.label != b.label]
    return rng.sample(candidates, min(n_pairs, len(candidates)))


# =====================================================================
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=["ravdess", "cremad"])
    ap.add_argument("--root", default=None,
                    help="папка с видео датасета; если не указана — берётся из experiments/local_paths.json")
    ap.add_argument("--actors", nargs="*", help="номера актёров, например 01 02")
    ap.add_argument("--emotions", nargs="*", choices=EMOTIONS, help="только эти эмоции")
    ap.add_argument("--intensity", default="all", choices=["all", "normal", "strong"])
    ap.add_argument("--per-label", type=int, help="не более N клипов на эмоцию (случайно)")
    ap.add_argument("--limit", type=int, help="не более N клипов всего")
    ap.add_argument("--models", nargs="+", default=["deepface"], choices=MODELS)
    ap.add_argument("--modes", nargs="+", default=["none", "ma:5", "ma:15", "ema:0.3", "ema:0.1"],
                    help="режимы сглаживания: none, ma:N, ema:A")
    ap.add_argument("--stride", type=int, default=1, help="анализировать каждый N-й кадр")
    ap.add_argument("--transitions", type=int, default=0,
                    help="число склеенных пар клипов для замера задержки реакции")
    ap.add_argument("--stable-frames", type=int, default=5,
                    help="сколько кадров подряд новая эмоция должна держаться, чтобы считать реакцию")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="пересчитать кэш предсказаний")
    ap.add_argument("--out", default=None, help="папка результатов (по умолчанию results/<дата-время>)")
    return ap.parse_args()


LOCAL_PATHS = Path(__file__).resolve().parent / "local_paths.json"


def resolve_root(dataset: str, root: str | None) -> str:
    """
    Путь к датасету: --root, иначе experiments/local_paths.json вида
    {"ravdess": "E:/.../extracted", "cremad": "artifacts/CREMA-D-small"}.
    Файл свой на каждом компьютере и в git не хранится.
    Относительные пути считаются от корня проекта.
    """
    if root is None:
        if not LOCAL_PATHS.exists():
            sys.exit(f"Не указан --root и нет {LOCAL_PATHS}. "
                     f"Создайте его по образцу experiments/local_paths.example.json")
        root = json.loads(LOCAL_PATHS.read_text(encoding="utf-8")).get(dataset)
        if not root:
            sys.exit(f"В {LOCAL_PATHS} нет пути для датасета {dataset!r}")
    p = Path(root)
    return str(p if p.is_absolute() else PROJECT_ROOT / p)


def main():
    args = parse_args()
    args.root = resolve_root(args.dataset, args.root)
    for m in args.modes:
        make_smoother(m)  # ранняя проверка синтаксиса режимов

    clips = list_clips(args.dataset, args.root, actors=args.actors, emotions=args.emotions,
                       intensity=args.intensity, per_label=args.per_label,
                       limit=args.limit, seed=args.seed)
    if not clips:
        sys.exit("Не найдено ни одного клипа — проверьте --root и фильтры.")

    out_dir = Path(args.out) if args.out else PROJECT_ROOT / "results" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = pd.Series([c.label for c in clips]).value_counts().to_dict()
    print(f"Клипов: {len(clips)} {counts}")
    print(f"Результаты: {out_dir}")

    detector = face_detection.load_face_detector()
    pairs = make_transition_pairs(clips, args.transitions, args.seed)

    all_summary, all_clips = [], []
    for model in args.models:
        # прогрев модели, чтобы загрузка весов не попала в замер времени
        get_emotions(np.zeros((64, 64, 3), dtype=np.uint8), model=model, is_face_crop=True)

        preds_by_clip = {}
        t_start = time.time()
        for i, clip in enumerate(clips, 1):
            preds_by_clip[clip.path] = predict_clip(clip, model, args.stride, detector, args.force)
            if i % 10 == 0 or i == len(clips):
                print(f"  [{model}] {i}/{len(clips)} клипов, {time.time() - t_start:.0f} с", flush=True)

        summary, clip_rows, confusions = evaluate(
            clips, preds_by_clip, args.modes, model, args.stride, pairs, args.stable_frames)
        all_summary += summary
        all_clips += clip_rows
        for mode, cm in confusions.items():
            safe = mode.replace(":", "_")
            pd.DataFrame(cm, index=[f"true_{e}" for e in EMOTIONS],
                         columns=[f"pred_{e}" for e in EMOTIONS]).to_csv(
                out_dir / f"confusion_{model}_{safe}.csv", encoding="utf-8")

    summary_df = pd.DataFrame(all_summary)
    summary_df.to_csv(out_dir / "summary.csv", index=False, encoding="utf-8")
    pd.DataFrame(all_clips).to_csv(out_dir / "per_clip.csv", index=False, encoding="utf-8")
    (out_dir / "config.json").write_text(json.dumps(
        vars(args) | {"n_clips": len(clips), "label_counts": counts,
                      "transition_pairs": [[a.name, b.name] for a, b in pairs]},
        ensure_ascii=False, indent=2), encoding="utf-8")

    cols = ["model", "mode", "accuracy", "f1_macro", "clip_acc_majority",
            "switches_per_sec", "mean_stable_sec", "fps_pipeline"]
    if pairs:
        cols += ["delay_sec_mean", "delay_miss_rate"]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print()
        print(summary_df[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
