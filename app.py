"""
Streamlit-интерфейс распознавания эмоций: фото, видео, веб-камера.

Запуск:  streamlit run app.py

Вся логика вынесена в модули:
  face_detection.py    — поиск лиц (BlazeFace)
  emotion_detection.py — распознавание эмоций (DeepFace / ViT), get_emotions()
  visualization.py     — рамки, подписи, цвета, русские названия
  report.py            — PDF-отчёт
"""

import base64
import subprocess
import tempfile
import threading
import time

import av
import cv2
import imageio_ffmpeg
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_webrtc import WebRtcMode, webrtc_streamer

import face_detection
from emotion_detection import get_emotions, load_vit_model
from report import generate_pdf_report
from visualization import color_hex, draw_overlay, ru

# ========== НАСТРОЙКА СТРАНИЦЫ ==========
st.set_page_config(page_title="Распознавание эмоций", page_icon="🎭", layout="centered")

st.markdown("""
<style>
    .block-container { padding-top: 2.5rem; padding-bottom: 3rem; max-width: 800px; }

    .app-header {
        display: flex; align-items: center; gap: 16px;
        padding: 20px 24px;
        background: linear-gradient(135deg, #6C5CE7 0%, #4834D4 100%);
        border-radius: 16px; margin-bottom: 24px;
    }
    .app-header .icon { font-size: 40px; line-height: 1; }
    .app-header .title { font-size: 26px; font-weight: 700; color: white; margin: 0; }
    .app-header .subtitle { font-size: 14px; color: rgba(255,255,255,0.85); margin: 2px 0 0 0; }

    .stButton button { border-radius: 10px; font-weight: 600; transition: all 0.2s ease; }
    .stButton button:hover { transform: translateY(-1px); }

    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { border-radius: 10px 10px 0 0; padding: 10px 18px; font-weight: 500; }

    [data-testid="stMetric"] {
        background: #F8F9FE; border: 1px solid #ECECF5;
        border-radius: 12px; padding: 14px 18px;
    }

    .stRadio > div { gap: 12px; }

    .face-card {
        display: flex; align-items: center; gap: 12px;
        padding: 12px 16px; border: 1px solid #ECECF5;
        border-radius: 12px; margin-bottom: 8px; background: white;
    }
    .face-dot { width: 12px; height: 12px; border-radius: 50%; flex-shrink: 0; }
    .face-name { font-size: 15px; flex: 1; }
    .face-score { font-size: 14px; font-weight: 600; color: #666; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="app-header">
    <div class="icon">🎭</div>
    <div>
        <p class="title">Распознавание эмоций</p>
        <p class="subtitle">Анализ медиапотока на основе ИИ</p>
    </div>
</div>
""", unsafe_allow_html=True)


# ========== КЭШИРУЕМЫЕ РЕСУРСЫ ==========
@st.cache_resource
def load_face_detector():
    with st.spinner("Загружаю детектор лиц (при первом запуске скачивается ~225 КБ)..."):
        return face_detection.load_face_detector()


@st.cache_resource
def load_vit():
    return load_vit_model()


# ========== АНАЛИЗ ОДНОГО ЛИЦА ==========
def analyze_face(face_bgr, model_key):
    """
    Возвращает (dominant, score, emotions) или None, если распознать не удалось.
    face_bgr — вырезанное лицо в BGR.
    """
    res = get_emotions(face_bgr, model=model_key, is_face_crop=True)
    if not res["emotions"]:
        return None
    dominant = res["dominant_emotion"]
    return dominant, res["emotions"][dominant], res["emotions"]


# ========== ПЕРЕКОДИРОВКА H.264 ==========
def convert_to_h264(input_path, output_path):
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ffmpeg_exe, "-y", "-i", input_path,
           "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p", output_path]
    subprocess.run(cmd, check=True, capture_output=True)


# ========== ВЫБОР МОДЕЛИ ==========
st.subheader("Выберите модель")
model_choice = st.radio(
    "Какой моделью анализировать?",
    options=["DeepFace (CNN, базовая)", "ViT (Transformer, улучшенная)"],
    horizontal=True,
)
model_name = "DeepFace" if "DeepFace" in model_choice else "ViT"
model_key = model_name.lower()  # "deepface" | "vit" — для get_emotions()

st.caption("ℹ️ Модели обучены на фронтальных изображениях лиц. На лицах в профиль или сильно "
           "под углом точность заметно падает — это известное ограничение датасета FER2013.")

tab_photo, tab_video, tab_webcam = st.tabs(["📷 Фото", "🎬 Видео", "📹 Веб-камера"])

# ========== ВКЛАДКА ФОТО ==========
with tab_photo:
    uploaded_photo = st.file_uploader("Выберите фото", type=["jpg", "jpeg", "png"], key="photo_uploader")

    if uploaded_photo is not None:
        image = Image.open(uploaded_photo).convert("RGB")
        frame_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        with st.spinner(f"Ищу лица и анализирую через {model_name}..."):
            try:
                faces = face_detection.detect_all_faces(frame_bgr, load_face_detector())

                if not faces:
                    st.warning("Лиц не обнаружено.")
                    st.image(image, caption="Загруженное фото", width="stretch")
                else:
                    faces_with_emotions = []
                    for (x, y, w, h) in faces:
                        analysis = analyze_face(frame_bgr[y:y + h, x:x + w], model_key)
                        if analysis is not None:
                            dominant, score, _ = analysis
                            faces_with_emotions.append(((x, y, w, h), dominant, score))

                    annotated = draw_overlay(frame_bgr, faces_with_emotions)
                    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                             caption=f"Найдено лиц: {len(faces)}", width="stretch")

                    if faces_with_emotions:
                        dominant_list = [ru(em) for _, em, _ in faces_with_emotions]
                        most_common_emotion = max(set(dominant_list), key=dominant_list.count)
                        avg_conf = sum(sc for _, _, sc in faces_with_emotions) / len(faces_with_emotions)

                        m1, m2, m3 = st.columns(3)
                        m1.metric("Найдено лиц", len(faces_with_emotions))
                        m2.metric("Преобладает", most_common_emotion)
                        m3.metric("Ср. уверенность", f"{avg_conf:.0f}%")

                        st.subheader("Результаты по каждому лицу")
                        for i, (_, emotion_en, score) in enumerate(faces_with_emotions):
                            st.markdown(f"""
                            <div class="face-card">
                                <div class="face-dot" style="background:{color_hex(emotion_en)};"></div>
                                <span class="face-name">#{i + 1} — {ru(emotion_en)}</span>
                                <span class="face-score">{score:.1f}%</span>
                            </div>
                            """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Ошибка: {e}")

# ========== ВКЛАДКА ВИДЕО ==========
with tab_video:
    uploaded_videos = st.file_uploader(
        "Выберите одно или несколько видео",
        type=["mp4", "avi", "mov", "mkv", "flv"],
        key="video_uploader",
        accept_multiple_files=True,
    )

    if uploaded_videos:
        st.subheader("Настройки обработки")
        analysis_interval = st.slider(
            "Обновлять эмоцию раз в N кадров",
            min_value=1, max_value=30, value=5,
            help="Меньше = точнее, но медленнее. 5 — хороший баланс.",
        )

        video_infos = []
        for vid in uploaded_videos:
            tfile_in = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile_in.write(vid.read())
            tfile_in.close()

            cap = cv2.VideoCapture(tfile_in.name)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = total_frames / fps if fps > 0 else 0
            cap.release()

            video_infos.append({
                "name": vid.name, "path": tfile_in.name, "fps": fps,
                "total_frames": total_frames, "width": width, "height": height,
                "duration": duration,
            })
            st.info(f"📹 **{vid.name}** — {duration:.1f} сек | {total_frames} кадров | "
                    f"{fps:.1f} FPS | {width}x{height}")

        total_analyses = sum(vi["total_frames"] // analysis_interval for vi in video_infos)
        total_all_frames = sum(vi["total_frames"] for vi in video_infos)
        sec_per_analysis = 0.4 if model_key == "deepface" else 1.0
        estimated_total = total_analyses * sec_per_analysis * 1.5 + total_all_frames * 0.03

        col1, col2, col3 = st.columns(3)
        col1.metric("Видеофайлов", len(uploaded_videos))
        col2.metric("Всего анализов", total_analyses)
        col3.metric("Примерное время", f"{estimated_total:.0f} сек")
        st.caption("⚠️ Время зависит от количества лиц в кадре. Видео обрабатываются последовательно.")

        if st.button("🎬 Обработать все видео", type="primary"):
            detector = load_face_detector()

            for vi_idx, vi in enumerate(video_infos):
                st.divider()
                st.subheader(f"📹 {vi_idx + 1}/{len(video_infos)}: {vi['name']}")

                fps = vi["fps"] or 25.0
                total_frames = max(vi["total_frames"], 1)

                video_path_raw = tempfile.NamedTemporaryFile(delete=False, suffix="_raw.mp4").name
                video_path_final = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

                out = cv2.VideoWriter(video_path_raw, cv2.VideoWriter_fourcc(*"mp4v"),
                                      fps, (vi["width"], vi["height"]))
                cap = cv2.VideoCapture(vi["path"])
                progress_bar = st.progress(0.0, text=f"Запуск {vi['name']}...")
                start_time = time.time()

                frame_idx = 0
                last_results = []
                results_log = []       # вероятности главного лица по времени
                screenshots_log = []   # кадры для PDF-отчёта

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    faces = face_detection.detect_all_faces(frame, detector)

                    if faces:
                        if frame_idx % analysis_interval == 0:
                            new_results = []
                            for i, (x, y, w, h) in enumerate(faces):
                                analysis = analyze_face(frame[y:y + h, x:x + w], model_key)
                                if analysis is None:
                                    continue
                                dominant, score, emotions = analysis
                                new_results.append(((x, y, w, h), dominant, score))

                                if i == 0:  # главное (самое крупное) лицо
                                    timestamp = frame_idx / fps
                                    results_log.append({"time": timestamp, **emotions})
                                    screenshots_log.append({
                                        "time": timestamp,
                                        "emotion": dominant,
                                        "score": score,
                                        "frame": draw_overlay(frame.copy(), new_results),
                                    })
                            last_results = new_results
                        else:
                            # Между анализами: новые координаты, старые эмоции.
                            # Лица сопоставляются по порядку (по площади) — это НЕ трекинг.
                            updated = [(box, last_results[i][1], last_results[i][2])
                                       for i, box in enumerate(faces) if i < len(last_results)]
                            if updated:
                                last_results = updated

                        if last_results:
                            frame = draw_overlay(frame, last_results)
                    else:
                        last_results = []

                    out.write(frame)
                    frame_idx += 1

                    if frame_idx % 10 == 0 or frame_idx == total_frames:
                        progress_bar.progress(
                            min(frame_idx / total_frames, 1.0),
                            text=f"{vi['name']}: {frame_idx}/{total_frames} | "
                                 f"{time.time() - start_time:.1f} сек",
                        )

                cap.release()
                out.release()

                progress_bar.progress(1.0, text="Перекодирую в H.264...")
                try:
                    convert_to_h264(video_path_raw, video_path_final)
                    final_path = video_path_final
                except Exception as e:
                    st.warning(f"Не удалось перекодировать: {e}")
                    final_path = video_path_raw

                progress_bar.progress(1.0, text=f"✅ {vi['name']} — готово за "
                                                f"{time.time() - start_time:.1f} сек")

                st.subheader(f"🎥 Результат: {vi['name']}")
                with open(final_path, "rb") as f:
                    video_bytes = f.read()
                st.video(video_bytes)
                st.download_button(
                    label=f"⬇️ Скачать {vi['name']}",
                    data=video_bytes,
                    file_name=f"emotion_{vi['name']}",
                    mime="video/mp4",
                    key=f"download_{vi_idx}",
                )

                if results_log:
                    df = pd.DataFrame(results_log).set_index("time")
                    df = df.rename(columns=ru)

                    st.subheader(f"📈 График эмоций: {vi['name']}")
                    st.line_chart(df)

                    most_common = df.idxmax(axis=1).mode()[0]
                    st.success(f"Преобладающая эмоция главного лица: **{most_common}**")

                    with st.spinner("Генерирую PDF-отчёт..."):
                        pdf_bytes = generate_pdf_report(
                            video_name=vi["name"],
                            model_name=model_name,
                            results_log=results_log,
                            screenshots_log=screenshots_log,
                            duration=vi["duration"],
                        )
                    b64 = base64.b64encode(pdf_bytes).decode()
                    st.markdown(
                        f'<a href="data:application/pdf;base64,{b64}" download="report_{vi["name"]}.pdf" '
                        f'target="_blank" style="display:inline-block;padding:0.5em 1em;'
                        f'background-color:#FF4B4B;color:white;text-decoration:none;'
                        f'border-radius:8px;font-weight:bold;">📄 Скачать PDF-отчёт: {vi["name"]}</a>',
                        unsafe_allow_html=True,
                    )

            st.success(f"✅ Все {len(video_infos)} видео обработаны!")

# ========== ВКЛАДКА ВЕБ-КАМЕРА ==========
with tab_webcam:
    st.write("Анализ эмоций в реальном времени через веб-камеру.")

    with st.spinner("Загружаю модели для веб-камеры..."):
        _ = load_face_detector()
        if model_key == "vit":
            _ = load_vit()

    webcam_interval = st.slider(
        "Анализировать каждый N-й кадр",
        min_value=1, max_value=30, value=10,
        help="Больше = быстрее видео, но реже обновляется эмоция. 10 — хороший баланс.",
        key="webcam_interval",
    )
    st.caption(f"🔧 Модель: **{model_name}** | Анализ каждого {webcam_interval}-го кадра")

    # Общее состояние между потоком Streamlit и потоком WebRTC
    class WebcamState:
        def __init__(self):
            self.lock = threading.Lock()
            self.frame_count = 0
            self.last_results = []
            self.detector = None
            self.model_key = "deepface"
            self.interval = 10

    if "webcam_state" not in st.session_state:
        st.session_state.webcam_state = WebcamState()

    state = st.session_state.webcam_state
    state.model_key = model_key
    state.interval = webcam_interval
    state.detector = load_face_detector()

    def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        with state.lock:
            state.frame_count += 1
            current_frame = state.frame_count
            interval = state.interval
            current_model = state.model_key
            detector = state.detector

        # Детекция лиц — каждый кадр (быстрая операция)
        faces = face_detection.detect_all_faces(img, detector)

        if faces:
            if current_frame % interval == 0:
                # Анализ эмоций — только каждый N-й кадр
                new_results = []
                for (x, y, w, h) in faces:
                    analysis = analyze_face(img[y:y + h, x:x + w], current_model)
                    if analysis is not None:
                        dominant, score, _ = analysis
                        new_results.append(((x, y, w, h), dominant, score))
                with state.lock:
                    state.last_results = new_results
            else:
                # Между анализами — обновляем позиции лиц, сохраняем эмоции
                with state.lock:
                    old_results = state.last_results
                updated = [(box, old_results[i][1], old_results[i][2])
                           for i, box in enumerate(faces) if i < len(old_results)]
                if updated:
                    with state.lock:
                        state.last_results = updated

            with state.lock:
                results_to_draw = list(state.last_results)
            if results_to_draw:
                img = draw_overlay(img, results_to_draw)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_streamer(
        key="emotion-webcam",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        sendback_audio=False,
    )

    st.info("💡 Нажмите **START** чтобы включить камеру. Эмоции определяются в реальном времени.")

# ========== ИНФО ==========
with st.expander("ℹ️ О моделях и технологиях"):
    st.write("""
    **Детекция лиц:** MediaPipe BlazeFace (Google). Быстрый и точный детектор, находит все лица в кадре.

    **DeepFace (CNN):** сверточная нейросеть на основе FER2013. Базовая модель первой итерации.

    **ViT (Vision Transformer):** модель `trpakov/vit-face-expression`.

    **Классы эмоций:** 6 классов, как в датасете CREMA-D — злость, отвращение, страх, радость,
    грусть, нейтральное. Класс «удивление» отбрасывается, вероятности перенормируются.

    **Обработка нескольких лиц:** каждое лицо анализируется отдельно.

    **Веб-камера:** анализ эмоций в реальном времени через WebRTC. Детекция лиц каждый кадр,
    анализ эмоций — с настраиваемой частотой.

    **PDF-отчёты:** график эмоций и ключевые кадры.

    **Ограничения:** модели обучены на фронтальных лицах (FER2013). На сильно повёрнутых лицах
    точность снижается.
    """)
