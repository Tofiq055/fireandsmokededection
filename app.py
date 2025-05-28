import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
import os
from dcp_defense import dcp_perturb
import numpy as np

# --- Sayfa Ayarları ---
st.set_page_config(page_title="Yangın & Duman Tespiti", layout="wide")

# --- Başlık ---
st.title("🔥 Yangın ve Duman Tespiti (YOLOv8)")
st.write("Bir video yükleyin. Model, videodaki yangın ve dumanı tespit ederek kareler üzerinde gösterecek ve ilk tespit anlarını kaydedecektir.")

# --- Model Yükleme ---
MODEL_PATH = "best.pt"

@st.cache_resource
def load_model(path):
    try:
        model = YOLO(path)
        return model, model.names
    except Exception as e:
        st.error(f"Model yüklenirken hata oluştu: {e}")
        st.error(f"Lütfen '{path}' dosyasının doğru yerde olduğundan ve geçerli bir YOLO modeli olduğundan emin olun.")
        return None, None

model, class_names = load_model(MODEL_PATH)

if not model:
    st.stop()

# --- Renk Tanımlamaları ---
COLOR_FIRE = (0, 0, 255)
COLOR_SMOKE = (0, 255, 255)
COLOR_DEFAULT = (0, 255, 0)

# --- Video Yükleyici ---
uploaded_file = st.file_uploader("🎥 Video dosyasını seçin", type=["mp4", "mov", "avi", "mkv"])

# --- Ana İşlem Bloğu ---
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tfile:
        tfile.write(uploaded_file.read())
        temp_video_path = tfile.name

    st.info("Video işleniyor, lütfen bekleyin...")

    cap = cv2.VideoCapture(temp_video_path)
    if not cap.isOpened():
        st.error("Video dosyası açılamadı. Lütfen geçerli bir video dosyası yükleyin.")
    else:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 25
            st.warning(f"Video FPS değeri okunamadı, varsayılan olarak {fps} kullanılıyor.")

        stframe = st.empty()
        first_fire_detection_time_placeholder = st.empty()
        first_smoke_detection_time_placeholder = st.empty()

        first_fire_detection_time = None
        first_smoke_detection_time = None
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            current_time_sec = frame_count / fps

            results = model.predict(frame, conf=0.3, verbose=False)

            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls_index = int(box.cls[0])
                    label = class_names.get(cls_index, f"Class_{cls_index}")
                    confidence = box.conf[0]
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label_lower = label.lower()
                    color_to_use = COLOR_DEFAULT

                    if label_lower == 'fire':
                        color_to_use = COLOR_FIRE
                        if first_fire_detection_time is None:
                            first_fire_detection_time = current_time_sec
                            first_fire_detection_time_placeholder.success(
                                f"🔥 İlk 'YANGIN' tespiti! Zaman: {first_fire_detection_time:.2f} saniye"
                            )
                    elif label_lower == 'smoke':
                        color_to_use = COLOR_SMOKE
                        if first_smoke_detection_time is None:
                            first_smoke_detection_time = current_time_sec
                            first_smoke_detection_time_placeholder.success(
                                f"💨 İlk 'DUMAN' tespiti! Zaman: {first_smoke_detection_time:.2f} saniye"
                            )

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color_to_use, 2)
                    cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_to_use, 2)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(rgb_frame, channels="RGB", use_column_width=True)

        cap.release()

        if first_fire_detection_time is None:
            st.warning("Videoda yangın tespit edilemedi.")
        if first_smoke_detection_time is None:
            st.warning("Videoda duman tespit edilemedi.")

    if os.path.exists(temp_video_path):
        os.remove(temp_video_path)
else:
    st.info("Lütfen işlemek için bir video dosyası yükleyin.")

st.sidebar.info(
    """
    **Nasıl Kullanılır?**
    1.  `best.pt` model dosyanızın bu script ile aynı klasörde olduğundan veya yukarıdaki `MODEL_PATH` değişkeninde doğru yolu belirttiğinizden emin olun.
    2.  "Video dosyasını seçin" butonu ile videonuzu yükleyin.
    3.  Model videoyu işleyecek ve tespitleri gösterecektir.
    """
)
