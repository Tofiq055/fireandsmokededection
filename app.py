import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
import os

# --- Page Settings ---
st.set_page_config(page_title="Fire & Smoke Detection", layout="wide")

# --- Title ---
st.title("🔥 Fire and Smoke Detection (YOLOv8)")
st.write(
    "Upload a video. The model will detect fire and smoke, display bounding boxes, "
    "and record the first detection times."
)

# --- Model Loading ---
MODEL_PATH = "best.pt"  # Make sure your model is in the same directory

@st.cache_resource
def load_model(path):
    try:
        model = YOLO(path)
        return model, model.names
    except Exception as e:
        return None, None, str(e)

model, class_names, model_error = load_model(MODEL_PATH) + (None,) if load_model(MODEL_PATH) else (None, None, "Model could not be loaded.")

if not model:
    st.error(f"Error loading model: {model_error}")
    st.stop()

# --- Color Definitions ---
COLOR_FIRE = (0, 0, 255)      # Red for fire
COLOR_SMOKE = (0, 255, 255)   # Yellow for smoke
COLOR_DEFAULT = (0, 255, 0)   # Green for other classes

# --- Video Uploader ---
uploaded_file = st.file_uploader("🎥 Select a video file", type=["mp4", "mov", "avi", "mkv"])

# --- Main Processing Block ---
if uploaded_file is not None:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tfile:
            tfile.write(uploaded_file.read())
            temp_video_path = tfile.name

        st.info("Processing video, please wait...")

        cap = cv2.VideoCapture(temp_video_path)
        if not cap.isOpened():
            st.error("Could not open video file. Please upload a valid video.")
        else:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0 or fps is None:
                fps = 25
                st.warning(f"Could not read video FPS, using default: {fps}")

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

                try:
                    results = model.predict(frame, conf=0.3, verbose=False)
                except Exception as e:
                    st.error(f"Error during prediction: {e}")
                    break

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
                                    f"🔥 First 'FIRE' detected at {first_fire_detection_time:.2f} seconds"
                                )
                        elif label_lower == 'smoke':
                            color_to_use = COLOR_SMOKE
                            if first_smoke_detection_time is None:
                                first_smoke_detection_time = current_time_sec
                                first_smoke_detection_time_placeholder.success(
                                    f"💨 First 'SMOKE' detected at {first_smoke_detection_time:.2f} seconds"
                                )

                        cv2.rectangle(frame, (x1, y1), (x2, y2), color_to_use, 2)
                        cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_to_use, 2)

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                stframe.image(rgb_frame, channels="RGB", use_container_width=True)

            cap.release()

            if first_fire_detection_time is None:
                st.warning("No fire detected in the video.")
            if first_smoke_detection_time is None:
                st.warning("No smoke detected in the video.")

    except Exception as e:
        st.error(f"An error occurred: {e}")

    finally:
        if 'temp_video_path' in locals() and os.path.exists(temp_video_path):
            os.remove(temp_video_path)
else:
    st.info("Please upload a video file to process.")

st.sidebar.info(
    """
    **How to Use:**
    1. Ensure your `best.pt` model file is in the same folder as this script, or update the `MODEL_PATH` variable.
    2. Upload your video using the uploader above.
    3. The model will process the video and display detections.
    """
) 