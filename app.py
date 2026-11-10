
import streamlit as st
from PIL import Image, ImageDraw
from ultralytics import YOLO
from streamlit_image_coordinates import streamlit_image_coordinates
from streamlit_cropper import st_cropper
import numpy as np
import io
import zipfile
import csv
import hashlib
from datetime import datetime

st.set_page_config(
    page_title="PINEAPPLE",
    page_icon="🍍",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .block-container {
        max-width: 900px;
        padding-top: 0.8rem;
        padding-bottom: 1rem;
        padding-left: 0.8rem;
        padding-right: 0.8rem;
    }
    h1 {
        text-align: center;
        font-size: 1.8rem !important;
        margin-bottom: 0.2rem;
    }
    h2, h3 {
        font-size: 1.15rem !important;
    }
    div.stButton > button,
    div.stDownloadButton > button {
        width: 100%;
        min-height: 44px;
        border-radius: 10px;
        font-weight: 600;
    }
    div[data-testid="stMetric"] {
        background: rgba(128,128,128,0.08);
        padding: 10px;
        border-radius: 10px;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.6rem;
    }
    [data-testid="stFileUploader"] {
        padding: 0;
    }
    .stRadio > div {
        gap: 0.5rem;
    }
    footer {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return YOLO("last_openvino_model")


model = load_model()

TARGET_SIZE = 960
CONFIDENCE = 0.5
MAX_DETECTIONS = 2000


def resize_image(img, max_size=TARGET_SIZE):
    w, h = img.size
    scale = min(max_size / max(w, h), 1.0)
    if scale < 1.0:
        return img.resize(
            (int(w * scale), int(h * scale)),
            Image.Resampling.LANCZOS
        )
    return img.copy()


def draw_result_image(base_image):
    result = base_image.copy()
    draw = ImageDraw.Draw(result)

    for box in st.session_state.auto_boxes:
        x1, y1, x2, y2 = box
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        r = 9
        draw.ellipse(
            (cx-r-1, cy-r-1, cx+r+1, cy+r+1),
            fill="black"
        )
        draw.ellipse(
            (cx-r, cy-r, cx+r, cy+r),
            fill="#FFD700"
        )

    for px, py in st.session_state.manual_points:
        r = 9
        draw.ellipse(
            (px-r-1, py-r-1, px+r+1, py+r+1),
            fill="white"
        )
        draw.ellipse(
            (px-r, py-r, px+r, py+r),
            fill="#00A8FF"
        )

    return result


def initialize_state():
    defaults = {
        "phase": "upload",
        "source_hash": None,
        "original_image": None,
        "cropped_image": None,
        "auto_boxes": [],
        "manual_points": [],
        "deleted_auto_count": 0,
        "last_click": None,
        "group_name": "A",
        "comment": "",
        "result_timestamp": None
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_detection():
    st.session_state.auto_boxes = []
    st.session_state.manual_points = []
    st.session_state.deleted_auto_count = 0
    st.session_state.last_click = None
    st.session_state.result_timestamp = None


def start_detection(image):
    inference_image = resize_image(image)

    with st.spinner("AIがパイプを検出しています..."):
        results = model.predict(
            inference_image,
            conf=CONFIDENCE,
            max_det=MAX_DETECTIONS,
            verbose=False
        )

    boxes = []

    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
        boxes.append([int(x1), int(y1), int(x2), int(y2)])

    st.session_state.cropped_image = inference_image
    st.session_state.auto_boxes = boxes
    st.session_state.manual_points = []
    st.session_state.deleted_auto_count = 0
    st.session_state.last_click = None
    st.session_state.result_timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )
    st.session_state.phase = "result"


def remove_nearest_point(x, y, threshold=30):
    nearest_index = None
    nearest_distance = threshold

    for i, box in enumerate(st.session_state.auto_boxes):
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        distance = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5

        if distance < nearest_distance:
            nearest_distance = distance
            nearest_index = i

    if nearest_index is not None:
        st.session_state.auto_boxes.pop(nearest_index)
        st.session_state.deleted_auto_count += 1
        return

    nearest_index = None
    nearest_distance = threshold

    for i, point in enumerate(st.session_state.manual_points):
        distance = ((x - point[0]) ** 2 + (y - point[1]) ** 2) ** 0.5

        if distance < nearest_distance:
            nearest_distance = distance
            nearest_index = i

    if nearest_index is not None:
        st.session_state.manual_points.pop(nearest_index)


def create_zip():
    timestamp = st.session_state.result_timestamp
    result_image = draw_result_image(st.session_state.cropped_image)

    total = (
        len(st.session_state.auto_boxes)
        + len(st.session_state.manual_points)
    )

    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(
        zip_buffer,
        "w",
        compression=zipfile.ZIP_DEFLATED
    ) as archive:

        image_buffer = io.BytesIO()
        result_image.save(image_buffer, format="PNG")

        archive.writestr(
            f"result_{timestamp}.png",
            image_buffer.getvalue()
        )

        csv_buffer = io.StringIO()
        writer = csv.writer(csv_buffer)

        writer.writerow([
            "日時",
            "グループ",
            "AI検出数",
            "手動追加数",
            "手動削減数",
            "最終合計",
            "備考"
        ])

        writer.writerow([
            timestamp,
            st.session_state.group_name,
            len(st.session_state.auto_boxes),
            len(st.session_state.manual_points),
            st.session_state.deleted_auto_count,
            total,
            st.session_state.comment
        ])

        archive.writestr(
            f"result_{timestamp}.csv",
            csv_buffer.getvalue().encode("utf-8-sig")
        )

    zip_buffer.seek(0)
    return zip_buffer.getvalue()


initialize_state()

st.title("PINEAPPLE 🍍")

if st.session_state.phase == "upload":

    st.caption("AIによるパイプ本数カウント")

    camera_file = st.camera_input(
        "📷 カメラで撮影",
        key="camera_input"
    )

    uploaded_file = st.file_uploader(
        "または画像をアップロード",
        type=["jpg", "jpeg", "png"],
        key="image_upload"
    )

    selected_file = camera_file if camera_file is not None else uploaded_file

    if selected_file is not None:
        file_bytes = selected_file.getvalue()
        current_hash = hashlib.sha256(file_bytes).hexdigest()

        if current_hash != st.session_state.source_hash:
            st.session_state.source_hash = current_hash
            st.session_state.original_image = Image.open(
                io.BytesIO(file_bytes)
            ).convert("RGB")
            st.session_state.cropped_image = None
            reset_detection()
            st.session_state.phase = "crop"

            st.rerun()

elif st.session_state.phase == "crop":

    st.subheader("✂️ 検出エリアの調整")
    st.caption("画像の四隅を動かして、数えたい範囲を選択してください。")

    original = resize_image(st.session_state.original_image)

    cropped = st_cropper(
        original,
        realtime_update=True,
        box_color="#FFD700",
        aspect_ratio=None,
        return_type="image",
        should_resize_image=True
    )

    st.caption("選択した範囲だけをAIで検出します。")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔄 元画像に戻す"):
            st.session_state.phase = "upload"
            st.session_state.source_hash = None
            st.rerun()

    with col2:
        if st.button("🚀 検出開始", type="primary"):
            if cropped is not None and cropped.size[0] > 0:
                start_detection(cropped)
                st.rerun()

elif st.session_state.phase == "result":

    st.subheader("📊 検出結果")

    auto_count = len(st.session_state.auto_boxes)
    manual_count = len(st.session_state.manual_points)
    total_count = auto_count + manual_count

    col1, col2 = st.columns(2)

    with col1:
        st.metric("合計本数", f"{total_count} 本")

    with col2:
        st.metric("AI自動検出", f"{auto_count} 本")

    st.caption(
        f"手動追加：{manual_count} 本　"
        f"手動削減：{st.session_state.deleted_auto_count} 本"
    )

    result_image = draw_result_image(
        st.session_state.cropped_image
    )

    st.image(
        result_image,
        caption="黄色：AI検出 / 青色：手動追加",
        use_container_width=True
    )

    st.markdown("**検出位置の修正**")

    mode = st.radio(
        "操作方法",
        ["📍 追加", "🧽 削除"],
        horizontal=True,
        label_visibility="collapsed",
        key="edit_mode"
    )

    st.caption("追加：パイプの中心をタップ / 削除：マーカーをタップ")

    click_image = streamlit_image_coordinates(
        result_image,
        key="manual_fix",
        width=result_image.width
    )

    if click_image:
        current_click = (
            int(click_image["x"]),
            int(click_image["y"])
        )

        if current_click != st.session_state.last_click:
            st.session_state.last_click = current_click

            if "追加" in mode:
                st.session_state.manual_points.append(current_click)
            else:
                remove_nearest_point(
                    current_click[0],
                    current_click[1]
                )

            st.rerun()

    col1, col2 = st.columns(2)

    with col1:
        if st.button("↩️ 手動追加を全消去"):
            st.session_state.manual_points = []
            st.session_state.last_click = None
            st.rerun()

    with col2:
        if st.button("🔄 最初からやり直す"):
            reset_detection()
            st.session_state.phase = "upload"
            st.session_state.source_hash = None
            st.rerun()

    st.divider()

    st.subheader("💾 結果の保存")

    col1, col2 = st.columns(2)

    with col1:
        st.session_state.group_name = st.selectbox(
            "グループ",
            ["A", "B", "C", "D"],
            key="group_select"
        )

    with col2:
        st.session_state.comment = st.text_input(
            "備考",
            value=st.session_state.comment,
            key="comment_input"
        )

    zip_data = create_zip()

    st.download_button(
        label="⬇️ 判定結果をZIPで保存",
        data=zip_data,
        file_name=f"pipe_report_{st.session_state.result_timestamp}.zip",
        mime="application/zip",
        type="primary",
        use_container_width=True
    )
