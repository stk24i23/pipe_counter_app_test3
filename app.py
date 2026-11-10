import csv
import hashlib
import io
import zipfile
from datetime import datetime

import streamlit as st
from PIL import Image, ImageDraw
from ultralytics import YOLO
from streamlit_cropper import st_cropper
from streamlit_image_coordinates import streamlit_image_coordinates


st.set_page_config(
    page_title="PINEAPPLE",
    page_icon="🍍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

TARGET_SIZE = 720
MODEL_PATH = "last_openvino_model"
CONFIDENCE = 0.5
MAX_DETECTIONS = 2000
CLICK_THRESHOLD = 30


st.markdown(
    """
    <style>
    .block-container {
        max-width: 1180px;
        padding-top: 0.8rem;
        padding-bottom: 1.2rem;
        padding-left: max(1rem, 3vw);
        padding-right: max(1rem, 3vw);
    }
    h1 {
        text-align: center;
        font-size: clamp(1.55rem, 3vw, 2.1rem) !important;
        margin-bottom: 0.15rem;
    }
    [data-testid="stMetric"] {
        background: rgba(128, 128, 128, 0.08);
        border-radius: 12px;
        padding: 0.65rem;
    }
    [data-testid="stMetricValue"] {
        font-size: clamp(1.25rem, 3vw, 1.8rem);
    }
    div.stButton > button,
    div.stDownloadButton > button {
        min-height: 46px;
        border-radius: 10px;
        font-weight: 600;
    }
    div[data-testid="stFileUploader"] {
        width: 100%;
    }
    @media (max-width: 700px) {
        .block-container {
            padding-top: 0.45rem;
            padding-left: 0.65rem;
            padding-right: 0.65rem;
        }
        h1 {
            font-size: 1.55rem !important;
        }
        [data-testid="stMetric"] {
            padding: 0.45rem;
        }
        [data-testid="stMetricLabel"] {
            font-size: 0.78rem;
        }
    }
    footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


def init_state():
    defaults = {
        "phase": "upload",
        "source_hash": None,
        "original_image": None,
        "working_image": None,
        "mask_rects": [],
        "mask_first_point": None,
        "auto_boxes": [],
        "manual_points": [],
        "deleted_auto_count": 0,
        "last_mask_click": None,
        "last_result_click": None,
        "result_timestamp": None,
        "group_name": "A",
        "comment": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def resize_image(image, max_side=TARGET_SIZE):
    image = image.convert("RGB")
    width, height = image.size
    scale = min(max_side / max(width, height), 1.0)
    if scale >= 1.0:
        return image.copy()
    return image.resize(
        (max(1, int(width * scale)), max(1, int(height * scale))),
        Image.Resampling.LANCZOS,
    )


def reset_result():
    st.session_state.auto_boxes = []
    st.session_state.manual_points = []
    st.session_state.deleted_auto_count = 0
    st.session_state.last_result_click = None
    st.session_state.result_timestamp = None


def reset_to_upload():
    st.session_state.phase = "upload"
    st.session_state.source_hash = None
    st.session_state.original_image = None
    st.session_state.working_image = None
    st.session_state.mask_rects = []
    st.session_state.mask_first_point = None
    st.session_state.last_mask_click = None
    reset_result()


def image_with_masks(image, rectangles, darken=False):
    result = image.copy()
    draw = ImageDraw.Draw(result)
    for x1, y1, x2, y2 in rectangles:
        if darken:
            draw.rectangle((x1, y1, x2, y2), fill=(45, 45, 45))
        else:
            draw.rectangle(
                (x1, y1, x2, y2),
                outline="#FF4B4B",
                width=3,
            )
    return result


def draw_result():
    image = st.session_state.working_image.copy()
    draw = ImageDraw.Draw(image)

    for x1, y1, x2, y2 in st.session_state.auto_boxes:
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        radius = 10
        draw.ellipse(
            (cx - radius - 2, cy - radius - 2,
             cx + radius + 2, cy + radius + 2),
            fill="black",
        )
        draw.ellipse(
            (cx - radius, cy - radius,
             cx + radius, cy + radius),
            fill="#FFD400",
        )

    for x, y in st.session_state.manual_points:
        radius = 10
        draw.ellipse(
            (x - radius - 2, y - radius - 2,
             x + radius + 2, y + radius + 2),
            fill="white",
        )
        draw.ellipse(
            (x - radius, y - radius,
             x + radius, y + radius),
            fill="#00A8FF",
        )

    return image


def run_detection(image):
    model = load_model()
    prepared = resize_image(image, TARGET_SIZE)
    inference_image = image_with_masks(
        prepared,
        st.session_state.mask_rects,
        darken=True,
    )

    with st.spinner("AIがパイプを検出しています..."):
        results = model.predict(
            inference_image,
            conf=CONFIDENCE,
            max_det=MAX_DETECTIONS,
            verbose=False,
        )

    boxes = []
    if results and results[0].boxes is not None:
        for box in results[0].boxes:
            coords = box.xyxy[0].cpu().numpy().tolist()
            x1, y1, x2, y2 = coords
            boxes.append([
                int(round(x1)),
                int(round(y1)),
                int(round(x2)),
                int(round(y2)),
            ])

    st.session_state.working_image = prepared
    st.session_state.auto_boxes = boxes
    st.session_state.manual_points = []
    st.session_state.deleted_auto_count = 0
    st.session_state.last_result_click = None
    st.session_state.result_timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )
    st.session_state.phase = "result"


def delete_nearest(x, y):
    nearest_index = None
    nearest_distance = CLICK_THRESHOLD

    for index, box in enumerate(st.session_state.auto_boxes):
        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2
        distance = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
        if distance < nearest_distance:
            nearest_distance = distance
            nearest_index = index

    if nearest_index is not None:
        st.session_state.auto_boxes.pop(nearest_index)
        st.session_state.deleted_auto_count += 1
        return

    nearest_index = None
    nearest_distance = CLICK_THRESHOLD

    for index, point in enumerate(st.session_state.manual_points):
        distance = ((x - point[0]) ** 2 + (y - point[1]) ** 2) ** 0.5
        if distance < nearest_distance:
            nearest_distance = distance
            nearest_index = index

    if nearest_index is not None:
        st.session_state.manual_points.pop(nearest_index)


def make_zip():
    timestamp = st.session_state.result_timestamp
    final_image = draw_result()
    auto_count = len(st.session_state.auto_boxes)
    manual_count = len(st.session_state.manual_points)
    total_count = auto_count + manual_count

    image_buffer = io.BytesIO()
    final_image.save(image_buffer, format="PNG")

    csv_buffer = io.StringIO(newline="")
    writer = csv.writer(csv_buffer)
    writer.writerow([
        "日時",
        "グループ",
        "AI自動検出数",
        "手動追加数",
        "手動削減数",
        "最終合計",
        "備考",
    ])
    writer.writerow([
        timestamp,
        st.session_state.group_name,
        auto_count,
        manual_count,
        st.session_state.deleted_auto_count,
        total_count,
        st.session_state.comment,
    ])

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(
        zip_buffer,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        archive.writestr(
            f"result_{timestamp}.png",
            image_buffer.getvalue(),
        )
        archive.writestr(
            f"result_{timestamp}.csv",
            csv_buffer.getvalue().encode("utf-8-sig"),
        )

    return zip_buffer.getvalue()


init_state()

st.title("PINEAPPLE 🍍")
st.caption("AIによるパイプ本数カウント")


if st.session_state.phase == "upload":
    st.subheader("画像を選択")

    camera_file = st.camera_input(
        "カメラで撮影する",
        key="camera_input",
    )
    uploaded_file = st.file_uploader(
        "画像ファイルをアップロード",
        type=["jpg", "jpeg", "png"],
        key="image_upload",
    )

    selected_file = (
        camera_file if camera_file is not None else uploaded_file
    )

    if selected_file is not None:
        file_bytes = selected_file.getvalue()
        file_hash = hashlib.sha256(file_bytes).hexdigest()

        if file_hash != st.session_state.source_hash:
            try:
                source_image = Image.open(
                    io.BytesIO(file_bytes)
                ).convert("RGB")
                st.session_state.source_hash = file_hash
                st.session_state.original_image = resize_image(source_image)
                st.session_state.mask_rects = []
                st.session_state.mask_first_point = None
                st.session_state.last_mask_click = None
                reset_result()
                st.session_state.phase = "crop"
                st.rerun()
            except Exception as error:
                st.error(f"画像を読み込めませんでした: {error}")

    st.info("撮影または画像アップロード後、検出範囲を調整します。")


elif st.session_state.phase == "crop":
    st.subheader("① 検出範囲をトリミング")

    st.caption(
        "画像の枠を動かして、数えたいパイプだけが入る範囲を選択してください。"
    )

    cropped_image = st_cropper(
        st.session_state.original_image,
        realtime_update=True,
        box_color="#FFD400",
        aspect_ratio=None,
        return_type="image",
        should_resize_image=True,
    )

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        if st.button("画像選択に戻る", use_container_width=True):
            reset_to_upload()
            st.rerun()

    with col2:
        if st.button(
            "トリミングして検出",
            type="primary",
            use_container_width=True,
        ):
            if cropped_image is not None:
                st.session_state.mask_rects = []
                run_detection(cropped_image)
                st.rerun()


elif st.session_state.phase == "result":
    st.subheader("② 検出結果の確認・修正")

    auto_count = len(st.session_state.auto_boxes)
    manual_count = len(st.session_state.manual_points)
    total_count = auto_count + manual_count

    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("合計本数", f"{total_count} 本")
    metric2.metric("AI検出", f"{auto_count} 本")
    metric3.metric("手動追加", f"{manual_count} 本")

    st.caption(
        f"手動削減：{st.session_state.deleted_auto_count} 本"
    )

    result_image = draw_result()

    st.image(
        result_image,
        caption="黄色：AI検出　青色：手動追加",
        use_container_width=True,
    )

    st.markdown("**画像をタップして検出結果を修正**")
    edit_mode = st.radio(
        "操作モード",
        ["追加", "削除"],
        horizontal=True,
        label_visibility="collapsed",
        key="edit_mode",
    )

    click_image = streamlit_image_coordinates(
        result_image,
        key="result_image_click",
        width=result_image.width,
    )

    if click_image:
        current_click = (
            int(click_image["x"]),
            int(click_image["y"]),
        )
        if current_click != st.session_state.last_result_click:
            st.session_state.last_result_click = current_click

            if edit_mode == "追加":
                st.session_state.manual_points.append(current_click)
            else:
                delete_nearest(*current_click)

            st.rerun()

    st.caption(
        "追加：検出されていないパイプの中心をタップ。"
        "削除：消したいマーカーの近くをタップ。"
    )

    with st.expander("検出結果の修正・やり直し"):
        col1, col2 = st.columns(2)

        with col1:
            if st.button(
                "手動追加を全て消去",
                use_container_width=True,
            ):
                st.session_state.manual_points = []
                st.session_state.last_result_click = None
                st.rerun()

        with col2:
            if st.button(
                "この画像で再検出",
                use_container_width=True,
            ):
                st.session_state.mask_rects = []
                run_detection(st.session_state.working_image)
                st.rerun()

    st.divider()
    st.subheader("③ 結果を保存")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.session_state.group_name = st.selectbox(
            "グループ",
            ["A", "B", "C", "D"],
            key="group_select",
        )

    with col2:
        st.session_state.comment = st.text_input(
            "備考",
            key="comment_input",
        )

    zip_data = make_zip()

    st.download_button(
        "ZIPで判定結果を保存",
        data=zip_data,
        file_name=f"pipe_report_{st.session_state.result_timestamp}.zip",
        mime="application/zip",
        type="primary",
        use_container_width=True,
    )

    if st.button(
        "最初に戻る（別の画像を選択）",
        use_container_width=True,
    ):
        reset_to_upload()
        st.rerun()
