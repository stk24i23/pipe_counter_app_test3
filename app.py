
import os
import io
import csv
import zipfile
import hashlib
from pathlib import Path
from datetime import datetime

import streamlit as st
from PIL import Image, ImageDraw, ImageOps
from ultralytics import YOLO
from streamlit_cropper import st_cropper
from streamlit_image_coordinates import streamlit_image_coordinates


st.set_page_config(
    page_title="PINEAPPLE",
    page_icon="🍍",
    layout="wide"
)


MODEL_CANDIDATES = [
    "last_openvino_model",
    "best_openvino_model",
    "best.pt"
]

MAX_IMAGE_SIZE = 1280
MAX_DISPLAY_WIDTH = 1000


def initialize_state():
    defaults = {
        "source_hash": None,
        "source_image": None,
        "work_image": None,
        "markers": [],
        "automatic_count": 0,
        "manual_added": 0,
        "manual_deleted": 0,
        "detected": False,
        "click_version": 0,
        "saved_records": []
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


initialize_state()


def find_model_path():
    configured_path = os.getenv("MODEL_PATH")

    if configured_path:
        return Path(configured_path)

    for candidate in MODEL_CANDIDATES:
        path = Path(candidate)

        if path.exists():
            return path

    return None


def resize_image(image, max_size=MAX_IMAGE_SIZE):
    image = image.convert("RGB")

    image.thumbnail(
        (max_size, max_size),
        Image.Resampling.LANCZOS
    )

    return image


@st.cache_resource
def load_model(model_path):
    return YOLO(str(model_path), task="detect")


def image_to_png_bytes(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def draw_markers(image, markers):
    result = image.copy()
    draw = ImageDraw.Draw(result)

    width, height = result.size

    radius = max(5, int(width / 150))
    outline_width = max(2, int(width / 400))

    for marker in markers:
        x = int(marker["x"])
        y = int(marker["y"])

        draw.ellipse(
            (
                x - radius,
                y - radius,
                x + radius,
                y + radius
            ),
            fill=None,
            outline="#FFFF00",
            width=outline_width
        )

    return result


def run_detection(image, model, confidence):
    results = model.predict(
        source=image,
        conf=confidence,
        imgsz=640,
        verbose=False
    )

    markers = []

    for result in results:
        if result.boxes is None:
            continue

        boxes = result.boxes.xyxy.cpu().numpy()

        for box in boxes:
            x1, y1, x2, y2 = box

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            markers.append({
                "x": float(center_x),
                "y": float(center_y),
                "kind": "auto"
            })

    return markers


def reset_detection():
    st.session_state.markers = []
    st.session_state.automatic_count = 0
    st.session_state.manual_added = 0
    st.session_state.manual_deleted = 0
    st.session_state.detected = False
    st.session_state.click_version += 1


def handle_image_click(coords, display_width, display_height):
    if coords is None:
        return

    if not st.session_state.detected:
        return

    original_image = st.session_state.work_image

    original_width, original_height = original_image.size

    x = coords["x"] * original_width / display_width
    y = coords["y"] * original_height / display_height

    mode = st.session_state.edit_mode

    if mode == "追加":
        st.session_state.markers.append({
            "x": x,
            "y": y,
            "kind": "manual"
        })

        st.session_state.manual_added += 1

    elif mode == "削除":
        if not st.session_state.markers:
            return

        nearest_index = None
        nearest_distance = float("inf")

        delete_range = max(25, original_width * 0.025)

        for index, marker in enumerate(st.session_state.markers):
            dx = marker["x"] - x
            dy = marker["y"] - y

            distance = (dx ** 2 + dy ** 2) ** 0.5

            if distance < nearest_distance:
                nearest_distance = distance
                nearest_index = index

        if nearest_index is not None and nearest_distance <= delete_range:
            st.session_state.markers.pop(nearest_index)
            st.session_state.manual_deleted += 1


def create_csv_bytes(records):
    buffer = io.StringIO(newline="")

    fieldnames = [
        "日時",
        "グループ",
        "自動検出本数",
        "手動追加本数",
        "手動削減本数",
        "合計本数",
        "コメント"
    ]

    writer = csv.DictWriter(
        buffer,
        fieldnames=fieldnames
    )

    writer.writeheader()

    for record in records:
        writer.writerow(record["data"])

    return buffer.getvalue().encode("utf-8-sig")


def create_zip_bytes(records):
    buffer = io.BytesIO()

    with zipfile.ZipFile(
        buffer,
        "w",
        compression=zipfile.ZIP_DEFLATED
    ) as archive:

        csv_data = create_csv_bytes(records)

        archive.writestr(
            "pipe_count.csv",
            csv_data
        )

        for index, record in enumerate(records, start=1):
            filename = f"results/result_{index:03d}.png"

            archive.writestr(
                filename,
                record["image"]
            )

    return buffer.getvalue()


st.title("🍍 PINEAPPLE")
st.caption("AIによるパイプ本数カウントアプリケーション")

st.divider()


st.header("1. 画像の取り込み")

camera_tab, upload_tab = st.tabs([
    "📷 カメラで撮影",
    "📁 画像をアップロード"
])

with camera_tab:
    camera_image = st.camera_input(
        "パイプの写真を撮影してください",
        key="camera_input"
    )

with upload_tab:
    uploaded_image = st.file_uploader(
        "画像ファイルを選択してください",
        type=["jpg", "jpeg", "png"],
        key="file_input"
    )


input_image = uploaded_image if uploaded_image is not None else camera_image

if input_image is not None:
    image_bytes = input_image.getvalue()
    current_hash = hashlib.sha256(image_bytes).hexdigest()

    if current_hash != st.session_state.source_hash:
        image = Image.open(io.BytesIO(image_bytes))
        image = ImageOps.exif_transpose(image)
        image = resize_image(image)

        st.session_state.source_hash = current_hash
        st.session_state.source_image = image.copy()
        st.session_state.work_image = image.copy()

        reset_detection()


if st.session_state.source_image is not None:

    st.divider()
    st.header("2. 検出範囲の調整")

    with st.expander("トリミングを行う", expanded=True):

        crop_candidate = st_cropper(
            st.session_state.source_image,
            realtime_update=False,
            box_color="#FFFF00",
            aspect_ratio=None,
            return_type="image"
        )

        if st.button(
            "トリミング範囲を確定",
            use_container_width=True
        ):

            if crop_candidate is not None:
                st.session_state.work_image = crop_candidate.convert("RGB")
                reset_detection()

                st.success("トリミングを確定しました。")
                st.rerun()


    st.subheader("検出に使用する画像")

    st.image(
        st.session_state.work_image,
        use_container_width=True
    )

    st.divider()
    st.header("3. AIによるパイプ検出")

    model_path = find_model_path()

    if model_path is None:
        st.error(
            "AIモデルが見つかりません。"
            "モデルファイルをGitHubにアップロードしてください。"
        )

    else:
        st.caption(f"使用モデル：{model_path}")

        confidence = st.slider(
            "検出信頼度",
            min_value=0.05,
            max_value=0.95,
            value=0.25,
            step=0.05
        )

        if st.button(
            "🔍 パイプ検出開始",
            type="primary",
            use_container_width=True
        ):

            try:
                with st.spinner("AIがパイプを検出しています..."):

                    model = load_model(str(model_path))

                    markers = run_detection(
                        st.session_state.work_image,
                        model,
                        confidence
                    )

                    st.session_state.markers = markers
                    st.session_state.automatic_count = len(markers)
                    st.session_state.manual_added = 0
                    st.session_state.manual_deleted = 0
                    st.session_state.detected = True
                    st.session_state.click_version += 1

                st.success("パイプの検出が完了しました。")

            except Exception as error:
                st.error(f"検出中にエラーが発生しました：{error}")


if st.session_state.detected:

    st.divider()
    st.header("4. 検出結果の確認と補正")

    total_count = len(st.session_state.markers)

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "自動検出本数",
        st.session_state.automatic_count
    )

    col2.metric(
        "手動追加本数",
        st.session_state.manual_added
    )

    col3.metric(
        "現在の合計本数",
        total_count
    )

    st.write(
        f"手動削除本数：{st.session_state.manual_deleted} 本"
    )

    st.divider()

    st.subheader("検出マーカーの補正")

    st.session_state.edit_mode = st.radio(
        "補正方法を選択してください",
        ["追加", "削除"],
        horizontal=True,
        key="edit_mode_radio"
    )

    if st.session_state.edit_mode == "追加":
        st.info("検出できなかったパイプの中心を画像上でクリックしてください。")

    else:
        st.info("削除したい黄色のマーカー付近を画像上でクリックしてください。")

    marked_image = draw_markers(
        st.session_state.work_image,
        st.session_state.markers
    )

    original_width, original_height = marked_image.size

    display_width = min(
        original_width,
        MAX_DISPLAY_WIDTH
    )

    display_height = int(
        original_height * display_width / original_width
    )

    display_image = marked_image.resize(
        (display_width, display_height),
        Image.Resampling.LANCZOS
    )

    coords = streamlit_image_coordinates(
        display_image,
        key=f"image_click_{st.session_state.click_version}"
    )

    if coords is not None:
        handle_image_click(
            coords,
            display_width,
            display_height
        )

        st.session_state.click_version += 1
        st.rerun()

    st.divider()

    st.subheader("補正後の検出結果")

    final_image = draw_markers(
        st.session_state.work_image,
        st.session_state.markers
    )

    st.image(
        final_image,
        use_container_width=True
    )

    st.metric(
        "最終的なパイプ本数",
        len(st.session_state.markers)
    )

    if st.button(
        "検出結果をリセット",
        use_container_width=True
    ):
        reset_detection()
        st.rerun()


    st.divider()
    st.header("5. 検出結果の保存")

    group_name = st.text_input(
        "グループ名",
        placeholder="例：Aグループ"
    )

    comment = st.text_area(
        "コメント",
        placeholder="必要に応じて入力してください"
    )

    if st.button(
        "この検出結果を保存データに追加",
        use_container_width=True
    ):

        now = datetime.now()

        record_data = {
            "日時": now.strftime("%Y-%m-%d %H:%M:%S"),
            "グループ": group_name,
            "自動検出本数": st.session_state.automatic_count,
            "手動追加本数": st.session_state.manual_added,
            "手動削減本数": st.session_state.manual_deleted,
            "合計本数": len(st.session_state.markers),
            "コメント": comment
        }

        record_image = image_to_png_bytes(final_image)

        st.session_state.saved_records.append({
            "data": record_data,
            "image": record_image
        })

        st.success("保存データに追加しました。")


st.divider()
st.header("6. 保存データのダウンロード")

if st.session_state.saved_records:

    st.write(
        f"保存済みデータ：{len(st.session_state.saved_records)} 件"
    )

    zip_data = create_zip_bytes(
        st.session_state.saved_records
    )

    st.download_button(
        label="📦 CSV・PNGをZIPでダウンロード",
        data=zip_data,
        file_name="PINEAPPLE_results.zip",
        mime="application/zip",
        use_container_width=True
    )

    with st.expander("保存データの一覧"):

        for index, record in enumerate(
            st.session_state.saved_records,
            start=1
        ):

            st.write(
                f"{index}. "
                f"{record['data']['日時']} / "
                f"{record['data']['グループ']} / "
                f"{record['data']['合計本数']}本"
            )

else:
    st.info("まだ保存されたデータはありません。")
