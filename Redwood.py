import os
import io
import math
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance


st.set_page_config(page_title="Redwood", page_icon="🌲", layout="wide")

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# =========================================================
# Helpers
# =========================================================
def load_image(uploaded_file) -> Image.Image:
    return Image.open(uploaded_file).convert("RGBA")


def fit_contain(img: Image.Image, max_w: int, max_h: int) -> Image.Image:
    copy = img.copy()
    copy.thumbnail((max_w, max_h))
    return copy


def safe_remove_background(img: Image.Image) -> Image.Image:
    """
    Background remover ringan untuk MVP.
    Tidak seakurat AI, tapi aman untuk Streamlit Cloud.
    Akan menghapus area yang sangat putih atau sangat hitam di pinggir.
    """
    arr = np.array(img).astype(np.uint8)
    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3].copy()

    brightness = rgb.mean(axis=2)

    # Remove near-white background
    white_mask = brightness > 245

    # Remove near-black background
    black_mask = brightness < 18

    # Only aggressively remove edges so main object lebih aman
    h, w = brightness.shape
    edge_mask = np.zeros((h, w), dtype=bool)
    margin_y = max(20, h // 12)
    margin_x = max(20, w // 12)

    edge_mask[:margin_y, :] = True
    edge_mask[-margin_y:, :] = True
    edge_mask[:, :margin_x] = True
    edge_mask[:, -margin_x:] = True

    final_mask = (white_mask | black_mask) & edge_mask
    alpha[final_mask] = 0

    arr[:, :, 3] = alpha
    out = Image.fromarray(arr, mode="RGBA")

    bbox = out.getbbox()
    if bbox:
        out = out.crop(bbox)

    return out


def create_gradient_background(size, preset="Midnight Blue"):
    w, h = size
    bg = Image.new("RGBA", (w, h), (5, 8, 24, 255))
    draw = ImageDraw.Draw(bg)

    if preset == "Midnight Blue":
        top = np.array([4, 7, 22], dtype=np.float32)
        bottom = np.array([2, 18, 72], dtype=np.float32)
    elif preset == "Warm Studio":
        top = np.array([20, 14, 12], dtype=np.float32)
        bottom = np.array([110, 80, 48], dtype=np.float32)
    else:
        top = np.array([12, 12, 18], dtype=np.float32)
        bottom = np.array([48, 48, 60], dtype=np.float32)

    for y in range(h):
        t = y / max(1, h - 1)
        rgb = tuple((top * (1 - t) + bottom * t).astype(np.uint8).tolist())
        draw.line([(0, y), (w, y)], fill=rgb + (255,))

    # soft vignette
    vignette = Image.new("L", (w, h), 0)
    vg = ImageDraw.Draw(vignette)
    for i in range(10):
        pad = i * 42
        alpha = int(12 + i * 8)
        vg.rounded_rectangle(
            [pad, pad, w - pad, h - pad],
            radius=90,
            outline=alpha,
            width=42,
        )
    vignette = vignette.filter(ImageFilter.GaussianBlur(100))

    shadow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    shadow.putalpha(vignette)
    return Image.alpha_composite(bg, shadow)


def create_table_surface(size):
    w, h = size
    surface = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(surface)

    # main desk plane
    draw.polygon(
        [
            (w * 0.12, h * 0.58),
            (w * 0.82, h * 0.49),
            (w * 1.03, h * 0.95),
            (w * -0.05, h * 1.02),
        ],
        fill=(224, 203, 181, 245),
    )

    # decorative arcs like deskmat
    curve_color = (150, 105, 85, 110)
    for i in range(7):
        bbox = [
            int(-w * 0.22 + i * 38),
            int(h * 0.50 + i * 16),
            int(w * 0.40 + i * 38),
            int(h * 1.02 + i * 16),
        ]
        draw.arc(bbox, start=200, end=310, fill=curve_color, width=3)

    return surface.filter(ImageFilter.GaussianBlur(0.4))


def create_light_beam(size):
    w, h = size
    beam = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(beam)

    d.polygon(
        [
            (int(w * 0.58), int(h * 0.23)),
            (int(w * 0.72), int(h * 0.20)),
            (int(w * 0.84), int(h * 0.72)),
            (int(w * 0.68), int(h * 0.78)),
        ],
        fill=(255, 244, 225, 34),
    )
    return beam.filter(ImageFilter.GaussianBlur(65))


def create_floor_shadow(obj: Image.Image, opacity=105):
    alpha = obj.getchannel("A")
    shadow = Image.new("RGBA", obj.size, (0, 0, 0, 0))
    shadow.putalpha(alpha)

    shadow = shadow.resize(
        (max(1, int(obj.size[0] * 1.08)), max(1, int(obj.size[1] * 0.24))),
        Image.Resampling.LANCZOS,
    )

    arr = np.array(shadow)
    arr[:, :, 0] = 0
    arr[:, :, 1] = 0
    arr[:, :, 2] = 0
    arr[:, :, 3] = (arr[:, :, 3].astype(np.float32) * (opacity / 255.0)).astype(np.uint8)

    shadow = Image.fromarray(arr, mode="RGBA")
    return shadow.filter(ImageFilter.GaussianBlur(22))


def try_font(size, bold=False):
    candidates = []
    if bold:
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
        ]
    else:
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        ]

    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def add_text_overlay(canvas, headline, subheadline, product_name, model_name):
    draw = ImageDraw.Draw(canvas)
    w, h = canvas.size

    title_font = try_font(82, bold=True)
    sub_font = try_font(28, bold=False)
    product_font = try_font(58, bold=True)
    model_font = try_font(24, bold=False)

    title = headline.upper()
    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    title_w = title_bbox[2] - title_bbox[0]
    draw.text(
        ((w - title_w) / 2, int(h * 0.12)),
        title,
        font=title_font,
        fill=(255, 255, 255, 245),
    )

    sub_bbox = draw.textbbox((0, 0), subheadline, font=sub_font)
    sub_w = sub_bbox[2] - sub_bbox[0]
    draw.text(
        ((w - sub_w) / 2, int(h * 0.18)),
        subheadline,
        font=sub_font,
        fill=(255, 255, 255, 230),
    )

    px = int(w * 0.50)
    py = int(h * 0.79)

    draw.text(
        (px, py),
        product_name.upper(),
        font=product_font,
        fill=(255, 255, 255, 245),
    )
    draw.text(
        (px, py + 60),
        model_name,
        font=model_font,
        fill=(255, 255, 255, 230),
    )

    arrow = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    ad = ImageDraw.Draw(arrow)
    ad.arc(
        [px - 60, py - 45, px + 82, py + 78],
        start=250,
        end=25,
        fill=(255, 255, 255, 235),
        width=4,
    )
    ad.line([(px + 62, py + 28), (px + 76, py + 44)], fill=(255, 255, 255, 235), width=4)
    ad.line([(px + 62, py + 28), (px + 58, py + 49)], fill=(255, 255, 255, 235), width=4)
    canvas.alpha_composite(arrow)

    return canvas


def compose_promo_image(
    laptop_img: Image.Image,
    headline: str,
    subheadline: str,
    product_name: str,
    model_name: str,
    bg_preset: str,
    enhance_level: float,
):
    canvas_size = (1080, 1920)
    canvas = create_gradient_background(canvas_size, bg_preset)
    canvas = Image.alpha_composite(canvas, create_table_surface(canvas_size))

    laptop = safe_remove_background(laptop_img)
    laptop = fit_contain(laptop, 760, 760)

    rgb = laptop.convert("RGB")
    rgb = ImageEnhance.Contrast(rgb).enhance(1.0 + 0.20 * enhance_level)
    rgb = ImageEnhance.Sharpness(rgb).enhance(1.0 + 0.65 * enhance_level)
    rgb = ImageEnhance.Color(rgb).enhance(1.0 + 0.08 * enhance_level)
    laptop = rgb.convert("RGBA")

    laptop = laptop.rotate(-12, expand=True, resample=Image.Resampling.BICUBIC)

    w, h = canvas.size
    lw, lh = laptop.size
    x = int((w - lw) * 0.44)
    y = int((h - lh) * 0.46)

    floor_shadow = create_floor_shadow(laptop, opacity=110)
    fsw, fsh = floor_shadow.size
    canvas.alpha_composite(floor_shadow, (x + 8, y + lh - int(fsh * 0.18)))

    canvas.alpha_composite(laptop, (x, y))
    canvas = Image.alpha_composite(canvas, create_light_beam(canvas_size))
    canvas = add_text_overlay(canvas, headline, subheadline, product_name, model_name)

    return canvas


def pil_to_bytes(img: Image.Image, fmt="PNG"):
    bio = io.BytesIO()
    if fmt.upper() in ("JPG", "JPEG"):
        img.convert("RGB").save(bio, format="JPEG", quality=95)
    else:
        img.save(bio, format=fmt)
    bio.seek(0)
    return bio.getvalue()


def generate_video_from_image(img: Image.Image, duration_sec=5, fps=24):
    base = img.convert("RGB")
    w, h = base.size
    total_frames = max(1, int(duration_sec * fps))

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        video_path = tmp.name

    writer = imageio.get_writer(video_path, fps=fps, codec="libx264", quality=8)

    for i in range(total_frames):
        t = i / max(1, total_frames - 1)

        zoom = 1.0 + (0.12 * t)
        crop_w = int(w / zoom)
        crop_h = int(h / zoom)

        cx = w // 2
        cy = int(h * (0.52 - 0.03 * t))

        left = max(0, min(w - crop_w, cx - crop_w // 2))
        top = max(0, min(h - crop_h, cy - crop_h // 2))

        frame = base.crop((left, top, left + crop_w, top + crop_h)).resize(
            (w, h), Image.Resampling.LANCZOS
        )
        frame = ImageEnhance.Brightness(frame).enhance(
            1.0 + 0.04 * math.sin(t * math.pi)
        )

        writer.append_data(np.array(frame))

    writer.close()
    return video_path


# =========================================================
# UI
# =========================================================
st.title("🌲 Redwood")
st.caption("Upload foto laptop → generate visual promo cinematic → export image dan video pendek.")

with st.sidebar:
    st.header("Redwood Settings")
    headline = st.text_input("Headline", value="Worth It")
    subheadline = st.text_input("Subheadline", value="di kelasnya?")
    product_name = st.text_input("Product Name", value="Lenovo LOQ")
    model_name = st.text_input("Model Name", value="15IAX9 1JID")
    bg_preset = st.selectbox(
        "Background Preset",
        ["Midnight Blue", "Warm Studio", "Slate Dark"],
    )
    enhance_level = st.slider(
        "Enhance Level",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
    )
    duration_sec = st.slider(
        "Video Duration (sec)",
        min_value=3,
        max_value=10,
        value=5,
    )

uploaded = st.file_uploader(
    "Upload foto laptop",
    type=["png", "jpg", "jpeg", "webp"],
)

col1, col2 = st.columns(2)

if uploaded:
    user_img = load_image(uploaded)

    with col1:
        st.subheader("Foto Asli")
        st.image(user_img, use_container_width=True)

    if st.button("Generate with Redwood", type="primary"):
        with st.spinner("Redwood sedang membuat promo image..."):
            try:
                promo = compose_promo_image(
                    laptop_img=user_img,
                    headline=headline,
                    subheadline=subheadline,
                    product_name=product_name,
                    model_name=model_name,
                    bg_preset=bg_preset,
                    enhance_level=enhance_level,
                )
                st.session_state["promo_img"] = promo
            except Exception as e:
                st.error(f"Gagal generate image: {e}")

if "promo_img" in st.session_state:
    promo = st.session_state["promo_img"]

    with col2:
        st.subheader("Hasil Redwood")
        st.image(promo, width=320)

    png_bytes = pil_to_bytes(promo, fmt="PNG")
    jpg_bytes = pil_to_bytes(promo, fmt="JPG")

    d1, d2 = st.columns(2)
    d1.download_button(
        "Download PNG",
        data=png_bytes,
        file_name="redwood_promo_laptop.png",
        mime="image/png",
    )
    d2.download_button(
        "Download JPG",
        data=jpg_bytes,
        file_name="redwood_promo_laptop.jpg",
        mime="image/jpeg",
    )

    if st.button("Generate Redwood Video"):
        with st.spinner("Redwood sedang render video..."):
            try:
                video_path = generate_video_from_image(
                    promo,
                    duration_sec=duration_sec,
                    fps=24,
                )
                with open(video_path, "rb") as f:
                    video_bytes = f.read()
                st.session_state["video_bytes"] = video_bytes
            except Exception as e:
                st.error(f"Gagal generate video: {e}")

if "video_bytes" in st.session_state:
    st.subheader("Preview Video")

    preview_col1, preview_col2, preview_col3 = st.columns([1, 2, 1])
    with preview_col2:
        st.video(st.session_state["video_bytes"])
    st.download_button(
        "Download MP4",
        data=st.session_state["video_bytes"],
        file_name="redwood_promo_laptop_video.mp4",
        mime="video/mp4",
    )

st.markdown("---")
st.markdown(
    """
### Next upgrade untuk Redwood
- tambah preset layout lain
- tambahkan auto background removal yang lebih pintar
- tambahkan text animation dan transition preset
- simpan history hasil generate
"""
)
