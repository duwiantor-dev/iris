import os
import io
import math
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import streamlit as st
import imageio.v2 as imageio


st.set_page_config(page_title="Redwood", page_icon="🌲", layout="wide")

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def load_image(uploaded_file) -> Image.Image:
    return Image.open(uploaded_file).convert("RGBA")


def fit_contain(img: Image.Image, max_w: int, max_h: int) -> Image.Image:
    copy = img.copy()
    copy.thumbnail((max_w, max_h))
    return copy


def simple_background_remove(img: Image.Image) -> Image.Image:
    """
    Background removal sederhana untuk MVP.
    Nanti bisa diganti dengan rembg atau API image editing.
    """
    arr = np.array(img).astype(np.uint8)
    rgb = arr[:, :, :3]
    brightness = rgb.mean(axis=2)

    alpha = arr[:, :, 3].copy()
    mask = brightness > 245
    alpha[mask] = 0
    arr[:, :, 3] = alpha
    return Image.fromarray(arr, mode="RGBA")


def create_gradient_background(size, preset="Midnight Blue"):
    w, h = size
    bg = Image.new("RGBA", (w, h), (8, 10, 20, 255))
    draw = ImageDraw.Draw(bg)

    if preset == "Midnight Blue":
        top = np.array([5, 8, 20], dtype=np.float32)
        bottom = np.array([8, 20, 70], dtype=np.float32)
    elif preset == "Warm Studio":
        top = np.array([18, 14, 12], dtype=np.float32)
        bottom = np.array([120, 88, 52], dtype=np.float32)
    else:
        top = np.array([10, 10, 14], dtype=np.float32)
        bottom = np.array([50, 50, 58], dtype=np.float32)

    for y in range(h):
        t = y / max(1, h - 1)
        rgb = tuple((top * (1 - t) + bottom * t).astype(np.uint8).tolist())
        draw.line([(0, y), (w, y)], fill=rgb + (255,))

    vignette = Image.new("L", (w, h), 0)
    vg = ImageDraw.Draw(vignette)
    for i in range(8):
        pad = i * 40
        alpha = int(18 + i * 10)
        vg.rounded_rectangle(
            [pad, pad, w - pad, h - pad],
            radius=80,
            outline=alpha,
            width=40,
        )
    vignette = vignette.filter(ImageFilter.GaussianBlur(80))

    shadow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    shadow.putalpha(vignette)
    bg = Image.alpha_composite(bg, shadow)
    return bg


def create_table_surface(size):
    w, h = size
    surface = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(surface)

    draw.polygon(
        [
            (w * 0.15, h * 0.58),
            (w * 0.85, h * 0.48),
            (w * 1.05, h * 0.95),
            (w * -0.05, h * 1.0),
        ],
        fill=(230, 205, 180, 240),
    )

    curve_color = (180, 120, 90, 110)
    for i in range(7):
        bbox = [
            int(-w * 0.2 + i * 35),
            int(h * 0.52 + i * 18),
            int(w * 0.45 + i * 35),
            int(h * 1.05 + i * 18),
        ]
        draw.arc(bbox, start=200, end=310, fill=curve_color, width=3)

    return surface.filter(ImageFilter.GaussianBlur(0.5))


def make_shadow(size, bbox, blur=35, opacity=120):
    w, h = size
    shadow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(shadow)
    d.rounded_rectangle(bbox, radius=40, fill=(0, 0, 0, opacity))
    return shadow.filter(ImageFilter.GaussianBlur(blur))


def try_font(size, bold=False):
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

    title_font = try_font(78, bold=True)
    small_font = try_font(28, bold=False)
    product_font = try_font(56, bold=True)
    model_font = try_font(24, bold=False)

    title = headline.upper()
    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    title_w = title_bbox[2] - title_bbox[0]

    draw.text(
        ((w - title_w) / 2, h * 0.12),
        title,
        font=title_font,
        fill=(255, 255, 255, 245),
    )

    sub_bbox = draw.textbbox((0, 0), subheadline, font=small_font)
    sub_w = sub_bbox[2] - sub_bbox[0]
    draw.text(
        ((w - sub_w) / 2, h * 0.18),
        subheadline,
        font=small_font,
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
        (px, py + 58),
        model_name,
        font=model_font,
        fill=(255, 255, 255, 230),
    )

    arrow = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    ad = ImageDraw.Draw(arrow)
    ad.arc(
        [px - 60, py - 40, px + 80, py + 80],
        start=250,
        end=25,
        fill=(255, 255, 255, 240),
        width=4,
    )
    ad.line([(px + 62, py + 30), (px + 76, py + 45)], fill=(255, 255, 255, 240), width=4)
    ad.line([(px + 62, py + 30), (px + 58, py + 50)], fill=(255, 255, 255, 240), width=4)
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
    desk = create_table_surface(canvas_size)
    canvas = Image.alpha_composite(canvas, desk)

    laptop = simple_background_remove(laptop_img)
    laptop = fit_contain(laptop, 760, 760)

    rgb = laptop.convert("RGB")
    rgb = ImageEnhance.Contrast(rgb).enhance(1.0 + 0.2 * enhance_level)
    rgb = ImageEnhance.Sharpness(rgb).enhance(1.0 + 0.6 * enhance_level)
    laptop = rgb.convert("RGBA")

    laptop = laptop.rotate(-12, expand=True, resample=Image.Resampling.BICUBIC)

    w, h = canvas.size
    lw, lh = laptop.size
    x = int((w - lw) * 0.48)
    y = int((h - lh) * 0.49)

    shadow1 = make_shadow(
        canvas_size,
        [x + 40, y + 110, x + lw - 50, y + lh - 30],
        blur=42,
        opacity=110,
    )
    shadow2 = make_shadow(
        canvas_size,
        [x - 20, y + 210, x + lw - 180, y + lh + 10],
        blur=55,
        opacity=70,
    )
    canvas = Image.alpha_composite(canvas, shadow1)
    canvas = Image.alpha_composite(canvas, shadow2)

    canvas.alpha_composite(laptop, (x, y))

    highlight = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    hd = ImageDraw.Draw(highlight)
    hd.polygon(
        [
            (w * 0.55, h * 0.25),
            (w * 0.75, h * 0.18),
            (w * 0.95, h * 0.72),
            (w * 0.72, h * 0.78),
        ],
        fill=(255, 240, 220, 30),
    )
    highlight = highlight.filter(ImageFilter.GaussianBlur(45))
    canvas = Image.alpha_composite(canvas, highlight)

    canvas = add_text_overlay(canvas, headline, subheadline, product_name, model_name)
    return canvas


def pil_to_bytes(img: Image.Image, format="PNG"):
    bio = io.BytesIO()
    if format.upper() in ("JPG", "JPEG"):
        img.convert("RGB").save(bio, format="JPEG", quality=95)
    else:
        img.save(bio, format=format)
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
            (w, h),
            Image.Resampling.LANCZOS,
        )

        frame = ImageEnhance.Brightness(frame).enhance(
            1.0 + 0.04 * math.sin(t * math.pi)
        )

        writer.append_data(np.array(frame))

    writer.close()
    return video_path


st.title("🌲 ")
st.caption("Upload foto laptop → ubah jadi visual promo cinematic → export image dan video pendek.")

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

uploaded = st.file_uploader("Upload foto laptop", type=["png", "jpg", "jpeg", "webp"])

col1, col2 = st.columns(2)

if uploaded:
    user_img = load_image(uploaded)

    with col1:
        st.subheader("Foto Asli")
        st.image(user_img, use_container_width=True)

    if st.button("Generate with Redwood", type="primary"):
        with st.spinner("Redwood is generating your promo image..."):
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

if "promo_img" in st.session_state:
    promo = st.session_state["promo_img"]

    with col2:
        st.subheader("Hasil Redwood")
        st.image(promo, use_container_width=True)

    png_bytes = pil_to_bytes(promo, format="PNG")
    jpg_bytes = pil_to_bytes(promo, format="JPG")

    d1, d2 = st.columns(2)
    d1.download_button(
        "Download PNG",
        data=png_bytes,
        file_name="Redwood_promo_laptop.png",
        mime="image/png",
    )
    d2.download_button(
        "Download JPG",
        data=jpg_bytes,
        file_name="Redwood_promo_laptop.jpg",
        mime="image/jpeg",
    )

    if st.button("Generate Redwood Video"):
        with st.spinner("Redwood is rendering your video..."):
            video_path = generate_video_from_image(
                promo,
                duration_sec=duration_sec,
                fps=24,
            )
            with open(video_path, "rb") as f:
                video_bytes = f.read()
            st.session_state["video_bytes"] = video_bytes

if "video_bytes" in st.session_state:
    st.subheader("Preview Video")
    st.video(st.session_state["video_bytes"])
    st.download_button(
        "Download MP4",
        data=st.session_state["video_bytes"],
        file_name="Redwood_promo_laptop_video.mp4",
        mime="video/mp4",
    )

st.markdown("---")
st.markdown(
    """
### Next upgrade untuk Redwood
- ganti background removal sederhana dengan `rembg`
- sambungkan ke image editing API
- tambahkan text animation dan transition preset
- simpan history hasil generate
"""
)
