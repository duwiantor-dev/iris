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
    Menghapus area putih / hitam ekstrem di pinggir gambar.
    """
    arr = np.array(img).astype(np.uint8)
    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3].copy()

    brightness = rgb.mean(axis=2)

    white_mask = brightness > 245
    black_mask = brightness < 18

    h, w = brightness.shape
    edge_mask = np.zeros((h, w), dtype=bool)
    margin_y = max(20, h // 10)
    margin_x = max(20, w // 10)

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


# =========================================================
# Scene building
# =========================================================
def create_background(size, preset="Midnight Blue"):
    w, h = size
    bg = Image.new("RGBA", (w, h), (3, 6, 20, 255))
    draw = ImageDraw.Draw(bg)

    if preset == "Midnight Blue":
        top = np.array([2, 4, 18], dtype=np.float32)
        bottom = np.array([3, 10, 62], dtype=np.float32)
    elif preset == "Warm Studio":
        top = np.array([18, 12, 10], dtype=np.float32)
        bottom = np.array([92, 68, 40], dtype=np.float32)
    else:
        top = np.array([10, 10, 16], dtype=np.float32)
        bottom = np.array([40, 40, 52], dtype=np.float32)

    for y in range(h):
        t = y / max(1, h - 1)
        rgb = tuple((top * (1 - t) + bottom * t).astype(np.uint8).tolist())
        draw.line([(0, y), (w, y)], fill=rgb + (255,))

    # vignette
    vignette = Image.new("L", (w, h), 0)
    vg = ImageDraw.Draw(vignette)
    for i in range(11):
        pad = i * 38
        alpha = int(8 + i * 7)
        vg.rounded_rectangle(
            [pad, pad, w - pad, h - pad],
            radius=100,
            outline=alpha,
            width=36,
        )
    vignette = vignette.filter(ImageFilter.GaussianBlur(130))
    sh = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    sh.putalpha(vignette)
    bg = Image.alpha_composite(bg, sh)

    return bg


def create_light_beam(size):
    w, h = size
    beam = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(beam)

    d.polygon(
        [
            (int(w * 0.58), int(h * 0.26)),
            (int(w * 0.73), int(h * 0.22)),
            (int(w * 0.86), int(h * 0.72)),
            (int(w * 0.68), int(h * 0.82)),
        ],
        fill=(255, 242, 220, 34),
    )

    return beam.filter(ImageFilter.GaussianBlur(80))


def create_table_surface(size):
    w, h = size
    surface = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(surface)

    # meja utama, sudut lebih mirip referensi
    desk_poly = [
        (w * 0.14, h * 0.58),
        (w * 0.82, h * 0.48),
        (w * 1.02, h * 0.94),
        (w * -0.02, h * 1.01),
    ]
    draw.polygon(desk_poly, fill=(227, 205, 180, 247))

    # highlight meja
    highlight = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    hd = ImageDraw.Draw(highlight)
    hd.polygon(
        [
            (w * 0.55, h * 0.54),
            (w * 0.78, h * 0.50),
            (w * 0.88, h * 0.70),
            (w * 0.66, h * 0.74),
        ],
        fill=(255, 242, 225, 28),
    )
    highlight = highlight.filter(ImageFilter.GaussianBlur(40))
    surface = Image.alpha_composite(surface, highlight)

    # garis mat kiri bawah
    curve_color = (148, 104, 84, 118)
    for i in range(7):
        bbox = [
            int(-w * 0.14 + i * 33),
            int(h * 0.56 + i * 14),
            int(w * 0.26 + i * 33),
            int(h * 0.98 + i * 14),
        ]
        draw.arc(bbox, start=188, end=303, fill=curve_color, width=3)

    return surface


def create_contact_shadow(obj: Image.Image, opacity=130):
    alpha = obj.getchannel("A")
    shadow = Image.new("RGBA", obj.size, (0, 0, 0, 0))
    shadow.putalpha(alpha)

    shadow = shadow.resize(
        (max(1, int(obj.size[0] * 1.04)), max(1, int(obj.size[1] * 0.18))),
        Image.Resampling.LANCZOS,
    )

    arr = np.array(shadow)
    arr[:, :, 0] = 0
    arr[:, :, 1] = 0
    arr[:, :, 2] = 0
    arr[:, :, 3] = (arr[:, :, 3].astype(np.float32) * (opacity / 255.0)).astype(np.uint8)

    out = Image.fromarray(arr, mode="RGBA")
    return out.filter(ImageFilter.GaussianBlur(16))


def create_soft_shadow(obj: Image.Image, opacity=92):
    alpha = obj.getchannel("A")
    shadow = Image.new("RGBA", obj.size, (0, 0, 0, 0))
    shadow.putalpha(alpha)

    shadow = shadow.resize(
        (max(1, int(obj.size[0] * 1.13)), max(1, int(obj.size[1] * 0.28))),
        Image.Resampling.LANCZOS,
    )

    arr = np.array(shadow)
    arr[:, :, 0] = 0
    arr[:, :, 1] = 0
    arr[:, :, 2] = 0
    arr[:, :, 3] = (arr[:, :, 3].astype(np.float32) * (opacity / 255.0)).astype(np.uint8)

    out = Image.fromarray(arr, mode="RGBA")
    return out.filter(ImageFilter.GaussianBlur(30))


def add_brand_text(canvas, headline, subheadline, product_name, model_name):
    draw = ImageDraw.Draw(canvas)
    w, h = canvas.size

    title_font = try_font(88, bold=True)
    sub_font = try_font(29, bold=False)
    product_font = try_font(62, bold=True)
    model_font = try_font(24, bold=False)

    title = headline.upper()
    bbox = draw.textbbox((0, 0), title, font=title_font)
    title_w = bbox[2] - bbox[0]
    draw.text(
        ((w - title_w) / 2, int(h * 0.11)),
        title,
        font=title_font,
        fill=(255, 255, 255, 245),
    )

    sb = draw.textbbox((0, 0), subheadline, font=sub_font)
    sub_w = sb[2] - sb[0]
    draw.text(
        ((w - sub_w) / 2, int(h * 0.17)),
        subheadline,
        font=sub_font,
        fill=(255, 255, 255, 230),
    )

    px = int(w * 0.49)
    py = int(h * 0.79)

    draw.text(
        (px, py),
        product_name.upper(),
        font=product_font,
        fill=(255, 255, 255, 245),
    )
    draw.text(
        (px, py + 64),
        model_name,
        font=model_font,
        fill=(255, 255, 255, 232),
    )

    arrow = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    ad = ImageDraw.Draw(arrow)
    ad.arc(
        [px - 72, py - 46, px + 72, py + 82],
        start=248,
        end=28,
        fill=(255, 255, 255, 235),
        width=4,
    )
    ad.line([(px + 56, py + 32), (px + 73, py + 48)], fill=(255, 255, 255, 235), width=4)
    ad.line([(px + 56, py + 32), (px + 54, py + 53)], fill=(255, 255, 255, 235), width=4)
    canvas.alpha_composite(arrow)

    return canvas


# =========================================================
# Core image generation
# =========================================================
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
    canvas = create_background(canvas_size, bg_preset)
    canvas = Image.alpha_composite(canvas, create_table_surface(canvas_size))
    canvas = Image.alpha_composite(canvas, create_light_beam(canvas_size))

    laptop = safe_remove_background(laptop_img)

    bbox = laptop.getbbox()
    if bbox:
        laptop = laptop.crop(bbox)

    # perbesar object agar lebih mendekati referensi
    laptop = fit_contain(laptop, 980, 980)

    rgb = laptop.convert("RGB")
    rgb = ImageEnhance.Contrast(rgb).enhance(1.0 + 0.18 * enhance_level)
    rgb = ImageEnhance.Sharpness(rgb).enhance(1.0 + 0.62 * enhance_level)
    rgb = ImageEnhance.Color(rgb).enhance(1.0 + 0.06 * enhance_level)
    rgb = ImageEnhance.Brightness(rgb).enhance(1.0 + 0.02 * enhance_level)
    laptop = rgb.convert("RGBA")

    # angle lebih halus
    laptop = laptop.rotate(-9, expand=True, resample=Image.Resampling.BICUBIC)

    w, h = canvas.size
    lw, lh = laptop.size

    # posisi utama
    x = int((w - lw) * 0.34)
    y = int((h - lh) * 0.40)

    soft_shadow = create_soft_shadow(laptop, opacity=88)
    ssw, ssh = soft_shadow.size
    canvas.alpha_composite(soft_shadow, (x + 20, y + lh - int(ssh * 0.10)))

    contact_shadow = create_contact_shadow(laptop, opacity=128)
    csw, csh = contact_shadow.size
    canvas.alpha_composite(contact_shadow, (x + 12, y + lh - int(csh * 0.05)))

    # sedikit glow di sekitar object
    glow = Image.new("RGBA", laptop.size, (255, 244, 228, 0))
    glow.putalpha(laptop.getchannel("A"))
    glow = glow.filter(ImageFilter.GaussianBlur(18))
    glow_arr = np.array(glow)
    glow_arr[:, :, 3] = (glow_arr[:, :, 3].astype(np.float32) * 0.10).astype(np.uint8)
    glow = Image.fromarray(glow_arr, mode="RGBA")
    canvas.alpha_composite(glow, (x, y))

    canvas.alpha_composite(laptop, (x, y))
    canvas = add_brand_text(canvas, headline, subheadline, product_name, model_name)

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

        zoom = 1.0 + (0.085 * t)
        crop_w = int(w / zoom)
        crop_h = int(h / zoom)

        cx = int(w * 0.52)
        cy = int(h * (0.55 - 0.018 * t))

        left = max(0, min(w - crop_w, cx - crop_w // 2))
        top = max(0, min(h - crop_h, cy - crop_h // 2))

        frame = base.crop((left, top, left + crop_w, top + crop_h)).resize(
            (w, h), Image.Resampling.LANCZOS
        )
        frame = ImageEnhance.Brightness(frame).enhance(1.0 + 0.028 * math.sin(t * math.pi))
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

preview_ratio_cols = [0.10, 0.80, 0.10]

if uploaded:
    user_img = load_image(uploaded)

    top_left, top_mid, top_right = st.columns([1.55, 1.0, 1.0])

    with top_left:
        st.subheader("Foto Asli")
        st.image(user_img, use_container_width=True)

    if st.button("Generate Redwood", type="primary"):
        with st.spinner("Redwood sedang membuat image dan video..."):
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

                video_path = generate_video_from_image(
                    promo,
                    duration_sec=duration_sec,
                    fps=24,
                )
                with open(video_path, "rb") as f:
                    video_bytes = f.read()
                st.session_state["video_bytes"] = video_bytes

                st.rerun()
            except Exception as e:
                st.error(f"Gagal generate: {e}")

    with top_mid:
        st.subheader("Hasil Redwood")
        mid_left, mid_center, mid_right = st.columns(preview_ratio_cols)
        with mid_center:
            if "promo_img" in st.session_state:
                st.image(st.session_state["promo_img"], use_container_width=True)

    with top_right:
        st.subheader("Preview Video")
        vid_left, vid_center, vid_right = st.columns(preview_ratio_cols)
        with vid_center:
            if "video_bytes" in st.session_state:
                st.video(st.session_state["video_bytes"])

    if "promo_img" in st.session_state:
        st.markdown("### Download Hasil")
        d1, d2 = st.columns(2)

        promo = st.session_state["promo_img"]
        jpg_bytes = pil_to_bytes(promo, fmt="JPG")

        with d1:
            st.download_button(
                "Download JPG",
                data=jpg_bytes,
                file_name="redwood_promo_laptop.jpg",
                mime="image/jpeg",
                use_container_width=True,
            )

        with d2:
            if "video_bytes" in st.session_state:
                st.download_button(
                    "Download MP4",
                    data=st.session_state["video_bytes"],
                    file_name="redwood_promo_laptop_video.mp4",
                    mime="video/mp4",
                    use_container_width=True,
                )

st.markdown("---")
st.markdown(
    """
### Next upgrade untuk Redwood
- background removal yang lebih pintar
- preset layout yang lebih banyak
- text animation dan transition preset
- history hasil generate
"""
)
