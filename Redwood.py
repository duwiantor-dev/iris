import io
import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Growth Dashboard Agres", page_icon="📈", layout="wide")

# =========================
# CSS (Light + Badge + Clip + SKU column widths + Smaller headers)
# =========================
st.markdown(
    """
<style>
.block-container { padding-top: 0.6rem; padding-bottom: 1.2rem; }
section[data-testid="stSidebar"] .block-container { padding-top: 0.6rem; }

/* Header (large -> small) */
.header-wrap { display:flex; align-items:center; gap:12px; margin: 0.2rem 0 0.8rem 0; }
.header-title { font-size: 22px; font-weight: 900; margin:0; line-height:1.1; color: #111827; }

/* Make section headers smaller */
h2 { font-size: 18px !important; }
h3 { font-size: 16px !important; }

/* Cards */
.kpi-grid { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 12px; }

.card {
  border: 1px solid rgba(17,24,39,0.08);
  background: #ffffff;
  border-radius: 14px;
  padding: 12px 12px 10px 12px;
  box-shadow: 0 10px 20px rgba(17,24,39,0.04);
}
.card-title { font-size: 12px; color: rgba(17,24,39,0.7); margin-bottom: 6px; }
.card-value { font-size: 18px; font-weight: 900; line-height: 1.15; color: #111827; }
.card-sub { font-size: 11px; margin-top: 6px; font-weight: 800; }

.pos { color: #16a34a; }
.neg { color: #dc2626; }
.na  { color: #64748b; }

hr { border: none; border-top: 1px solid rgba(17,24,39,0.10); margin: 14px 0; }

/* Small headings (for long titles) */
.small-h { font-size: 16px; font-weight: 900; margin: 0 0 6px 0; }
.small-h .muted { color: rgba(17,24,39,0.6); font-weight: 800; }

/* HTML table styling + CLIP */
table {
  width: 100%;
  border-collapse: separate;
  border-spacing: 0;
  overflow: hidden;
  border-radius: 12px;
  border: 1px solid rgba(17,24,39,0.08);
  background: #fff;
  table-layout: fixed;
}
thead th {
  background: rgba(17,24,39,0.03);
  font-size: 12px;
  color: rgba(17,24,39,0.8);
  padding: 10px 10px;
  border-bottom: 1px solid rgba(17,24,39,0.08);
  text-align: left;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
tbody td {
  padding: 9px 10px;
  font-size: 12px;
  border-bottom: 1px solid rgba(17,24,39,0.06);
  color: #111827;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
tbody tr:hover td { background: rgba(225,29,46,0.04); }

/* Growth badges: merah/hijau */
.badge-pos, .badge-neg, .badge-na {
  display:inline-block;
  font-size: 11px;
  font-weight: 900;
  padding: 2px 8px;
  border-radius: 999px;
  white-space: nowrap;
}
.badge-pos { background: rgba(22,163,74,0.12); color:#16a34a; }
.badge-neg { background: rgba(220,38,38,0.12); color:#dc2626; }
.badge-na  { background: rgba(100,116,139,0.12); color:#64748b; }

/* SKU table column widths (SPESIFIKASI wider, Growth wider so header & % visible) */
table.sku-table th:nth-child(1), table.sku-table td:nth-child(1) { width: 50%; }
table.sku-table th:nth-child(2), table.sku-table td:nth-child(2) { width: 14%; }
table.sku-table th:nth-child(3), table.sku-table td:nth-child(3) { width: 14%; }
table.sku-table th:nth-child(4), table.sku-table td:nth-child(4) { width: 10%; } /* Delta */
table.sku-table th:nth-child(5), table.sku-table td:nth-child(5) { width: 12%; } /* Growth */
</style>
""",
    unsafe_allow_html=True,
)

DEFAULT_DATE_FORMAT_HINT = "Format TGL: dd-mm-yyyy / dd/mm/yyyy / yyyy-mm-dd"


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().upper() for c in df.columns]
    return df


def coerce_numeric_series(s: pd.Series) -> pd.Series:
    def to_num(x):
        if pd.isna(x):
            return np.nan
        if isinstance(x, (int, float, np.integer, np.floating)):
            return float(x)
        txt = str(x).strip()
        txt = re.sub(r"[^0-9\-\.,]", "", txt)
        if txt in ("", "-", ".", ","):
            return np.nan
        txt = txt.replace(" ", "").replace(".", "").replace(",", "")
        try:
            return float(txt)
        except Exception:
            return np.nan

    return s.map(to_num)


def parse_tgl(df: pd.DataFrame, col: str = "TGL") -> pd.Series:
    s = df[col]
    if np.issubdtype(s.dtype, np.datetime64):
        return pd.to_datetime(s, errors="coerce").dt.date
    parsed = pd.to_datetime(s, errors="coerce", dayfirst=True)
    return parsed.dt.date


def ensure_required_columns(df: pd.DataFrame) -> Tuple[bool, str]:
    required = [
        "STATUS", "TGL", "TRANSAKSI", "TEAM",
        "PRODUCT", "BRAND", "QTY", "JUMLAH",
        "SO NO",
        "COUNTRY",
        "SPESIFIKASI",
        "NAMA CUSTOMER",
        "OTO",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return False, f"Kolom wajib tidak ditemukan: {', '.join(missing)}"
    return True, ""


def drop_total_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "NO" in out.columns:
        out = out[~out["NO"].astype(str).str.strip().str.upper().eq("TOTAL")].copy()
    for c in ["STATUS", "TGL"]:
        if c in out.columns:
            out = out[~out[c].astype(str).str.strip().str.upper().eq("TOTAL")].copy()
    return out.dropna(how="all")


def format_idr(x: float) -> str:
    if pd.isna(x):
        return "-"
    n = int(round(float(x)))
    s = f"{n:,}".replace(",", ".")
    return f"IDR {s}"


def format_int_id(x: float) -> str:
    if pd.isna(x):
        return "-"
    return f"{int(round(float(x))):,}".replace(",", ".")


def compact_number(x: float) -> str:
    if pd.isna(x):
        return ""
    x = float(x)
    ax = abs(x)
    if ax >= 1_000_000_000:
        return f"{x/1_000_000_000:.2f}B".replace(".", ",")
    if ax >= 1_000_000:
        return f"{x/1_000_000:.2f}M".replace(".", ",")
    if ax >= 1_000:
        return f"{x/1_000:.2f}K".replace(".", ",")
    return str(int(round(x)))


def safe_growth_pct(this_val: float, last_val: float) -> Optional[float]:
    if last_val is None or pd.isna(last_val):
        return None
    last_val = float(last_val)
    if last_val == 0.0:
        return None
    return (float(this_val) - last_val) / last_val * 100.0


def growth_label(g: Optional[float]) -> str:
    if g is None or pd.isna(g):
        return "N/A"
    s = f"{g:,.2f}%".replace(",", "X").replace(".", ",").replace("X", ".")
    return ("▲ " + s) if g >= 0 else ("▼ " + s)


def growth_badge_html(g: Optional[float]) -> str:
    if g is None or pd.isna(g):
        return '<span class="badge-na">N/A</span>'
    s = f"{g:,.2f}%".replace(",", "X").replace(".", ",").replace("X", ".")
    if g >= 0:
        return f'<span class="badge-pos">▲ {s}</span>'
    return f'<span class="badge-neg">▼ {s}</span>'


def kpi_delta_class(g: Optional[float]) -> str:
    if g is None or pd.isna(g):
        return "na"
    return "pos" if g >= 0 else "neg"


@dataclass
class CleanData:
    df: pd.DataFrame
    date_min: date
    date_max: date


@st.cache_data(show_spinner=False)
def read_excel_cached(file_bytes: bytes, sheet_name: str, header_row_1based: int) -> pd.DataFrame:
    bio = io.BytesIO(file_bytes)
    header_idx = int(header_row_1based) - 1
    if sheet_name.strip() == "":
        return pd.read_excel(bio, header=header_idx)
    return pd.read_excel(bio, sheet_name=sheet_name.strip(), header=header_idx)


@st.cache_data(show_spinner=False)
def clean_sales_df_cached(df_raw: pd.DataFrame) -> CleanData:
    df = normalize_columns(df_raw)
    ok, msg = ensure_required_columns(df)
    if not ok:
        raise ValueError(msg)

    df = drop_total_rows(df)

    keep_cols = [
        "STATUS", "TGL", "TRANSAKSI", "TEAM",
        "PRODUCT", "BRAND", "QTY", "JUMLAH",
        "SO NO",
        "COUNTRY",
        "SPESIFIKASI",
        "NAMA CUSTOMER",
        "OTO",
        "AREA",
    ]
    df = df[keep_cols].copy()

    df["TGL"] = parse_tgl(df, "TGL")
    if df["TGL"].isna().any():
        bad = df[df["TGL"].isna()].head(8)
        raise ValueError(
            f"Ada TGL gagal diparse.\n\nContoh:\n{bad[['TGL']].to_string(index=False)}\n\n{DEFAULT_DATE_FORMAT_HINT}"
        )

    df["QTY"] = pd.to_numeric(coerce_numeric_series(df["QTY"]), errors="coerce").fillna(0.0)
    df["JUMLAH"] = pd.to_numeric(coerce_numeric_series(df["JUMLAH"]), errors="coerce").fillna(0.0)

    for c in [
        "STATUS", "TRANSAKSI", "TEAM", "PRODUCT", "BRAND",
        "SO NO", "COUNTRY", "SPESIFIKASI", "NAMA CUSTOMER", "OTO"
    ]:
        df[c] = df[c].astype(str).str.strip()

    df["ROW_TYPE"] = np.where(df["STATUS"].str.upper().str.contains("RETUR"), "RETUR", "SO_OUT")
    df["OTO_YES"] = df["OTO"].str.upper().eq("YES")
    df["PLATFORM"] = df["NAMA CUSTOMER"].astype(str).str.strip()

    df = df[df["STATUS"].str.strip().ne("")].copy()
    return CleanData(df=df, date_min=df["TGL"].min(), date_max=df["TGL"].max())


@st.cache_data(show_spinner=False)
def compute_kpis_cached(df: pd.DataFrame) -> Dict[str, float]:
    sales = float(df["JUMLAH"].sum())
    qty = float(df["QTY"].sum())

    so = df[df["ROW_TYPE"] == "SO_OUT"]
    orders = float(so["SO NO"].nunique())

    returns = float(len(df[df["ROW_TYPE"] == "RETUR"]))
    aov = sales / orders if orders else np.nan
    return {"sales": sales, "qty": qty, "orders": orders, "returns": returns, "aov": float(aov) if not pd.isna(aov) else np.nan}


def get_week_start(d: date) -> date:
    return d - timedelta(days=d.weekday())


def month_start(d: date) -> date:
    return date(d.year, d.month, 1)


def prev_month_same_day(d: date) -> date:
    first = month_start(d)
    prev_last = first - timedelta(days=1)
    day = min(d.day, prev_last.day)
    return date(prev_last.year, prev_last.month, day)


def slice_period(df: pd.DataFrame, start: date, end_inclusive: date) -> pd.DataFrame:
    return df[(df["TGL"] >= start) & (df["TGL"] <= end_inclusive)].copy()


def build_period_frames(df_all: pd.DataFrame, mode: str, df_upload_a: pd.DataFrame, df_upload_b: pd.DataFrame):
    if mode == "UPLOAD":
        return df_upload_a, df_upload_b, "Periode A", "Periode B"

    anchor = df_all["TGL"].max()

    if mode == "WOW":
        this_start = get_week_start(anchor)
        this_end = anchor
        last_end = this_start - timedelta(days=1)
        last_start = get_week_start(last_end)
        return (
            slice_period(df_all, last_start, last_end),
            slice_period(df_all, this_start, this_end),
            "Week Lalu",
            "Week Ini",
        )

    if mode == "MOM":
        this_start = month_start(anchor)
        this_end = anchor
        last_anchor = prev_month_same_day(anchor)
        last_start = month_start(last_anchor)
        last_month_mask = (df_all["TGL"] >= last_start) & (df_all["TGL"] < this_start)
        last_end = df_all.loc[last_month_mask, "TGL"].max() if last_month_mask.any() else last_start
        return (
            slice_period(df_all, last_start, last_end),
            slice_period(df_all, this_start, this_end),
            "Bulan Lalu",
            "Bulan Ini",
        )

    if mode == "MTD":
        this_start = month_start(anchor)
        this_end = anchor
        last_anchor = prev_month_same_day(anchor)
        last_start = month_start(last_anchor)

        next_month_first = month_start(anchor)
        last_month_last_day = next_month_first - timedelta(days=1)
        last_end_candidate = date(last_month_last_day.year, last_month_last_day.month, min(anchor.day, last_month_last_day.day))

        last_month_mask = (df_all["TGL"] >= last_start) & (df_all["TGL"] < this_start)
        if last_month_mask.any():
            last_end_data = df_all.loc[last_month_mask, "TGL"].max()
            last_end = min(last_end_candidate, last_end_data)
        else:
            last_end = last_end_candidate

        return (
            slice_period(df_all, last_start, last_end),
            slice_period(df_all, this_start, this_end),
            "MTD Bulan Lalu",
            "MTD Bulan Ini",
        )

    return df_upload_a, df_upload_b, "Periode Lalu", "Periode Ini"


def options_for(df_all: pd.DataFrame, col: str) -> List[str]:
    return sorted([v for v in df_all[col].dropna().unique().tolist() if str(v).strip() != ""])


def apply_multifilter(df: pd.DataFrame, col: str, selected: List[str]) -> pd.DataFrame:
    if not selected:
        return df
    return df[df[col].isin(selected)].copy()


@st.cache_data(show_spinner=False)
def top_table_cached(df_this: pd.DataFrame, df_last: pd.DataFrame, by_col: str, metric: str, top_n: int) -> pd.DataFrame:
    agg_this = df_this.groupby(by_col, as_index=False).agg(THIS=(metric, "sum"))
    agg_last = df_last.groupby(by_col, as_index=False).agg(LAST=(metric, "sum"))
    merged = agg_this.merge(agg_last, on=by_col, how="outer").fillna(0.0)
    merged["DELTA"] = merged["THIS"] - merged["LAST"]
    merged["GROWTH_NUM"] = merged.apply(lambda r: safe_growth_pct(r["THIS"], r["LAST"]), axis=1)
    merged = merged.sort_values("THIS", ascending=False).head(top_n)

    if metric == "JUMLAH":
        merged["Periode Ini"] = merged["THIS"].map(format_idr)
        merged["Periode Lalu"] = merged["LAST"].map(format_idr)
        merged["Delta"] = merged["DELTA"].map(format_idr)
    else:
        merged["Periode Ini"] = merged["THIS"].map(format_int_id)
        merged["Periode Lalu"] = merged["LAST"].map(format_int_id)
        merged["Delta"] = merged["DELTA"].map(format_int_id)

    merged["Growth"] = merged["GROWTH_NUM"].apply(growth_badge_html)
    merged = merged[[by_col, "Periode Ini", "Periode Lalu", "Delta", "Growth"]]
    return merged


def render_html_table(df: pd.DataFrame, table_class: str = ""):
    html = df.to_html(escape=False, index=False)
    if table_class:
        html = html.replace("<table ", f'<table class="{table_class}" ', 1)
    st.markdown(html, unsafe_allow_html=True)


def small_title(text: str, hint: str = ""):
    hint_html = f' <span class="muted">{hint}</span>' if hint else ""
    st.markdown(f'<div class="small-h">{text}{hint_html}</div>', unsafe_allow_html=True)


def render_header():
    st.markdown(
        """
<div class="header-wrap">
  <div>
    <div class="header-title">Growth Dashboard Agres</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def style_growth_pct_df(df_in: pd.DataFrame):
    df = df_in.copy()

    def color_growth(val):
        try:
            if pd.isna(val):
                return "color: #64748b;"
            return "color: #dc2626; font-weight: 800;" if float(val) < 0 else "color: #16a34a; font-weight: 800;"
        except Exception:
            return "color: #64748b;"

    return (
        df.style
        .format({"Growth %": "{:.2f}%"})
        .applymap(color_growth, subset=["Growth %"])
    )


# ===== NEW: Team Down % Table =====
@st.cache_data(show_spinner=False)
def team_down_ratio_table_cached(df_last: pd.DataFrame, df_this: pd.DataFrame) -> pd.DataFrame:
    last = df_last.groupby("TEAM", as_index=False).agg(QTY_LALU=("QTY", "sum"))
    this = df_this.groupby("TEAM", as_index=False).agg(QTY_INI=("QTY", "sum"))
    t = last.merge(this, on="TEAM", how="outer").fillna(0.0)
    t["DELTA"] = t["QTY_INI"] - t["QTY_LALU"]

    # total team aktif: punya activity di salah satu periode
    active = t[(t["QTY_LALU"] > 0) | (t["QTY_INI"] > 0)].copy()
    total = int(len(active))
    turun = int((active["DELTA"] < 0).sum())
    naik = int((active["DELTA"] > 0).sum())
    flat = int((active["DELTA"] == 0).sum())
    pct_turun = (turun / total * 100.0) if total else 0.0

    out = pd.DataFrame(
        {
            "Total TEAM aktif": [total],
            "TEAM Turun": [turun],
            "TEAM Naik": [naik],
            "TEAM Tetap": [flat],
            "% TEAM Turun": [pct_turun],
        }
    )
    return out


def _drivers_as_text(df_delta: pd.DataFrame, team_dir: str, top_k: int = 3) -> Dict[str, str]:
    """
    df_delta columns: TEAM, DIM, DELTA
    team_dir: mapping TEAM -> +1 (naik) or -1 (turun)
    """
    # Join direction for filtering
    d = df_delta.copy()
    d["DIR"] = d["TEAM"].map(team_dir).fillna(0).astype(int)

    # For naik: keep DELTA > 0, take top_k biggest
    # For turun: keep DELTA < 0, take top_k most negative
    naik_df = d[(d["DIR"] > 0) & (d["DELTA"] > 0)].copy()
    turun_df = d[(d["DIR"] < 0) & (d["DELTA"] < 0)].copy()

    naik_df = naik_df.sort_values(["TEAM", "DELTA"], ascending=[True, False]).groupby("TEAM").head(top_k)
    turun_df = turun_df.sort_values(["TEAM", "DELTA"], ascending=[True, True]).groupby("TEAM").head(top_k)

    # Build strings
    out: Dict[str, str] = {}

    def fmt_row(dim: str, delta: float) -> str:
        sign = "+" if delta > 0 else ""
        return f"{dim} ({sign}{int(delta):,})".replace(",", ".")

    for team, g in naik_df.groupby("TEAM"):
        out[team] = ", ".join([fmt_row(r["DIM"], r["DELTA"]) for _, r in g.iterrows()])

    for team, g in turun_df.groupby("TEAM"):
        out[team] = ", ".join([fmt_row(r["DIM"], r["DELTA"]) for _, r in g.iterrows()])

    return out


@st.cache_data(show_spinner=False)
def team_driver_analysis_table_cached(df_last: pd.DataFrame, df_this: pd.DataFrame, top_k: int = 3) -> pd.DataFrame:
    # team totals
    last_t = df_last.groupby("TEAM", as_index=False).agg(QTY_LALU=("QTY", "sum"))
    this_t = df_this.groupby("TEAM", as_index=False).agg(QTY_INI=("QTY", "sum"))
    team = last_t.merge(this_t, on="TEAM", how="outer").fillna(0.0)
    team = team[(team["QTY_LALU"] > 0) | (team["QTY_INI"] > 0)].copy()
    team["DELTA_QTY"] = team["QTY_INI"] - team["QTY_LALU"]
    team["GROWTH_PCT"] = team.apply(lambda r: safe_growth_pct(r["QTY_INI"], r["QTY_LALU"]), axis=1)

    # direction mapping (+1 naik, -1 turun, 0 flat)
    team_dir = {r["TEAM"]: (1 if r["DELTA_QTY"] > 0 else (-1 if r["DELTA_QTY"] < 0 else 0)) for _, r in team.iterrows()}

    # build delta per TEAM x DIM for PRODUCT / BRAND / SKU(SPESIFIKASI)
    def build_delta(dim_col: str) -> pd.DataFrame:
        a = df_this.groupby(["TEAM", dim_col], as_index=False).agg(THIS=("QTY", "sum"))
        b = df_last.groupby(["TEAM", dim_col], as_index=False).agg(LAST=("QTY", "sum"))
        m = a.merge(b, on=["TEAM", dim_col], how="outer").fillna(0.0)
        m["DELTA"] = m["THIS"] - m["LAST"]
        m = m.rename(columns={dim_col: "DIM"})
        return m[["TEAM", "DIM", "DELTA"]]

    prod_delta = build_delta("PRODUCT")
    brand_delta = build_delta("BRAND")
    sku_delta = build_delta("SPESIFIKASI")

    prod_map = _drivers_as_text(prod_delta, team_dir, top_k=top_k)
    brand_map = _drivers_as_text(brand_delta, team_dir, top_k=top_k)
    sku_map = _drivers_as_text(sku_delta, team_dir, top_k=top_k)

    team["Arah"] = team["DELTA_QTY"].apply(lambda x: "NAIK" if x > 0 else ("TURUN" if x < 0 else "TETAP"))
    team["Produk (driver)"] = team["TEAM"].map(prod_map).fillna("-")
    team["Brand (driver)"] = team["TEAM"].map(brand_map).fillna("-")
    team["SKU/Spesifikasi (driver)"] = team["TEAM"].map(sku_map).fillna("-")

    # pretty
    out = team.copy()
    out["QTY Lalu"] = out["QTY_LALU"].map(format_int_id)
    out["QTY Ini"] = out["QTY_INI"].map(format_int_id)
    out["Delta"] = out["DELTA_QTY"].map(lambda x: f"{int(x):,}".replace(",", "."))
    out["Growth %"] = out["GROWTH_PCT"].apply(lambda x: float(x) if (x is not None and not pd.isna(x)) else np.nan)

    out = out.sort_values(["DELTA_QTY"], ascending=True)  # yang turun paling parah di atas (biar langsung keliatan)
    out = out[
        [
            "TEAM", "Arah", "QTY Lalu", "QTY Ini", "Delta", "Growth %",
            "Produk (driver)", "Brand (driver)", "SKU/Spesifikasi (driver)"
        ]
    ]
    return out


# =========================
# UI
# =========================
render_header()

with st.sidebar:
    st.subheader("Upload Data")
    file_a = st.file_uploader("Excel A (.xlsx) — periode lama", type=["xlsx"], key="a")
    file_b = st.file_uploader("Excel B (.xlsx) — periode baru", type=["xlsx"], key="b")

    st.markdown("---")
    st.subheader("Header & Sheet")
    header_row_a = st.number_input("Header row Excel A (mulai dari 1)", 1, 30, 2, 1)
    header_row_b = st.number_input("Header row Excel B (mulai dari 1)", 1, 30, 2, 1)
    sheet_a = st.text_input("Nama sheet Excel A (kosongkan = sheet pertama)", "")
    sheet_b = st.text_input("Nama sheet Excel B (kosongkan = sheet pertama)", "")

if not file_a or not file_b:
    st.info("Upload 2 file Excel dulu.")
    st.stop()

with st.spinner("Membaca & membersihkan Excel (sekali di awal)..."):
    df_a_raw = read_excel_cached(file_a.getvalue(), sheet_a, header_row_a)
    df_b_raw = read_excel_cached(file_b.getvalue(), sheet_b, header_row_b)
    a = clean_sales_df_cached(df_a_raw)
    b = clean_sales_df_cached(df_b_raw)

df_a = a.df
df_b = b.df
df_all = pd.concat([df_a, df_b], ignore_index=True)

with st.sidebar:
    st.markdown("---")
    st.subheader("Mode Perbandingan")
    compare_mode = st.selectbox("Pilih periode", ["MOM", "WOW", "MTD", "UPLOAD"], 0)

    st.markdown("---")
    st.subheader("Opsi Tampilan")
    top_n = st.slider("Top N", 5, 30, 10, 1)
    metric_choice = st.selectbox("Metric", ["Qty (QTY)", "Sales (JUMLAH)"], 0)
    show_point_labels = st.toggle("Tampilkan angka di titik grafik", value=False)

    st.markdown("---")
    st.subheader("Filter (multi pilih)")
    with st.form("filter_form", clear_on_submit=False):
        category_sel = st.multiselect("CATEGORY (COUNTRY)", options_for(df_all, "COUNTRY"), default=[])
        transaksi_sel = st.multiselect("TRANSAKSI", options_for(df_all, "TRANSAKSI"), default=[])
        team_sel = st.multiselect("TEAM", options_for(df_all, "TEAM"), default=[])
        product_sel = st.multiselect("PRODUCT", options_for(df_all, "PRODUCT"), default=[])
        brand_sel = st.multiselect("BRAND", options_for(df_all, "BRAND"), default=[])
        platform_sel = st.multiselect("PLATFORM (NAMA CUSTOMER)", options_for(df_all, "PLATFORM"), default=[])
        apply_clicked = st.form_submit_button("✅ Apply Filter")

if "filters" not in st.session_state:
    st.session_state["filters"] = {"COUNTRY": [], "TRANSAKSI": [], "TEAM": [], "PRODUCT": [], "BRAND": [], "PLATFORM": []}
if apply_clicked:
    st.session_state["filters"] = {
        "COUNTRY": category_sel,
        "TRANSAKSI": transaksi_sel,
        "TEAM": team_sel,
        "PRODUCT": product_sel,
        "BRAND": brand_sel,
        "PLATFORM": platform_sel,
    }

flt = st.session_state["filters"]


def apply_all_filters(df: pd.DataFrame) -> pd.DataFrame:
    out = df
    out = apply_multifilter(out, "COUNTRY", flt["COUNTRY"])
    out = apply_multifilter(out, "TRANSAKSI", flt["TRANSAKSI"])
    out = apply_multifilter(out, "TEAM", flt["TEAM"])
    out = apply_multifilter(out, "PRODUCT", flt["PRODUCT"])
    out = apply_multifilter(out, "BRAND", flt["BRAND"])
    out = apply_multifilter(out, "PLATFORM", flt["PLATFORM"])
    return out


df_all_f = apply_all_filters(df_all)
df_a_f = apply_all_filters(df_a)
df_b_f = apply_all_filters(df_b)

df_last, df_this, label_last, label_this = build_period_frames(df_all_f, compare_mode, df_a_f, df_b_f)

metric_col = "QTY" if metric_choice.startswith("Qty") else "JUMLAH"
metric_name = "Qty" if metric_col == "QTY" else "Sales (IDR)"

# =========================
# KPI
# =========================
k_last = compute_kpis_cached(df_last)
k_this = compute_kpis_cached(df_this)

sales_g = safe_growth_pct(k_this["sales"], k_last["sales"])
orders_g = safe_growth_pct(k_this["orders"], k_last["orders"])
qty_g = safe_growth_pct(k_this["qty"], k_last["qty"])
aov_g = safe_growth_pct(k_this["aov"], k_last["aov"]) if (not pd.isna(k_this["aov"]) and not pd.isna(k_last["aov"])) else None

st.subheader("Ringkasan Periode")
c1, c2, c3, c4 = st.columns(4)


def summary_card(title: str, value: str):
    st.markdown(
        f"""
<div class="card">
  <div class="card-title">{title}</div>
  <div class="card-value" style="font-size:15px">{value}</div>
</div>
""",
        unsafe_allow_html=True,
    )


with c1:
    summary_card(label_last, f"{df_last['TGL'].min()} → {df_last['TGL'].max()}" if len(df_last) else "-")
with c2:
    summary_card(label_this, f"{df_this['TGL'].min()} → {df_this['TGL'].max()}" if len(df_this) else "-")
with c3:
    summary_card("Rows Periode Lalu", f"{len(df_last):,}".replace(",", "."))
with c4:
    summary_card("Rows Periode Ini", f"{len(df_this):,}".replace(",", "."))

st.markdown("<hr/>", unsafe_allow_html=True)

st.subheader("KPI Utama")
kpi_html = f"""
<div class="kpi-grid">
  <div class="card">
    <div class="card-title">Total Sales (Periode Ini)</div>
    <div class="card-value">{format_idr(k_this["sales"])}</div>
    <div class="card-sub {kpi_delta_class(sales_g)}">{growth_label(sales_g)}</div>
  </div>
  <div class="card">
    <div class="card-title">Orders</div>
    <div class="card-value">{format_int_id(k_this["orders"])}</div>
    <div class="card-sub {kpi_delta_class(orders_g)}">{growth_label(orders_g)}</div>
  </div>
  <div class="card">
    <div class="card-title">Total Qty</div>
    <div class="card-value">{format_int_id(k_this["qty"])}</div>
    <div class="card-sub {kpi_delta_class(qty_g)}">{growth_label(qty_g)}</div>
  </div>
  <div class="card">
    <div class="card-title">AOV</div>
    <div class="card-value">{format_idr(k_this["aov"])}</div>
    <div class="card-sub {kpi_delta_class(aov_g)}">{growth_label(aov_g)}</div>
  </div>
  <div class="card">
    <div class="card-title">Retur (lines)</div>
    <div class="card-value">{format_int_id(k_this["returns"])}</div>
  </div>
</div>
""".replace(",", ".")
st.markdown(kpi_html, unsafe_allow_html=True)

# =========================
# NEW TABLE #1: % TEAM TURUN
# =========================
st.markdown("<hr/>", unsafe_allow_html=True)
st.subheader("📉 Proporsi TEAM yang Turun (berdasarkan QTY)")
down_tbl = team_down_ratio_table_cached(df_last, df_this)
st.dataframe(
    down_tbl,
    use_container_width=True,
    hide_index=True,
    column_config={"% TEAM Turun": st.column_config.NumberColumn(format="%.2f")},
)

st.markdown("<hr/>", unsafe_allow_html=True)

COLOR_MAP_PERIOD = {
    "Bulan Ini": "#1f77b4",
    "Bulan Lalu": "#aec7e8",
}

# =========================
# Trend chart (Day-of-Month comparison)
# =========================
st.subheader(f"Tren Harian ({metric_name}) — Day of Month Comparison")


def day_of_month_series(df: pd.DataFrame, label: str) -> pd.DataFrame:
    tmp = df.copy()
    tmp["DAY"] = pd.to_datetime(tmp["TGL"]).dt.day
    g = tmp.groupby("DAY", as_index=False).agg(VALUE=(metric_col, "sum"))
    g["PERIODE"] = label
    return g


all_days = pd.DataFrame({"DAY": list(range(1, 32))})
trend_dom = pd.concat(
    [
        all_days.merge(day_of_month_series(df_last, label_last), on="DAY", how="left").assign(PERIODE=label_last),
        all_days.merge(day_of_month_series(df_this, label_this), on="DAY", how="left").assign(PERIODE=label_this),
    ],
    ignore_index=True,
)
trend_dom["VALUE"] = trend_dom["VALUE"].fillna(0.0)

if show_point_labels:
    trend_dom["LABEL"] = trend_dom["VALUE"].apply(compact_number)
    fig = px.line(
        trend_dom,
        x="DAY",
        y="VALUE",
        color="PERIODE",
        markers=True,
        text="LABEL",
        color_discrete_map=COLOR_MAP_PERIOD,
    )
    fig.update_traces(textposition="top center")
else:
    fig = px.line(
        trend_dom,
        x="DAY",
        y="VALUE",
        color="PERIODE",
        markers=True,
        color_discrete_map=COLOR_MAP_PERIOD,
    )

fig.update_layout(
    xaxis_title="Tanggal (Day of Month)",
    yaxis_title=metric_name,
    legend_title_text="",
    xaxis=dict(dtick=1),
)
st.plotly_chart(fig, use_container_width=True)

# =========================
# Cumulative chart
# =========================
st.markdown("<hr/>", unsafe_allow_html=True)
st.subheader(f"Statistik Kumulatif ({metric_name}) — Day of Month Comparison")

trend_cum = trend_dom.copy()
trend_cum = trend_cum.sort_values(["PERIODE", "DAY"]).copy()
trend_cum["CUM_VALUE"] = trend_cum.groupby("PERIODE")["VALUE"].cumsum()

# super clean compare hover: tampilkan Bulan Ini + Bulan Lalu + Delta dalam satu hover
trend_compare = (
    trend_cum.pivot(index="DAY", columns="PERIODE", values="CUM_VALUE")
    .reset_index()
    .rename_axis(None, axis=1)
)

if "Bulan Ini" not in trend_compare.columns:
    trend_compare["Bulan Ini"] = 0
if "Bulan Lalu" not in trend_compare.columns:
    trend_compare["Bulan Lalu"] = 0

trend_compare["Bulan Ini"] = trend_compare["Bulan Ini"].fillna(0)
trend_compare["Bulan Lalu"] = trend_compare["Bulan Lalu"].fillna(0)
trend_compare["DELTA"] = trend_compare["Bulan Ini"] - trend_compare["Bulan Lalu"]

trend_cum = trend_cum.merge(
    trend_compare[["DAY", "Bulan Ini", "Bulan Lalu", "DELTA"]],
    on="DAY",
    how="left"
)

trend_cum["BULAN_INI_TXT"] = trend_cum["Bulan Ini"].apply(compact_number)
trend_cum["BULAN_LALU_TXT"] = trend_cum["Bulan Lalu"].apply(compact_number)
trend_cum["DELTA_TXT"] = trend_cum["DELTA"].apply(
    lambda x: f"+{compact_number(x)}" if x > 0 else compact_number(x)
)

fig_cum = px.line(
    trend_cum,
    x="DAY",
    y="CUM_VALUE",
    color="PERIODE",
    markers=True,
    color_discrete_map=COLOR_MAP_PERIOD,
    custom_data=["BULAN_INI_TXT", "BULAN_LALU_TXT", "DELTA_TXT"],
)

fig_cum.update_traces(
    hovertemplate=(
        "<b>Hari %{x}</b><br>"
        "Bulan Ini: %{customdata[0]}<br>"
        "Bulan Lalu: %{customdata[1]}<br>"
        "Delta: %{customdata[2]}<extra>%{fullData.name}</extra>"
    )
)

fig_cum.update_layout(
    xaxis_title="Tanggal (Day of Month)",
    yaxis_title=f"Kumulatif {metric_name}",
    legend_title_text="",
    xaxis=dict(dtick=1),
)
st.plotly_chart(fig_cum, use_container_width=True)


# =========================
# Pareto + Delta + Comparison
# =========================
st.markdown("<hr/>", unsafe_allow_html=True)
st.subheader("Pareto + Delta + Comparison (This Vs Last Month)")

pareto_dim = st.selectbox(
    "Dimensi Pareto",
    ["PLATFORM", "TEAM", "PRODUCT", "BRAND", "TRANSAKSI", "AREA"],
    index=0,
)

pareto_top_n = st.slider("Top N Pareto", min_value=5, max_value=30, value=10, step=1)

def build_pareto_comparison(df_this: pd.DataFrame, df_last: pd.DataFrame, dim_col: str, value_col: str, top_n: int = 10):
    this_agg = (
        df_this.groupby(dim_col, as_index=False)
        .agg(THIS_VALUE=(value_col, "sum"))
    )
    last_agg = (
        df_last.groupby(dim_col, as_index=False)
        .agg(LAST_VALUE=(value_col, "sum"))
    )

    comp = this_agg.merge(last_agg, on=dim_col, how="outer").fillna(0.0)
    comp[dim_col] = comp[dim_col].astype(str).str.strip()
    comp = comp[comp[dim_col].ne("")].copy()
    comp = comp.sort_values("THIS_VALUE", ascending=False).head(top_n).copy()

    if comp.empty:
        return comp

    total_this = comp["THIS_VALUE"].sum()
    total_last = comp["LAST_VALUE"].sum()

    comp["THIS_SHARE"] = np.where(total_this != 0, comp["THIS_VALUE"] / total_this * 100.0, 0.0)
    comp["LAST_SHARE"] = np.where(total_last != 0, comp["LAST_VALUE"] / total_last * 100.0, 0.0)
    comp["PARETO_THIS"] = comp["THIS_SHARE"].cumsum()
    comp["PARETO_LAST"] = comp["LAST_SHARE"].cumsum()
    comp["DELTA_SHARE"] = comp["THIS_SHARE"] - comp["LAST_SHARE"]
    comp["DELTA_LABEL"] = comp["DELTA_SHARE"].map(lambda x: f"{x:+.1f}%")

    comp["BAR_HOVER"] = comp.apply(
        lambda r: (
            f"<b>{r[dim_col]}</b><br>"
            f"{label_this}: {compact_number(r['THIS_VALUE'])}<br>"
            f"{label_last}: {compact_number(r['LAST_VALUE'])}<br>"
            f"Kontribusi {label_this}: {r['THIS_SHARE']:.2f}%<br>"
            f"Kontribusi {label_last}: {r['LAST_SHARE']:.2f}%<br>"
            f"Delta kontribusi: {r['DELTA_SHARE']:+.2f}%"
        ),
        axis=1,
    )

    return comp

pareto_df = build_pareto_comparison(df_this, df_last, pareto_dim, metric_col, pareto_top_n)

if pareto_df.empty:
    st.info("Belum ada data untuk grafik Pareto pada filter saat ini.")
else:
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    fig_pareto = make_subplots(specs=[[{"secondary_y": True}]])

    fig_pareto.add_trace(
        go.Bar(
            x=pareto_df[pareto_dim],
            y=pareto_df["THIS_VALUE"],
            name=label_this,
            marker_color="#8ecae6",
            text=pareto_df["DELTA_LABEL"],
            textposition="outside",
            hovertext=pareto_df["BAR_HOVER"],
            hovertemplate="%{hovertext}<extra></extra>",
        ),
        secondary_y=False,
    )

    fig_pareto.add_trace(
        go.Scatter(
            x=pareto_df[pareto_dim],
            y=pareto_df["PARETO_THIS"],
            name=f"Pareto {label_this}",
            mode="lines+markers",
            line=dict(color="#1f77b4", width=3),
            marker=dict(symbol="circle", size=7, color="#1f77b4"),
            hovertemplate=f"<b>%{{x}}</b><br>Pareto {label_this}: %{{y:.2f}}%<extra></extra>",
        ),
        secondary_y=True,
    )

    fig_pareto.add_trace(
        go.Scatter(
            x=pareto_df[pareto_dim],
            y=pareto_df["PARETO_LAST"],
            name=f"Pareto {label_last}",
            mode="lines+markers",
            line=dict(color="#f59e0b", width=2.5, dash="dash"),
            marker=dict(symbol="x", size=8, color="#f59e0b"),
            hovertemplate=f"<b>%{{x}}</b><br>Pareto {label_last}: %{{y:.2f}}%<extra></extra>",
        ),
        secondary_y=True,
    )

    fig_pareto.add_hline(
        y=80,
        line_width=1.5,
        line_dash="solid",
        line_color="#3b82f6",
        opacity=0.85,
        secondary_y=True,
    )

    fig_pareto.update_layout(
        title="Pareto + Delta + Comparison (This Vs Last Month)",
        hovermode="x unified",
        legend_title_text="",
        xaxis_title=pareto_dim.title(),
        yaxis_title=f"{metric_name} ({label_this})",
        margin=dict(l=40, r=40, t=70, b=40),
    )

    fig_pareto.update_yaxes(
        title_text=f"{metric_name} ({label_this})",
        secondary_y=False,
        showgrid=True,
        gridcolor="rgba(0,0,0,0.08)",
    )
    fig_pareto.update_yaxes(
        title_text="Cumulative (%)",
        range=[0, 105],
        ticksuffix="%",
        secondary_y=True,
        showgrid=False,
    )

    st.plotly_chart(fig_pareto, use_container_width=True)


st.markdown("<hr/>", unsafe_allow_html=True)

# =========================
# NEW TABLE #2: TEAM DRIVER ANALYSIS
# =========================
st.markdown("<hr/>", unsafe_allow_html=True)
st.subheader("🧠 Analisis Penyebab Perubahan")
st.caption("Untuk TEAM yang TURUN: ditampilkan top driver yang paling narik turun. Untuk TEAM yang NAIK: top driver yang paling narik naik.")

topk = st.slider("Top driver per kategori", 1, 10, 3, 1)
analysis_df = team_driver_analysis_table_cached(df_last, df_this, top_k=topk)

st.dataframe(
    style_growth_pct_df(analysis_df),
    use_container_width=True,
    height=520,
)

# =========================
# Top tables
# =========================
c1, c2 = st.columns(2)
with c1:
    small_title(f"Top {top_n} TEAM", f"(by {metric_col})")
    render_html_table(top_table_cached(df_this, df_last, "TEAM", metric_col, top_n))
with c2:
    small_title(f"Top {top_n} PRODUCT", f"(by {metric_col})")
    render_html_table(top_table_cached(df_this, df_last, "PRODUCT", metric_col, top_n))

c3, c4 = st.columns(2)
with c3:
    small_title(f"Top {top_n} BRAND", f"(by {metric_col})")
    render_html_table(top_table_cached(df_this, df_last, "BRAND", metric_col, top_n))
with c4:
    small_title(f"Top {top_n} TRANSAKSI", f"(by {metric_col})")
    render_html_table(top_table_cached(df_this, df_last, "TRANSAKSI", metric_col, top_n))

c5, c6 = st.columns(2)
with c5:
    small_title(f"Top {top_n} SKU", "(SPESIFIKASI)")
    render_html_table(top_table_cached(df_this, df_last, "SPESIFIKASI", metric_col, top_n), table_class="sku-table")
with c6:
    small_title(f"Top {top_n} PLATFORM", "(sumber: NAMA CUSTOMER)")
    render_html_table(top_table_cached(df_this, df_last, "PLATFORM", metric_col, top_n))

st.markdown("<hr/>", unsafe_allow_html=True)

# =========================
# TEAM PERFORMANCE (3 columns)
# =========================
st.subheader("📊 Team Performance Insight (QTY)")

team_last = df_last.groupby("TEAM", as_index=False).agg(QTY_LALU=("QTY", "sum"))
team_this = df_this.groupby("TEAM", as_index=False).agg(QTY_INI=("QTY", "sum"))
team = team_last.merge(team_this, on="TEAM", how="outer").fillna(0.0)
team["DELTA_QTY"] = team["QTY_INI"] - team["QTY_LALU"]
team["GROWTH_PCT"] = team.apply(lambda r: safe_growth_pct(r["QTY_INI"], r["QTY_LALU"]), axis=1)

under = team[team["QTY_INI"] < 30].copy().sort_values(["QTY_INI", "GROWTH_PCT"], ascending=[True, True])

oto_team = (
    df_this.groupby("TEAM", as_index=False)
    .agg(
        OTO_YES_LINES=("OTO_YES", "sum"),
        TOTAL_LINES=("OTO_YES", "count"),
        QTY_INI=("QTY", "sum"),
    )
)
oto_team["OTO_RATE"] = np.where(oto_team["TOTAL_LINES"] > 0, oto_team["OTO_YES_LINES"] / oto_team["TOTAL_LINES"] * 100.0, 0.0)
oto_team = oto_team.sort_values(["OTO_YES_LINES", "OTO_RATE"], ascending=[False, False])

top_all = team.copy().sort_values(["GROWTH_PCT", "QTY_INI"], ascending=[False, False])


def prep_team_view(df_in: pd.DataFrame) -> pd.DataFrame:
    d = df_in.copy()
    d["QTY Lalu"] = d["QTY_LALU"].map(format_int_id)
    d["QTY Ini"] = d["QTY_INI"].map(format_int_id)
    d["Delta"] = d["DELTA_QTY"].map(format_int_id)
    d["Growth %"] = d["GROWTH_PCT"].apply(lambda x: float(x) if (x is not None and not pd.isna(x)) else np.nan)
    return d[["TEAM", "QTY Lalu", "QTY Ini", "Delta", "Growth %"]]


def prep_oto_view(df_in: pd.DataFrame) -> pd.DataFrame:
    d = df_in.copy()
    d["OTO YES (lines)"] = d["OTO_YES_LINES"].astype(int)
    d["OTO Rate %"] = d["OTO_RATE"]
    d["QTY (Periode Ini)"] = d["QTY_INI"].map(format_int_id)
    return d[["TEAM", "OTO YES (lines)", "OTO Rate %", "QTY (Periode Ini)"]]


col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🏆 Top Performer (All TEAM)")
    st.caption("Scroll & klik header kolom untuk sort (Growth% / QTY).")
    df_top = prep_team_view(top_all).copy()
    st.dataframe(style_growth_pct_df(df_top), use_container_width=True, height=420)

with col2:
    st.markdown("### ⚠️ Under Perform (QTY < 30)")
    st.caption("Team dengan QTY periode ini di bawah 30.")
    df_under = prep_team_view(under).copy()
    st.dataframe(style_growth_pct_df(df_under), use_container_width=True, height=420)

with col3:
    st.markdown('### 🚫 Team sering OTO "YES"')
    st.caption('Urut berdasarkan jumlah OTO == "YES" (periode ini).')
    st.dataframe(
        prep_oto_view(oto_team),
        use_container_width=True,
        height=420,
        column_config={"OTO Rate %": st.column_config.NumberColumn(format="%.2f")},
        hide_index=True,
    )

# =========================
# EXTRA INSIGHT (tambahan)
# =========================
st.markdown("<hr/>", unsafe_allow_html=True)
st.subheader("🧠 Insight Tambahan (QTY / Retur / AREA)")

# ---------- 1) TEAM qty besar tapi turun ----------
st.markdown("### 1) TEAM QTY besar tapi turun")

team_qty = (
    df_this.groupby("TEAM", as_index=False)
    .agg(QTY_INI=("QTY", "sum"))
    .merge(
        df_last.groupby("TEAM", as_index=False).agg(QTY_LALU=("QTY", "sum")),
        on="TEAM",
        how="left",
    )
    .fillna(0.0)
)
team_qty["DELTA"] = team_qty["QTY_INI"] - team_qty["QTY_LALU"]
team_qty["GROWTH_PCT"] = team_qty.apply(lambda r: safe_growth_pct(r["QTY_INI"], r["QTY_LALU"]), axis=1)

# ambil kandidat TEAM dengan QTY ini terbesar, lalu filter yang turun
TOP_BIG = 30  # bisa kamu ubah 20/50
big_down = (
    team_qty.sort_values("QTY_INI", ascending=False)
    .head(TOP_BIG)
    .query("DELTA < 0")
    .copy()
    .sort_values("DELTA", ascending=True)
    .head(20)
)

if len(big_down) == 0:
    st.info("Tidak ada TEAM 'QTY besar tapi turun' pada filter & periode saat ini.")
else:
    big_down_view = big_down.copy()
    big_down_view["QTY Ini"] = big_down_view["QTY_INI"].map(format_int_id)
    big_down_view["QTY Lalu"] = big_down_view["QTY_LALU"].map(format_int_id)
    big_down_view["Delta"] = big_down_view["DELTA"].map(format_int_id)
    big_down_view["Growth"] = big_down_view["GROWTH_PCT"].apply(growth_badge_html)
    big_down_view = big_down_view[["TEAM", "QTY Ini", "QTY Lalu", "Delta", "Growth"]]
    render_html_table(big_down_view)

# ---------- 2) TEAM paling banyak retur ----------
st.markdown("### 2) TEAM paling banyak retur")

ret_this = df_this[df_this["ROW_TYPE"] == "RETUR"].copy()

if len(ret_this) == 0:
    st.info("Tidak ada data RETUR pada periode ini (berdasarkan kolom STATUS).")
else:
    ret_team = (
        ret_this.groupby("TEAM", as_index=False)
        .agg(
            Retur_Lines=("TEAM", "count"),
            Retur_QTY=("QTY", "sum"),
        )
        .sort_values(["Retur_Lines", "Retur_QTY"], ascending=[False, True])
        .head(20)
        .copy()
    )
    # QTY retur biasanya negatif, biar enak lihat pakai ABS
    ret_team["Retur_QTY (abs)"] = ret_team["Retur_QTY"].abs().map(format_int_id)
    ret_team["Retur_Lines"] = ret_team["Retur_Lines"].map(format_int_id)
    ret_team = ret_team[["TEAM", "Retur_Lines", "Retur_QTY (abs)"]]
    render_html_table(ret_team)

# ---------- 3) Perform AREA (Naik/Turun) ----------
st.markdown("### 3) Perform AREA (Naik / Turun)")

if "AREA" not in df_this.columns or "AREA" not in df_last.columns:
    st.warning("Kolom 'AREA' tidak ditemukan di data yang terbaca. (Pastikan Excel punya kolom AREA & tidak terbuang saat cleaning).")
else:
    area_this = df_this.groupby("AREA", as_index=False).agg(QTY_INI=("QTY", "sum"))
    area_last = df_last.groupby("AREA", as_index=False).agg(QTY_LALU=("QTY", "sum"))
    area = area_this.merge(area_last, on="AREA", how="outer").fillna(0.0)

    area["DELTA"] = area["QTY_INI"] - area["QTY_LALU"]
    area["GROWTH_PCT"] = area.apply(lambda r: safe_growth_pct(r["QTY_INI"], r["QTY_LALU"]), axis=1)

    # tampilkan top naik & top turun
    top_up = area.sort_values("DELTA", ascending=False).head(10).copy()
    top_dn = area.sort_values("DELTA", ascending=True).head(10).copy()

    colA, colB = st.columns(2)
    with colA:
        st.markdown("#### 🔼 Top AREA Naik (QTY)")
        v = top_up.copy()
        v["QTY Ini"] = v["QTY_INI"].map(format_int_id)
        v["QTY Lalu"] = v["QTY_LALU"].map(format_int_id)
        v["Delta"] = v["DELTA"].map(format_int_id)
        v["Growth"] = v["GROWTH_PCT"].apply(growth_badge_html)
        v = v[["AREA", "QTY Ini", "QTY Lalu", "Delta", "Growth"]]
        render_html_table(v)

    with colB:
        st.markdown("#### 🔽 Top AREA Turun (QTY)")
        v = top_dn.copy()
        v["QTY Ini"] = v["QTY_INI"].map(format_int_id)
        v["QTY Lalu"] = v["QTY_LALU"].map(format_int_id)
        v["Delta"] = v["DELTA"].map(format_int_id)
        v["Growth"] = v["GROWTH_PCT"].apply(growth_badge_html)
        v = v[["AREA", "QTY Ini", "QTY Lalu", "Delta", "Growth"]]
        render_html_table(v)
