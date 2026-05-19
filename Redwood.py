import streamlit as st
import pandas as pd
import re
import os
import subprocess
from urllib.parse import quote_plus
from playwright.sync_api import sync_playwright

os.system("playwright install chromium")

st.set_page_config(page_title="Shopee Price Checker", layout="wide")
st.title("Shopee Price Checker")


def extract_ids(url):
    patterns = [
        r"i\.(\d+)\.(\d+)",
        r"product/(\d+)/(\d+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1), match.group(2)

    return "-", "-"


def extract_price(text):
    prices = re.findall(r"Rp\s?[\d\.]+", text)
    return prices[0] if prices else "-"


def scrape_shopee(keyword, limit=20):
    results = []
    encoded_keyword = quote_plus(keyword)

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-blink-features=AutomationControlled",
            ],
        )

        page = browser.new_page(
            viewport={"width": 1366, "height": 768},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
        )

        url = f"https://shopee.co.id/search?keyword={encoded_keyword}"
        page.goto(url, wait_until="networkidle", timeout=90000)

        page.wait_for_timeout(8000)

        for _ in range(5):
            page.mouse.wheel(0, 1200)
            page.wait_for_timeout(1500)

        links = page.query_selector_all("a[href]")

        for link in links:
            try:
                href = link.get_attribute("href")
                text = link.inner_text().strip()

                if not href:
                    continue

                if "i." not in href and "/product/" not in href:
                    continue

                if "Rp" not in text:
                    continue

                full_url = href
                if href.startswith("/"):
                    full_url = "https://shopee.co.id" + href

                shopid, itemid = extract_ids(full_url)
                price = extract_price(text)

                lines = [x.strip() for x in text.split("\n") if x.strip()]
                title = "-"

                for line in lines:
                    if "Rp" not in line and len(line) > 10:
                        title = line
                        break

                results.append(
                    {
                        "title": title,
                        "price": price,
                        "shopid": shopid,
                        "itemid": itemid,
                        "url": full_url,
                    }
                )

            except Exception:
                continue

        browser.close()

    clean = []
    seen = set()

    for item in results:
        key = item["itemid"]

        if key != "-" and key not in seen:
            seen.add(key)
            clean.append(item)

    return clean[:limit]


keyword = st.text_input("Keyword Produk", placeholder="contoh: e1404fa ryzen 3")
limit = st.slider("Jumlah Produk", 5, 100, 20)

if st.button("Cari"):
    if not keyword:
        st.error("Keyword wajib diisi")
    else:
        with st.spinner("Scraping Shopee..."):
            data = scrape_shopee(keyword, limit)

        if data:
            df = pd.DataFrame(data)
            st.success(f"{len(df)} produk ditemukan")
            st.dataframe(df, use_container_width=True)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download CSV",
                csv,
                "shopee_products.csv",
                "text/csv",
            )
        else:
            st.warning("Tidak ada hasil")
