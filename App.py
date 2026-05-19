import streamlit as st
import pandas as pd
import re
from playwright.sync_api import sync_playwright
import os

# auto install chromium
os.system("playwright install chromium")

st.set_page_config(
    page_title="Shopee Checker",
    layout="wide"
)

st.title("Shopee Price Checker")


def clean_price(text):
    prices = re.findall(r'Rp\\s?[\\d\\.]+', text)

    if prices:
        return prices[0]

    return "-"


def extract_ids(url):

    match = re.search(r'i\\.(\\d+)\\.(\\d+)', url)

    if match:
        return match.group(1), match.group(2)

    return "-", "-"


def scrape_shopee(keyword, limit=20):

    results = []

    with sync_playwright() as p:

        browser = p.chromium.launch(
            headless=True
        )

        page = browser.new_page()

        url = f"https://shopee.co.id/search?keyword={keyword}"

        page.goto(url, timeout=60000)

        page.wait_for_timeout(7000)

        products = page.query_selector_all('a[data-sqe="link"]')

        for product in products:

            try:

                href = product.get_attribute("href")

                if not href:
                    continue

                full_url = "https://shopee.co.id" + href

                text = product.inner_text()

                lines = text.split("\\n")

                title = lines[0] if lines else "-"

                price = clean_price(text)

                shopid, itemid = extract_ids(href)

                results.append({
                    "title": title,
                    "price": price,
                    "shopid": shopid,
                    "itemid": itemid,
                    "url": full_url
                })

            except:
                pass

        browser.close()

    return results[:limit]


keyword = st.text_input(
    "Keyword Produk",
    placeholder="contoh: e1404fa ryzen 3"
)

limit = st.slider(
    "Jumlah Produk",
    5,
    100,
    20
)

if st.button("Cari"):

    if keyword:

        with st.spinner("Scraping Shopee..."):

            results = scrape_shopee(keyword, limit)

            if results:

                df = pd.DataFrame(results)

                st.success(f"{len(df)} produk ditemukan")

                st.dataframe(
                    df,
                    use_container_width=True
                )

            else:
                st.warning("Tidak ada hasil")

    else:
        st.error("Keyword wajib diisi")
