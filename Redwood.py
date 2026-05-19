import streamlit as st
import pandas as pd
import re
import os
from urllib.parse import quote_plus
from playwright.sync_api import sync_playwright

# Install Chromium saat running di Streamlit Cloud
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


def scrape_shopee(keyword, limit=20, debug=False):
    results = []
    debug_info = {
        "url": "",
        "page_title": "",
        "current_url": "",
        "html_sample": "",
        "screenshot_path": "",
        "total_links": 0,
        "candidate_links": 0,
        "error": "",
    }

    encoded_keyword = quote_plus(keyword)

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-blink-features=AutomationControlled",
                "--disable-gpu",
                "--disable-setuid-sandbox",
            ],
        )

        page = browser.new_page(
            viewport={"width": 1366, "height": 1600},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            locale="id-ID",
        )

        try:
            url = f"https://shopee.co.id/search?keyword={encoded_keyword}"
            debug_info["url"] = url

            page.goto(url, wait_until="domcontentloaded", timeout=90000)
            page.wait_for_timeout(10000)

            for _ in range(8):
                page.mouse.wheel(0, 1000)
                page.wait_for_timeout(1000)

            debug_info["page_title"] = page.title()
            debug_info["current_url"] = page.url

            if debug:
                screenshot_path = "/tmp/shopee_debug.png"
                page.screenshot(path=screenshot_path, full_page=True)
                debug_info["screenshot_path"] = screenshot_path
                html = page.content()
                debug_info["html_sample"] = html[:4000]

            links = page.query_selector_all("a[href]")
            debug_info["total_links"] = len(links)

            for link in links:
                try:
                    href = link.get_attribute("href")
                    text = link.inner_text().strip()

                    if not href:
                        continue

                    is_product_link = (
                        "i." in href
                        or "/product/" in href
                        or re.search(r"-i\.\d+\.\d+", href or "")
                    )

                    if not is_product_link:
                        continue

                    debug_info["candidate_links"] += 1

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
                        if "Rp" not in line and len(line) > 8:
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

        except Exception as e:
            debug_info["error"] = str(e)

        finally:
            browser.close()

    clean = []
    seen = set()

    for item in results:
        key = item["itemid"] + item["shopid"]

        if key not in seen:
            seen.add(key)
            clean.append(item)

    return clean[:limit], debug_info


keyword = st.text_input("Keyword Produk", placeholder="contoh: e1404fa ryzen 3")
limit = st.slider("Jumlah Produk", 5, 100, 20)
debug_mode = st.checkbox("Debug mode - tampilkan screenshot halaman Shopee dari server")

if st.button("Cari"):
    if not keyword:
        st.error("Keyword wajib diisi")
    else:
        with st.spinner("Scraping Shopee..."):
            data, debug_info = scrape_shopee(keyword, limit, debug=debug_mode)

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

        if debug_mode:
            st.subheader("Debug Info")
            st.write("Target URL:", debug_info["url"])
            st.write("Current URL:", debug_info["current_url"])
            st.write("Page title:", debug_info["page_title"])
            st.write("Total link ditemukan:", debug_info["total_links"])
            st.write("Candidate product link:", debug_info["candidate_links"])

            if debug_info["error"]:
                st.error(debug_info["error"])

            if debug_info["screenshot_path"] and os.path.exists(debug_info["screenshot_path"]):
                st.image(debug_info["screenshot_path"], caption="Screenshot Shopee dari server Streamlit")

            with st.expander("HTML sample"):
                st.code(debug_info["html_sample"][:4000])
