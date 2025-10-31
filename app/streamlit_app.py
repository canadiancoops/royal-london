# streamlit_app.py
# Royal London Insight Hub — minimal, clean reboot:
#  - Voice of the Customer (Trustpilot)
#  - Voice of the Industry (FCA + FOS)
#  - Voice of the Employee (Glassdoor)
#  - Requests-first with a Demo fallback
#  - Optional Playwright path for Glassdoor to bypass 403 and parse Apollo JSON

import re
import time
import json
from datetime import date
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ---------- basic scraping deps ----------
SCRAPE_AVAILABLE = True
try:
    import requests
    from bs4 import BeautifulSoup
except Exception:
    SCRAPE_AVAILABLE = False

# ---------- playwright (optional) ----------
PLAYWRIGHT_AVAILABLE = True
try:
    from playwright.sync_api import sync_playwright
except Exception:
    PLAYWRIGHT_AVAILABLE = False

# =========================
# App config / constants
# =========================
st.set_page_config(page_title="Royal London Insight Hub", page_icon="🏛️", layout="wide")
st.title("Royal London Insight Hub")

COMPANY = "Royal London"
GLASSDOOR_EMPLOYER_ID = "E12432"  # numeric core is 12432
RL_COLOR  = "#6A0DAD"
NEG_COLOR = "#CF242A"
NEU_COLOR = "#F0B323"
POS_COLOR = "#2E7D32"
AVG_LINE_COLOR = "#68707a"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/123.0 Safari/537.36",
    "Accept-Language": "en-GB,en;q=0.9",
}

# =========================
# Minimal styling
# =========================
st.markdown(
    """
    <style>
      .metric-card { border: 1px solid #e6e6e6; border-radius: 10px; padding: 12px; background: #fafafa; }
      .small-note { font-size: 12px; color: #666; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# Utils & schema
# =========================
def _to_datetime_safe(values):
    try:
        return pd.to_datetime(values, errors="coerce", format="mixed")  # pandas >= 2.0
    except TypeError:
        return pd.to_datetime(values, errors="coerce", dayfirst=True)

REVIEW_SCHEMA = ["site", "date", "rating", "text"]

def _empty_reviews_df(site: str) -> pd.DataFrame:
    df = pd.DataFrame({"site": [], "date": [], "rating": [], "text": []})
    df = df.astype({"site": "object", "date": "datetime64[ns]", "rating": "float64", "text": "object"})
    df["site"] = site
    return df.iloc[0:0].copy()

def _coerce_reviews_schema(df: pd.DataFrame, site: str) -> pd.DataFrame:
    if df is None or df.empty:
        return _empty_reviews_df(site)
    d = df.copy()
    d.columns = [str(c).strip().lower() for c in d.columns]
    if "site" not in d.columns: d["site"] = site
    date_col   = next((c for c in ["date","review_date","time","posted","date_published"] if c in d.columns), "date")
    rating_col = next((c for c in ["rating","stars","score","ratingvalue"] if c in d.columns), "rating")
    text_col   = next((c for c in ["text","review","comment","body","content","summary","review_body"] if c in d.columns), "text")
    for need in [date_col, rating_col, text_col]:
        if need not in d.columns: d[need] = np.nan if need != text_col else ""
    d = d[["site", date_col, rating_col, text_col]].rename(columns={date_col:"date", rating_col:"rating", text_col:"text"})
    d["site"] = site
    d["date"] = _to_datetime_safe(d["date"])
    d["rating"] = pd.to_numeric(d["rating"], errors="coerce").clip(1, 5)
    d["text"] = d["text"].astype(str)
    d = d.dropna(subset=["date"]).reset_index(drop=True)
    return d[REVIEW_SCHEMA]

def within_last_365(d: pd.Timestamp) -> bool:
    today = pd.Timestamp(date.today())
    return (today - d).days <= 365

def round_1dp(v: float) -> float:
    return float(np.round(float(v) + 1e-8, 1))

def aggregate_monthly(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return pd.DataFrame(columns=["site","month","avg"])
    g = df.copy()
    g["month"] = g["date"].dt.to_period("M").dt.to_timestamp()
    g = g.groupby(["site","month"], as_index=False)["rating"].mean()
    g = g.rename(columns={"rating":"avg"})
    g["avg"] = g["avg"].map(round_1dp)
    return g

def color_bucket(val: float) -> str:
    if val <= 2.0: return "Negative"
    if val <= 3.5: return "Neutral"
    return "Positive"

def chart_by_source(df: pd.DataFrame, title: str):
    if df.empty: return go.Figure()
    g = df.groupby("site", as_index=False)["rating"].mean()
    g["avg"] = g["rating"].map(round_1dp)
    g["bucket"] = g["avg"].map(color_bucket)
    g = g.sort_values("avg", ascending=True)
    fig = px.bar(
        g, x="site", y="avg", color="bucket",
        color_discrete_map={"Negative":NEG_COLOR,"Neutral":NEU_COLOR,"Positive":POS_COLOR},
        labels={"site":"Source","avg":"Avg rating (1–5)"},
        title=title
    )
    fig.update_yaxes(range=[1,5], dtick=0.5)
    return fig

def chart_monthly(df_month: pd.DataFrame, title: str):
    if df_month.empty: return go.Figure()
    fig = px.line(df_month, x="month", y="avg", color="site", markers=True,
                  labels={"month":"Month","avg":"Avg rating (1–5)"}, title=title)
    # overall average line
    y = round_1dp(df_month["avg"].mean())
    fig.add_hline(y=y, line_color=AVG_LINE_COLOR, line_dash="dash", annotation_text=f"Avg {y}")
    fig.update_yaxes(range=[1,5], dtick=0.5)
    return fig

def verbatim_table(df: pd.DataFrame):
    if df.empty: return df
    cols = [c for c in ["date","site","rating","text"] if c in df.columns]
    return df[cols].sort_values("date", ascending=False)

# ---------- period filter ----------
def _filter_period(df_month: pd.DataFrame, key: str) -> pd.DataFrame:
    if df_month.empty: return df_month
    today = pd.Timestamp(date.today().replace(day=1))
    if key == "R12M":
        start = today - pd.DateOffset(months=12)
        end = today + pd.DateOffset(months=1) - pd.DateOffset(days=1)
    elif key == "R6M":
        start = today - pd.DateOffset(months=6)
        end = today + pd.DateOffset(months=1) - pd.DateOffset(days=1)
    elif key == "R3M":
        start = today - pd.DateOffset(months=3)
        end = today + pd.DateOffset(months=1) - pd.DateOffset(days=1)
    else:
        start = today - pd.DateOffset(months=1)
        end = today - pd.DateOffset(days=1)
    return df_month[(df_month["month"] >= start) & (df_month["month"] <= end)].copy()

# =========================
# Trustpilot (Requests)
# =========================
def _tp_cards(soup: BeautifulSoup) -> List[dict]:
    rows, seen = [], set()
    def _text(el): return el.get_text(" ", strip=True) if el else ""
    cards = []
    cards += soup.select("article")
    cards += soup.select("div[data-reviewid], div[data-review-id]")
    cards += soup.select('div[class*="review-card"]')
    cards += soup.select('div[class*="styles_reviewCard__"]')
    cards += soup.select('section[class*="review"]')
    for c in cards:
        rating = None
        star_img = c.select_one("img[alt*='star' i]")
        if star_img and star_img.get("alt"):
            m = re.search(r"(\d+)", star_img["alt"])
            if m:
                try: rating = int(m.group(1))
                except: pass
        if rating is None:
            ar = (c.get("aria-label") or "") + " " + _text(c.select_one('[aria-label*="star" i]'))
            m = re.search(r"(\d)\s*star", ar, flags=re.I)
            if m: rating = int(m.group(1))
        if rating is None:
            m = re.search(r"Rated\s+(\d(?:\.\d)?)\s+out of 5", c.get_text(" ", strip=True), re.I)
            if m: rating = float(m.group(1))
        date_iso = ""
        t = c.select_one("time[datetime]")
        if t and t.has_attr("datetime"): date_iso = t["datetime"]
        title = _text(c.select_one("h2, h3"))
        body  = _text(c.select_one("p, div[data-review-content-typography], div[class*='reviewContent']"))
        text = (title + " " + body).strip()
        if not (date_iso and rating and text): continue
        rid = c.get("data-reviewid") or c.get("data-review-id") or title
        key = (rid, date_iso, rating, text[:50])
        if key in seen: continue
        seen.add(key)
        rows.append({"site":"Trustpilot","date":date_iso,"rating":rating,"text":text})
    return rows

def scrape_trustpilot(max_pages: int = 10, delay: float = 0.6) -> pd.DataFrame:
    if not SCRAPE_AVAILABLE: return _empty_reviews_df("Trustpilot")
    variants = [
        "https://uk.trustpilot.com/review/www.royallondon.com",
        "https://uk.trustpilot.com/review/royallondon.com",
        "https://uk.trustpilot.com/review/royal-london",
    ]
    rows = []
    for base in variants:
        hits = 0
        for p in range(1, max_pages + 1):
            try:
                r = requests.get(f"{base}?page={p}", headers=HEADERS, timeout=30)
                if not r.ok: break
                soup = BeautifulSoup(r.text, "html.parser")
                page_rows = _tp_cards(soup)
                # last 365 only, keep valid ratings
                out = []
                for rr in page_rows:
                    d = _to_datetime_safe(rr["date"])
                    if pd.isna(d) or not within_last_365(d): continue
                    rr["date"] = d
                    rr["rating"] = pd.to_numeric(rr["rating"], errors="coerce")
                    if pd.isna(rr["rating"]): continue
                    out.append(rr)
                rows.extend(out); hits += len(out)
                time.sleep(delay)
                if len(out) == 0: break
            except Exception:
                break
        if hits > 0: break
    return _coerce_reviews_schema(pd.DataFrame(rows), "Trustpilot")

# =========================
# Industry (FCA/FOS)
# =========================
_NEG_KWS = {
    "fine","penalty","enforcement","breach","censure","sanction","reprimand",
    "redress","mis-selling","mis selling","failings","shortcomings","warning",
    "ban","prohibition","unlawful","illegal","prosecut"
}
_POS_KWS = {
    "authorised","authorized","permission","approved","approval",
    "no further action","resolved","improvement","commitment",
    "guidance","consultation","update","good practice"
}
def _industry_score(text: str) -> float:
    s = (text or "").lower()
    score = 3.0 + 0.6*sum(w in s for w in _POS_KWS) - 0.8*sum(w in s for w in _NEG_KWS)
    return float(np.clip(score, 1.0, 5.0))

def _parse_fca_list(html: str) -> List[dict]:
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    items = []
    items += soup.select("article")
    items += soup.select("li.search-result, li.search-results__item, div.search-result, div.search-results__item")
    for it in items:
        a = it.select_one("h3 a, h2 a, h4 a, a")
        title = a.get_text(" ", strip=True) if a else ""
        p = it.select_one("p, .summary, .teaser")
        summary = p.get_text(" ", strip=True) if p else ""
        t = it.find("time")
        dt = _to_datetime_safe(t.get("datetime") if t and t.has_attr("datetime") else (t.get_text(strip=True) if t else None))
        if title and not pd.isna(dt):
            txt = f"{title} — {summary}".strip(" —")
            rows.append({"site":"FCA","date":dt,"rating":_industry_score(txt),"text":txt})
    return rows

def scrape_fca(query: str = COMPANY, max_pages: int = 2, delay: float = 0.4) -> pd.DataFrame:
    if not SCRAPE_AVAILABLE: return _empty_reviews_df("FCA")
    q = requests.utils.quote(query)
    urls = [
        f"https://www.fca.org.uk/search-results?keywords={q}",
        f"https://www.fca.org.uk/news/search-results?search_term={q}",
    ]
    rows = []
    for base in urls:
        hits = 0
        for p in range(max_pages):
            url = base + (f"&start={p*10}" if "search-results?" in base else "")
            try:
                r = requests.get(url, headers=HEADERS, timeout=20)
                if not r.ok: break
                page_rows = [r for r in _parse_fca_list(r.text) if within_last_365(r["date"])]
                rows.extend(page_rows); hits += len(page_rows)
                time.sleep(delay)
                if len(page_rows) == 0: break
            except Exception:
                break
        if hits > 0: break
    return _coerce_reviews_schema(pd.DataFrame(rows), "FCA")

def _parse_fos_list(html: str) -> List[dict]:
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    items = []
    items += soup.select("article, li, div.card, div.views-row")
    for it in items:
        a = it.select_one("h3 a, h2 a, a")
        title = a.get_text(" ", strip=True) if a else ""
        p = it.select_one("p, .summary, .teaser")
        summary = p.get_text(" ", strip=True) if p else ""
        t = it.find("time")
        dt = _to_datetime_safe(t.get("datetime") if t and t.has_attr("datetime") else (t.get_text(strip=True) if t else None))
        if title and not pd.isna(dt):
            txt = f"{title} — {summary}".strip(" —")
            rows.append({"site":"FOS","date":dt,"rating":_industry_score(txt),"text":txt})
    return rows

def scrape_fos(query: str = COMPANY, max_pages: int = 1) -> pd.DataFrame:
    if not SCRAPE_AVAILABLE: return _empty_reviews_df("FOS")
    q = requests.utils.quote(query)
    urls = [
        f"https://www.financial-ombudsman.org.uk/decisions-case-studies?search={q}",
        f"https://www.financial-ombudsman.org.uk/search?search={q}",
    ]
    rows = []
    for url in urls:
        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
            if not r.ok: continue
            page_rows = [r for r in _parse_fos_list(r.text) if within_last_365(r["date"])]
            rows.extend(page_rows)
            if page_rows: break
        except Exception:
            continue
    return _coerce_reviews_schema(pd.DataFrame(rows), "FOS")

# =========================
# Glassdoor
# =========================
def _gd_extract_apollo(html: str) -> Optional[dict]:
    # 1) Next.js payload
    m = re.search(r'<script id="__NEXT_DATA__" type="application/json">\s*(\{.*?\})\s*</script>', html, flags=re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(1))
            cache = data.get("props", {}).get("pageProps", {}).get("apolloCache")
            if cache: return cache.get("ROOT_QUERY") or cache
        except Exception:
            pass
    # 2) apolloState inline
    m = re.search(r'apolloState":\s*({.*?})\s*};', html, flags=re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(1))
            return data.get("ROOT_QUERY") or data
        except Exception:
            pass
    return None

def _gd_parse_reviews_from_html(html: str) -> Tuple[List[dict], int]:
    cache = _gd_extract_apollo(html)
    if not cache: return [], 1
    node = None
    for k, v in cache.items():
        if isinstance(v, dict) and v.get("reviews") and (k.lower().startswith("employerreviewsrg") or k.lower().startswith("employerreviews")):
            node = v; break
    if not node: return [], 1
    return node.get("reviews", []), int(node.get("numberOfPages", 1) or 1)

def _gd_flatten_rows(items: List[dict]) -> List[dict]:
    rows = []
    for r in items:
        rating = r.get("ratingOverall") or r.get("overallRating")
        text_parts = []
        if r.get("summary"): text_parts.append(str(r["summary"]))
        if r.get("pros"):    text_parts.append(f"Pros: {r['pros']}")
        if r.get("cons"):    text_parts.append(f"Cons: {r['cons']}")
        if r.get("advice"):  text_parts.append(f"Advice: {r['advice']}")
        rows.append({
            "site":"Glassdoor",
            "date": r.get("reviewDateTime") or r.get("atDateTime"),
            "rating": rating,
            "text": ". ".join(tp for tp in text_parts if tp).strip(". "),
        })
    return rows

def scrape_glassdoor_requests(slug: str = "Royal-London", employer_id: str = GLASSDOOR_EMPLOYER_ID, max_pages: int = 6) -> pd.DataFrame:
    if not SCRAPE_AVAILABLE: return _empty_reviews_df("Glassdoor")
    # Requests approach often 403s; try both .co.uk and .com, with simple card scrape
    eid = employer_id[1:] if employer_id.startswith("E") else employer_id
    bases = [
        f"https://www.glassdoor.co.uk/Reviews/{slug}-Reviews-{employer_id}.htm",
        f"https://www.glassdoor.com/Reviews/{slug}-Reviews-{employer_id}.htm",
        f"https://www.glassdoor.co.uk/Reviews/{slug}-Reviews-EI_IE{eid}.htm",
        f"https://www.glassdoor.com/Reviews/{slug}-Reviews-EI_IE{eid}.htm",
    ]
    rows = []
    for base0 in list(dict.fromkeys(bases)):
        base = base0 + "?filter.iso3Language=eng&sort.sortType=RD&sort.ascending=false&p="
        hits = 0
        for p in range(1, max_pages+1):
            r = requests.get(base + str(p), headers=HEADERS, timeout=25)
            if r.status_code != 200:
                break
            soup = BeautifulSoup(r.text, "html.parser")
            # Try hidden JSON first
            items, _ = _gd_parse_reviews_from_html(r.text)
            if items:
                rows.extend(_gd_flatten_rows(items)); hits += len(items); continue
            # fallback to visible cards (rarely stable)
            cards = soup.select("div.review-card, div.reviewCard, li.empReview, div.gdReview, div.review")
            page_hits = 0
            for c in cards:
                t = c.find("time")
                d = _to_datetime_safe((t.get("datetime") if t and t.has_attr("datetime") else t.get_text(strip=True)) if t else None)
                rating = None
                aria = c.find(attrs={"aria-label": re.compile(r"out of 5", re.I)})
                if aria:
                    m = re.search(r"(\d(?:\.\d)?)\s*out of\s*5", aria.get("aria-label",""), re.I)
                    if m: rating = float(m.group(1))
                if rating is None:
                    rv = c.find(attrs={"data-test":"rating-value"})
                    if rv:
                        try: rating = float(rv.get_text(strip=True))
                        except: pass
                body = c.select_one("[data-test='reviewBody']") or c
                text = (body.get_text(" ", strip=True) if body else "").strip()
                if d is not pd.NaT and rating and text:
                    rows.append({"site":"Glassdoor","date":d,"rating":rating,"text":text}); page_hits += 1
            if page_hits == 0 and not items: break
            hits += page_hits
            time.sleep(0.3)
        if hits > 0: break
    return _coerce_reviews_schema(pd.DataFrame(rows), "Glassdoor")

def scrape_glassdoor_playwright(
    slug: str = "Royal-London",
    employer_id: str = GLASSDOOR_EMPLOYER_ID,
    max_pages: int = 8,
    country_id: int = 2,
    headless: bool = True,
    timeout_ms: int = 60000,
) -> pd.DataFrame:
    if not PLAYWRIGHT_AVAILABLE:
        return _empty_reviews_df("Glassdoor")

    rows_all = []
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=headless)
            context = browser.new_context(
                user_agent=HEADERS["User-Agent"],
                locale="en-GB",
            )
            context.add_cookies([
                {"name":"tldp","value":str(country_id),"domain":".glassdoor.co.uk","path":"/"},
                {"name":"tldp","value":str(country_id),"domain":".glassdoor.com","path":"/"},
            ])
            page = context.new_page()
            page.set_default_timeout(timeout_ms)

            base = (
                f"https://www.glassdoor.co.uk/Reviews/{slug}-Reviews-"
                f"{employer_id}.htm?filter.iso3Language=eng&sort.sortType=RD&sort.ascending=false"
            )

            page.goto(base, wait_until="domcontentloaded")
            page.wait_for_load_state("networkidle")
            html = page.content()
            items, pages = _gd_parse_reviews_from_html(html)
            rows_all.extend(_gd_flatten_rows(items))

            total = min(int(pages or 1), max_pages)
            for p in range(2, total + 1):
                url = re.sub(r"\.htm", f"_P{p}.htm", base)
                page.goto(url, wait_until="domcontentloaded")
                page.wait_for_load_state("networkidle")
                html = page.content()
                items, _ = _gd_parse_reviews_from_html(html)
                rows_all.extend(_gd_flatten_rows(items))

            context.close()
            browser.close()
    except Exception as e:
        st.warning(f"Playwright scrape failed (fallback to Requests): {e}")
        return scrape_glassdoor_requests(slug, employer_id, max_pages=6)

    return _coerce_reviews_schema(pd.DataFrame(rows_all), "Glassdoor")

# =========================
# Demo data
# =========================
def _demo_reviews(site: str, seed: int) -> pd.DataFrame:
    today = pd.Timestamp(date.today())
    months = [today - pd.DateOffset(months=i) for i in range(12)]
    rng = np.random.default_rng(seed)
    rows = []
    for m in months:
        for _ in range(rng.integers(3, 7)):
            rating = float(rng.choice([1,2,3,4,5], p=[0.12,0.18,0.22,0.28,0.20]))
            txt = f"{site} review mentioning service and app."
            rows.append({"site":site,"date":m + pd.Timedelta(days=int(rng.integers(0,27))),"rating":rating,"text":txt})
    return _coerce_reviews_schema(pd.DataFrame(rows), site)

# =========================
# Sidebar controls + diagnostics
# =========================
st.sidebar.header("Controls")
period_key = st.sidebar.radio("Period", ["R12M","R6M","R3M","Last month"], index=0)
use_demo = st.sidebar.toggle("Demo mode (use synthetic data if a scrape is empty)", value=False)
use_playwright = st.sidebar.toggle("Use Playwright for Glassdoor (bypass 403)", value=False)
auto_export = st.sidebar.toggle("Auto-save CSV after scraping", value=True)
gd_max_pages   = st.sidebar.slider("Glassdoor pages to fetch", 1, 20, 8)
gd_country_id  = st.sidebar.number_input("Glassdoor countryId", min_value=1, max_value=300, value=2)
gd_show_browser = st.sidebar.toggle("Show browser (debug Glassdoor)", value=False)

dbg = st.sidebar.expander("Diagnostics")
with dbg:
    st.caption(f"Python deps: requests={'✅' if SCRAPE_AVAILABLE else '❌'} | Playwright={'✅' if PLAYWRIGHT_AVAILABLE else '❌'}")
    if SCRAPE_AVAILABLE:
        try:
            tp = requests.get("https://uk.trustpilot.com/review/www.royallondon.com", headers=HEADERS, timeout=12)
            st.write("Trustpilot GET:", tp.status_code)
        except Exception as e:
            st.write("Trustpilot error:", str(e))
        try:
            gd = requests.get("https://www.glassdoor.co.uk/Reviews/Royal-London-Reviews-E12432.htm", headers=HEADERS, timeout=12)
            st.write("Glassdoor GET:", gd.status_code)
        except Exception as e:
            st.write("Glassdoor error:", str(e))
        try:
            f1 = requests.get(f"https://www.fca.org.uk/search-results?keywords={requests.utils.quote(COMPANY)}", headers=HEADERS, timeout=12)
            st.write("FCA GET:", f1.status_code)
        except Exception as e:
            st.write("FCA error:", str(e))
        try:
            f2 = requests.get(f"https://www.financial-ombudsman.org.uk/decisions-case-studies?search={requests.utils.quote(COMPANY)}", headers=HEADERS, timeout=12)
            st.write("FOS GET:", f2.status_code)
        except Exception as e:
            st.write("FOS error:", str(e))

# =========================
# Tabs
# =========================
tab1, tab2, tab3 = st.tabs([
    "Voice of the Customer (Trustpilot)",
    "Voice of the Industry (FCA + FOS)",
    "Voice of the Employee (Glassdoor)"
])

# =========================
# TAB 1: Trustpilot
# =========================
with tab1:
    st.subheader("Voice of the Customer — Trustpilot")
    run_tp = st.button("Scrape Trustpilot now", type="primary", use_container_width=True)

    tp_df = _empty_reviews_df("Trustpilot")
    msg = st.empty()
    if run_tp:
        if SCRAPE_AVAILABLE:
            with st.spinner("Scraping Trustpilot…"):
                tp_raw = scrape_trustpilot()
                tp_df = _coerce_reviews_schema(tp_raw, "Trustpilot")
                if auto_export and not tp_df.empty:
                    tp_df.to_csv("trustpilot.csv", index=False, encoding="utf-8")
                    msg.success("Saved trustpilot.csv")
        else:
            msg.error("Install: requests, beautifulsoup4")

    if tp_df.empty and use_demo:
        tp_df = _demo_reviews("Trustpilot", seed=42)
        msg.info("Showing DEMO Trustpilot data (scrape empty/blocked).")

    st.markdown(f"**Rows loaded:** `{len(tp_df)}`")
    if tp_df.empty:
        st.info("No Trustpilot reviews loaded.")
    else:
        mon = aggregate_monthly(tp_df)
        sel = _filter_period(mon, period_key)
        span = tp_df[tp_df["date"].dt.to_period("M").dt.to_timestamp().isin(sel["month"].unique())] if not sel.empty else tp_df

        c1, c2 = st.columns([1.2, 1.8])
        with c1:
            st.plotly_chart(chart_by_source(span, "Avg rating by source"), use_container_width=True)
        with c2:
            st.plotly_chart(chart_monthly(sel, f"Customer sentiment — {period_key}"), use_container_width=True)

        st.markdown("### Verbatim")
        st.dataframe(verbatim_table(span), use_container_width=True, height=320)
        st.download_button("Download Trustpilot CSV", data=tp_df.to_csv(index=False), file_name="trustpilot.csv", mime="text/csv", use_container_width=True)

# =========================
# TAB 2: Industry (FCA + FOS)
# =========================
with tab2:
    st.subheader("Voice of the Industry — FCA & FOS")
    colA, colB = st.columns(2)
    with colA:
        run_fca = st.button("Scrape FCA", use_container_width=True)
    with colB:
        run_fos = st.button("Scrape FOS", use_container_width=True)

    fca_df = _empty_reviews_df("FCA")
    fos_df = _empty_reviews_df("FOS")
    if run_fca and SCRAPE_AVAILABLE:
        with st.spinner("Scraping FCA…"):
            fca_df = _coerce_reviews_schema(scrape_fca(COMPANY), "FCA")
            if auto_export and not fca_df.empty:
                fca_df.to_csv("fca.csv", index=False, encoding="utf-8")
                st.success("Saved fca.csv")
    if run_fos and SCRAPE_AVAILABLE:
        with st.spinner("Scraping FOS…"):
            fos_df = _coerce_reviews_schema(scrape_fos(COMPANY), "FOS")
            if auto_export and not fos_df.empty:
                fos_df.to_csv("fos.csv", index=False, encoding="utf-8")
                st.success("Saved fos.csv")

    if fca_df.empty and use_demo:
        fca_df = _demo_reviews("FCA", seed=7)
    if fos_df.empty and use_demo:
        fos_df = _demo_reviews("FOS", seed=8)

    both = pd.concat([fca_df, fos_df], ignore_index=True) if (not fca_df.empty or not fos_df.empty) else pd.DataFrame()
    st.markdown(f"**Rows loaded:** FCA `{len(fca_df)}`, FOS `{len(fos_df)}`, Combined `{len(both)}`")

    if both.empty:
        st.info("No FCA/FOS items loaded.")
    else:
        mon = aggregate_monthly(both)
        sel = _filter_period(mon, period_key)
        span = both[both["date"].dt.to_period("M").dt.to_timestamp().isin(sel["month"].unique())] if not sel.empty else both

        c1, c2 = st.columns([1.2, 1.8])
        with c1:
            st.plotly_chart(chart_by_source(span, "Avg rating by source (Industry)"), use_container_width=True)
        with c2:
            st.plotly_chart(chart_monthly(sel, f"Industry sentiment — {period_key}"), use_container_width=True)

        st.markdown("### Verbatim (FCA & FOS)")
        st.dataframe(verbatim_table(span), use_container_width=True, height=320)

        c3, c4, c5 = st.columns(3)
        with c3:
            st.download_button("Download FCA CSV", data=fca_df.to_csv(index=False), file_name="fca.csv", mime="text/csv", use_container_width=True, disabled=fca_df.empty)
        with c4:
            st.download_button("Download FOS CSV", data=fos_df.to_csv(index=False), file_name="fos.csv", mime="text/csv", use_container_width=True, disabled=fos_df.empty)
        with c5:
            st.download_button("Download Combined CSV", data=both.to_csv(index=False), file_name="industry_combined.csv", mime="text/csv", use_container_width=True, disabled=both.empty)

# =========================
# TAB 3: Glassdoor
# =========================
with tab3:
    st.subheader("Voice of the Employee — Glassdoor")

    # Trigger button defines run_gd on every run (prevents NameError)
    run_gd = st.button("Scrape Glassdoor now", use_container_width=True)

    # Safety defaults if globals not set for any reason
    gd_max_pages    = int(globals().get("gd_max_pages", gd_max_pages if "gd_max_pages" in globals() else 8))
    gd_country_id   = int(globals().get("gd_country_id", gd_country_id if "gd_country_id" in globals() else 2))
    gd_show_browser = bool(globals().get("gd_show_browser", gd_show_browser if "gd_show_browser" in globals() else False))

    gd_df = _empty_reviews_df("Glassdoor")
    msg = st.empty()

    if run_gd:
        with st.spinner("Scraping Glassdoor…"):
            if use_playwright and PLAYWRIGHT_AVAILABLE:
                # Try simple signature first
                try:
                    gd_df = scrape_glassdoor_playwright(
                        "Royal-London",
                        employer_id=GLASSDOOR_EMPLOYER_ID,
                        max_pages=gd_max_pages,
                        country_id=gd_country_id,
                    )
                except TypeError:
                    # Some versions expose headless/timeout
                    try:
                        gd_df = scrape_glassdoor_playwright(
                            "Royal-London",
                            employer_id=GLASSDOOR_EMPLOYER_ID,
                            max_pages=gd_max_pages,
                            country_id=gd_country_id,
                            headless=(not gd_show_browser),
                            timeout_ms=90000,
                        )
                    except Exception as e:
                        st.warning(f"Playwright scrape failed ({e}). Falling back to Requests.")
                        gd_df = scrape_glassdoor_requests(
                            "Royal-London",
                            employer_id=GLASSDOOR_EMPLOYER_ID,
                            max_pages=6,
                        )
                except Exception as e:
                    st.warning(f"Playwright scrape failed ({e}). Falling back to Requests.")
                    gd_df = scrape_glassdoor_requests(
                        "Royal-London",
                        employer_id=GLASSDOOR_EMPLOYER_ID,
                        max_pages=6,
                    )
            else:
                if use_playwright and not PLAYWRIGHT_AVAILABLE:
                    st.warning("Playwright not installed — using Requests fallback.")
                gd_df = scrape_glassdoor_requests(
                    "Royal-London",
                    employer_id=GLASSDOOR_EMPLOYER_ID,
                    max_pages=6,
                )

            if auto_export and not gd_df.empty:
                gd_df.to_csv("glassdoor.csv", index=False, encoding="utf-8")
                msg.success("Saved glassdoor.csv")

    # Demo fallback if nothing loaded
    if gd_df.empty and use_demo:
        gd_df = _demo_reviews("Glassdoor", seed=9)
        msg.info("Showing DEMO Glassdoor data (scrape empty/blocked).")

    st.markdown(f"**Rows loaded:** `{len(gd_df)}`")

    if gd_df.empty:
        st.info("No Glassdoor reviews loaded.")
    else:
        # Monthly aggregation & period selection
        mon = aggregate_monthly(gd_df)
        sel = _filter_period(mon, period_key)
        if not sel.empty:
            months_keep = set(sel["month"].unique())
            span = gd_df[gd_df["date"].dt.to_period("M").dt.to_timestamp().isin(months_keep)].copy()
        else:
            span = gd_df.copy()

        c1, c2 = st.columns([1.2, 1.8])
        with c1:
            st.plotly_chart(chart_by_source(span, "Avg rating by source (Employees)"), use_container_width=True)
        with c2:
            st.plotly_chart(chart_monthly(sel, f"Employee sentiment — {period_key}"), use_container_width=True)

        st.markdown("### Verbatim (Glassdoor)")
        st.dataframe(verbatim_table(span), use_container_width=True, height=320)

        st.download_button(
            "Download Glassdoor CSV",
            data=gd_df.to_csv(index=False),
            file_name="glassdoor.csv",
            mime="text/csv",
            use_container_width=True,
            disabled=gd_df.empty
        )
