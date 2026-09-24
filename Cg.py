
import math
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="India Market Intelligence Engine", page_icon="📊", layout="wide")

# -----------------------------
# Model configuration
# -----------------------------
GLOBAL_WEIGHTS = {
    "Verified information": 0.10,
    "Other news": 0.10,
    "Money flow": 0.15,
    "Indian economy": 0.10,
    "Global economy": 0.10,
    "RBI / liquidity": 0.08,
    "Commodities": 0.07,
    "Geopolitics": 0.07,
    "Policy & political events": 0.05,
    "Currency": 0.05,
    "Corporate fundamentals": 0.05,
    "Natural disasters / climate": 0.04,
    "Technical / derivatives": 0.04,
}

STOCK_WEIGHTS = {
    "News": .20, "Money flow": .15, "Market regime": .10, "Sector regime": .10,
    "Fundamentals": .08, "Earnings": .07, "Futures OI": .08, "Options positioning": .07,
    "Price / volume": .05, "Technical structure": .05, "Global exposure": .02,
    "Commodity / currency exposure": .03,
}

RANGES = [
    (80, 100, "Strong bullish"),
    (70, 79.999, "Bullish"),
    (61, 69.999, "Mild bullish"),
    (50, 60.999, "Neutral / transition"),
    (40, 49.999, "Mild bearish"),
    (20, 39.999, "Bearish"),
    (0, 19.999, "Strong bearish"),
]

def band(score):
    for lo, hi, name in RANGES:
        if lo <= score <= hi:
            return name
    return "Neutral / transition"

def clamp(x, lo=0, hi=100):
    return max(lo, min(hi, float(x)))

def weighted_score(values, weights):
    return sum(clamp(values[k]) * weights[k] for k in weights)

def confidence(data_completeness, freshness, sample_size, agreement, ambiguity, regime_stability):
    # Confidence is intentionally separate from directional score.
    c = (
        .25 * data_completeness +
        .20 * freshness +
        .15 * sample_size +
        .20 * agreement +
        .10 * (100 - ambiguity) +
        .10 * regime_stability
    )
    return round(clamp(c), 1)

# -----------------------------
# Demo universe / adapters
# Replace these adapters with licensed/live feeds.
# -----------------------------
stocks = [
    "RELIANCE","HDFCBANK","ICICIBANK","SBIN","INFY","TCS","BHARTIARTL","LT",
    "ITC","AXISBANK","KOTAKBANK","M&M","MARUTI","SUNPHARMA","NTPC","POWERGRID",
    "TATASTEEL","ADANIPORTS","BAJFINANCE","HINDALCO","ONGC","COALINDIA",
    "HINDPETRO","BPCL","TATAMOTORS","BEL","HAL","TRENT","DLF","INDUSINDBK"
]

@st.cache_data(ttl=60)
def demo_market_data():
    rng = np.random.default_rng(42)
    rows = []
    for s in stocks:
        base = rng.uniform(35, 85)
        bull = clamp(base + rng.normal(0, 9))
        bear = clamp(100 - base + rng.normal(0, 9))
        rows.append({
            "Symbol": s,
            "Bullish evidence": round(bull,1),
            "Bearish evidence": round(bear,1),
            "News": round(rng.uniform(35,95),1),
            "Money flow": round(rng.uniform(30,90),1),
            "Market regime": round(rng.uniform(40,85),1),
            "Sector regime": round(rng.uniform(40,90),1),
            "Fundamentals": round(rng.uniform(40,90),1),
            "Earnings": round(rng.uniform(40,90),1),
            "Futures OI": round(rng.uniform(30,90),1),
            "Options positioning": round(rng.uniform(30,90),1),
            "Price / volume": round(rng.uniform(35,90),1),
            "Technical structure": round(rng.uniform(35,90),1),
            "Global exposure": round(rng.uniform(35,80),1),
            "Commodity / currency exposure": round(rng.uniform(35,80),1),
            "OI": int(rng.integers(100000, 9000000)),
            "OI change %": round(rng.normal(0, 5),2),
            "IV": round(rng.uniform(12,48),2),
            "PCR": round(rng.uniform(.55,1.45),2),
            "Volume": int(rng.integers(100000, 8000000)),
        })
    return pd.DataFrame(rows)

def stock_score(row):
    vals = {k: row[k] for k in STOCK_WEIGHTS}
    return round(weighted_score(vals, STOCK_WEIGHTS), 1)

def futures_score(row):
    # Independent bullish/bearish evidence; not 100 - the other side.
    oi = row["OI change %"]
    price_signal = row["Price / volume"]
    oi_signal = clamp(50 + oi * 4)
    return round(.42*row["Market regime"] + .24*row["Futures OI"] +
                 .18*price_signal + .16*row["News"], 1)

def options_score(row):
    pcr = row["PCR"]
    pcr_signal = clamp(50 + (pcr-1)*35)
    iv_signal = clamp(70 - abs(row["IV"]-22)*1.3)
    return round(.45*row["Options positioning"] + .30*pcr_signal + .25*iv_signal, 1)

df = demo_market_data()
df["Stock score"] = df.apply(stock_score, axis=1)
df["Futures score"] = df.apply(futures_score, axis=1)
df["Options score"] = df.apply(options_score, axis=1)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("Market Intelligence")
st.sidebar.caption("Quantitative decision-support dashboard — not a guaranteed forecast.")
horizon = st.sidebar.selectbox("Horizon", ["Short: 1–10 trading days", "Medium: 1–6 months", "Long: 1–5 years"])
refresh = st.sidebar.button("Refresh model")
if refresh:
    st.cache_data.clear()
    st.rerun()

# -----------------------------
# Header
# -----------------------------
st.title("🇮🇳 India Market Intelligence Engine")
st.caption(f"Model snapshot: {datetime.now().strftime('%d %b %Y, %H:%M:%S')} • {horizon}")

tabs = st.tabs([
    "Dashboard","Top 20 Stocks","Futures & Options","News Intelligence",
    "Money Flow","Macro & RBI","Sectors","Historical Patterns",
    "Scenarios","Model Audit"
])

# -----------------------------
# Dashboard
# -----------------------------
with tabs[0]:
    st.subheader("Composite market score")
    components = {
        "Verified information": 72, "Other news": 61, "Money flow": 58,
        "Indian economy": 68, "Global economy": 55, "RBI / liquidity": 64,
        "Commodities": 44, "Geopolitics": 48, "Policy & political events": 60,
        "Currency": 52, "Corporate fundamentals": 73,
        "Natural disasters / climate": 57, "Technical / derivatives": 49
    }
    score = round(sum(components[k]*GLOBAL_WEIGHTS[k] for k in GLOBAL_WEIGHTS), 1)
    conf = confidence(91, 88, 76, 72, 28, 79)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Market score", score)
    c2.metric("Regime", band(score))
    c3.metric("Confidence", f"{conf}%")
    c4.metric("Signals", "13 / 13")

    st.progress(score/100)
    st.caption("Score bands: 80–100 strong bullish • 70–79 bullish • 61–69 mild bullish • 50–60 neutral/transition • 40–49 mild bearish • 20–39 bearish • 0–19 strong bearish")

    comp = pd.DataFrame({"Component": list(components), "Score": list(components.values()),
                         "Weight %": [GLOBAL_WEIGHTS[k]*100 for k in components]})
    comp["Weighted contribution"] = (comp["Score"]*comp["Weight %"]/100).round(2)
    st.dataframe(comp.sort_values("Weighted contribution", ascending=False), use_container_width=True, hide_index=True)

    st.subheader("Why the score changed")
    st.info("Example explainability layer: commodity pressure and weak external flow reduce the composite score, while corporate fundamentals and verified information add support. In production, each line is generated from timestamped source evidence.")

# -----------------------------
# Top 20
# -----------------------------
with tabs[1]:
    st.subheader("Daily stock opportunity scan")
    st.caption("The production version should scan the complete eligible F&O universe, not just today's gainers/losers.")
    bullish = df.sort_values(["Stock score","Bullish evidence"], ascending=False).head(20).copy()
    bearish = df.sort_values(["Bearish evidence","Stock score"], ascending=[False, True]).head(20).copy()

    left,right = st.columns(2)
    with left:
        st.markdown("### Top 20 bullish setups")
        st.dataframe(bullish[["Symbol","Stock score","Bullish evidence","News","Money flow","Futures OI","Options positioning"]], use_container_width=True, hide_index=True)
    with right:
        st.markdown("### Top 20 bearish setups")
        st.dataframe(bearish[["Symbol","Stock score","Bearish evidence","News","Money flow","Futures OI","Options positioning"]], use_container_width=True, hide_index=True)

    st.markdown("### Confirmed / early / reversal")
    a,b,c = st.columns(3)
    with a: st.dataframe(df.sort_values("Stock score",ascending=False).head(20)[["Symbol","Stock score"]], use_container_width=True, hide_index=True)
    with b: st.dataframe(df.sort_values("Bullish evidence",ascending=False).head(20)[["Symbol","Bullish evidence"]], use_container_width=True, hide_index=True)
    with c:
        reversal = df.assign(Reversal=(df["Bullish evidence"]-df["Bearish evidence"]).abs()).sort_values("Reversal").head(20)
        st.dataframe(reversal[["Symbol","Bullish evidence","Bearish evidence"]], use_container_width=True, hide_index=True)

# -----------------------------
# F&O
# -----------------------------
with tabs[2]:
    st.subheader("Futures & Options Intelligence")
    f1,f2 = st.columns(2)
    with f1:
        st.markdown("### Top 20 bullish futures setups")
        x=df.sort_values("Futures score",ascending=False).head(20)
        st.dataframe(x[["Symbol","Futures score","OI","OI change %","Stock score"]], use_container_width=True, hide_index=True)
    with f2:
        st.markdown("### Top 20 bearish futures setups")
        y=df.sort_values("Futures score",ascending=True).head(20)
        st.dataframe(y[["Symbol","Futures score","OI","OI change %","Stock score"]], use_container_width=True, hide_index=True)

    st.markdown("### Options intelligence")
    st.dataframe(df.sort_values("Options score",ascending=False).head(20)[
        ["Symbol","Options score","PCR","IV","OI","Volume"]], use_container_width=True, hide_index=True)

    st.info("The live engine should combine spot/futures price, OI and OI change, call/put OI and changes, PCR, IV, strike concentration, volume, expiry, news, money flow, sector regime and market regime. High call OI or high put OI is not treated as automatically bearish or bullish.")

# -----------------------------
# News
# -----------------------------
with tabs[3]:
    st.subheader("News Intelligence")
    st.markdown("### Authentication hierarchy")
    st.write("1. Primary official confirmation → 2. Multiple independent confirmations → 3. Single credible source → 4. Unverified social/community signal.")
    news = pd.DataFrame([
        ["RBI / regulator / government / company filing", "Verified material information", 96, "High", "Timestamped primary evidence"],
        ["Two independent high-quality reports", "Confirmed news", 82, "High", "Cross-source confirmation"],
        ["Single credible financial-news report", "Relevant news", 68, "Medium", "Needs corroboration"],
        ["Social/community alert", "Early signal", 25, "Low", "Never auto-verified"],
    ], columns=["Source class","Type","Authentication","Reliability","Handling"])
    st.dataframe(news, use_container_width=True, hide_index=True)

    st.markdown("### Event transmission chain")
    st.code("Event → supply/demand → commodity/currency → inflation → RBI/liquidity → earnings/margins → valuation → index/sector/stock")
    st.caption("The production news engine should also estimate surprise, priced-in level, magnitude, duration and exposed sectors/companies.")

# -----------------------------
# Money flow / macro
# -----------------------------
with tabs[4]:
    st.subheader("Money Flow")
    flow = pd.DataFrame({
        "Flow": ["FII/FPI", "DII", "Mutual funds", "ETF / passive", "Proprietary", "Retail"],
        "1D": [-1.2, 1.5, .4, .2, -.3, .1],
        "5D": [-2.1, 2.8, 1.1, .7, -.6, .2],
        "20D": [-4.8, 5.6, 2.4, 1.5, -.9, .5]
    })
    st.dataframe(flow, use_container_width=True, hide_index=True)
    st.info("Replace demo values with exchange/official flow data in production.")

with tabs[5]:
    st.subheader("Macro + RBI / liquidity")
    macro = pd.DataFrame([
        ["Inflation", 62, "Consumer prices / expectations"],
        ["Growth", 70, "Activity / production / credit"],
        ["Rates", 58, "Policy-rate and curve conditions"],
        ["Liquidity", 64, "System liquidity / money-market conditions"],
        ["Credit", 69, "Bank and market credit conditions"],
        ["Fiscal", 61, "Budget / deficit / spending impulse"],
        ["Global yields", 48, "External rate pressure"],
    ], columns=["Factor","Score","Channel"])
    st.dataframe(macro, use_container_width=True, hide_index=True)

# -----------------------------
# Sectors
# -----------------------------
with tabs[6]:
    sector_names = ["Banks","IT","Energy","Auto","Pharma","Metals","Telecom","Capital Goods","FMCG","Realty"]
    rng = np.random.default_rng(7)
    sec = pd.DataFrame({"Sector":sector_names, "Score":rng.integers(38,86,len(sector_names)),
                        "Confidence":rng.integers(55,94,len(sector_names))})
    sec["Regime"] = sec["Score"].map(band)
    st.dataframe(sec.sort_values("Score",ascending=False), use_container_width=True, hide_index=True)

# -----------------------------
# Historical
# -----------------------------
with tabs[7]:
    st.subheader("Historical regime matching")
    st.write("Match the current combination of crude, USDINR, FII/DII, yields, volatility, inflation and geopolitical conditions against historical windows.")
    analogues = pd.DataFrame([
        ["Regime A", 87, "Crude ↑, INR weak, external yields ↑", "+1.8%", "+0.9%", "+3.4%"],
        ["Regime B", 81, "FII selling + domestic support", "-0.7%", "+0.6%", "+2.1%"],
        ["Regime C", 74, "Liquidity support + falling yields", "+1.2%", "+2.7%", "+5.9%"],
    ], columns=["Analogue","Similarity","Key features","1W outcome","1M outcome","3M outcome"])
    st.dataframe(analogues, use_container_width=True, hide_index=True)
    st.warning("Illustrative demo rows only. Production probabilities must come from actual historical observations and should show sample size and uncertainty.")

# -----------------------------
# Scenarios
# -----------------------------
with tabs[8]:
    st.subheader("Scenario engine")
    scenarios = pd.DataFrame([
        ["Base / continuation", 45, "Current regime persists; moderate signal agreement."],
        ["Bullish regime shift", 30, "Liquidity and earnings improve while external pressure eases."],
        ["Bearish shock", 25, "Commodity/geopolitical/currency shock overwhelms domestic support."],
    ], columns=["Scenario","Model weight*","Transmission"])
    st.dataframe(scenarios, use_container_width=True, hide_index=True)
    st.caption("*Demo weights. Production probabilities should only be shown when supported by historical/model evidence; otherwise display scenario weights without implying certainty.")

# -----------------------------
# Audit
# -----------------------------
with tabs[9]:
    st.subheader("Model / data audit")
    audit = pd.DataFrame([
        ["NSE derivatives", "Primary", "Contract, volume, value, OI", "Adapter required", "High"],
        ["Official macro / RBI", "Primary", "Rates, liquidity, macro", "Adapter required", "High"],
        ["Company filings", "Primary", "Results, announcements", "Adapter required", "High"],
        ["Reuters / major news", "Secondary", "Event detection / corroboration", "Adapter required", "Medium-high"],
        ["Specialist research", "Secondary", "Context / estimates", "Adapter required", "Medium"],
        ["Social/community", "Early signal", "Lead generation only", "Adapter required", "Low"],
    ], columns=["Source","Tier","Use","Status","Evidence priority"])
    st.dataframe(audit, use_container_width=True, hide_index=True)
    st.markdown("### Production safeguards")
    st.write("- Never label rumors as verified.  - Keep score and confidence separate.  - Keep bullish and bearish evidence independent.  - Store source, timestamp and evidence for every input.  - Detect contradictory signals.  - Show historical sample size for analogues.  - Do not claim guaranteed future prices.")

st.divider()
st.caption("Architecture: data adapters → normalization → event transmission graph → component models → score + confidence → historical/scenario engines → ranked dashboards. Demo data is synthetic until live licensed/official feeds are connected.")
