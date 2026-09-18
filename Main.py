import pandas as pd
import streamlit as st

# Configure mobile viewport
st.set_page_config(
    page_title="Check Marks",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="collapsed",
)


def load_data():
    # Replace this with your actual database loading function/file
    try:
        return pd.read_json("leaderboard.json").to_dict(orient="records")
    except Exception:
        return []


# Load stored evaluation data
db = load_data()

st.title("🎓 Marks Lookup Portal")

# 1. Read 'roll_no' from URL query parameters (e.g., ?roll_no=12345)
query_params = st.query_params
url_roll_no = query_params.get("roll_no", "")

# 2. Input Box pre-filled if 'roll_no' is present in link
roll_no_input = st.text_input(
    "Enter Roll Number:", value=url_roll_no, placeholder="e.g. 1001"
)

if roll_no_input:
    # Match candidate record from database
    candidate = next(
        (item for item in db if str(item["roll_no"]) == str(roll_no_input)),
        None,
    )

    if candidate:
        scores = candidate["scores"]

        st.success(f"Candidate: **{candidate['name']}**")

        # Compact 2-column mobile layout
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "English (P1)",
                f"{scores.get('p1', 0)}/100",
                "Passed" if scores.get("p1", 0) >= 30 else "Failed",
            )
            st.metric("Merit Score", f"{scores.get('merit_total', 0)}/400")
        with col2:
            st.metric(
                "Hindi (P2)",
                f"{scores.get('p2', 0)}/100",
                "Passed" if scores.get("p2", 0) >= 30 else "Failed",
            )
            st.metric("Overall Rank", f"#{candidate.get('rank', 'N/A')}")

        # Direct Link Generator for Candidate
        st.divider()
        share_url = f"https://your-app.streamlit.app/?roll_no={roll_no_input}"
        st.caption("Direct Link for this Marksheet:")
        st.code(share_url, language="text")

    else:
        st.error("No record found for this Roll Number.")
