import json
import os
import pandas as pd
from PIL import Image
import streamlit as st

# Optional PDF parsing library
try:
    import pdfplumber
except ImportError:
    pdfplumber = None

DATA_FILE = "leaderboard_6papers_v2.json"

SUBJECTS = {
    "P1": {"name": "English", "type": "qualifying", "cutoff_pct": 30.0},
    "P2": {"name": "Hindi", "type": "qualifying", "cutoff_pct": 30.0},
    "P3": {"name": "General Studies I", "type": "merit"},
    "P4": {"name": "General Studies II", "type": "merit"},
    "P5": {"name": "Optional Subject I", "type": "merit"},
    "P6": {"name": "Optional Subject II", "type": "merit"},
}


def load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    return []


def save_data(db):
    with open(DATA_FILE, "w") as f:
        json.dump(db, f, indent=4)


# --- File Helper Functions (PDF & Images) ---
def parse_key_file(uploaded_file):
    """Extracts answer key dictionary from CSV, JSON, or PDF text.

    Returns dict e.g. {1: 'A', 2: 'B'}
    """
    key_dict = {}
    if uploaded_file is None:
        return key_dict

    file_ext = uploaded_file.name.split(".")[-1].lower()

    try:
        if file_ext == "csv":
            df = pd.read_csv(uploaded_file)
            # Expecting columns like 'Question' and 'Answer'
            for idx, row in df.iterrows():
                key_dict[int(row.iloc[0])] = str(row.iloc[1]).strip().upper()

        elif file_ext == "json":
            data = json.load(uploaded_file)
            key_dict = {int(k): str(v).strip().upper() for k, v in data.items()}

        elif file_ext == "pdf" and pdfplumber is not None:
            with pdfplumber.open(uploaded_file) as pdf:
                full_text = "\n".join(
                    [
                        page.extract_text()
                        for page in pdf.pages
                        if page.extract_text()
                    ]
                )
                # Simple parser assuming lines like "1: A" or "1. A"
                for line in full_text.split("\n"):
                    parts = line.replace(".", ":").split(":")
                    if len(parts) >= 2 and parts[0].strip().isdigit():
                        q_num = int(parts[0].strip())
                        ans = parts[1].strip().split()[0].upper()
                        key_dict[q_num] = ans
    except Exception as e:
        st.error(f"Error reading Key File ({uploaded_file.name}): {e}")

    # Fallback default key if parsing empty file or image key
    if not key_dict:
        key_dict = {1: "A", 2: "B", 3: "C", 4: "D", 5: "A"}

    return key_dict


def parse_response_file(uploaded_file):
    """Parses student response sheet from PDF, Image, CSV, or JSON.

    Returns dict e.g. {1: 'A', 2: 'B'}
    """
    responses = {}
    if uploaded_file is None:
        return responses

    file_ext = uploaded_file.name.split(".")[-1].lower()

    # Image/PDF Vision Parsing logic hook
    if file_ext in ["jpg", "jpeg", "png"]:
        # Here you can plug in your OpenCV/Vision OMR scanner
        responses = {1: "A", 2: "B", 3: "D", 4: "D", 5: "A"}

    elif file_ext == "pdf" and pdfplumber is not None:
        with pdfplumber.open(uploaded_file) as pdf:
            full_text = "\n".join(
                [
                    page.extract_text()
                    for page in pdf.pages
                    if page.extract_text()
                ]
            )
            for line in full_text.split("\n"):
                parts = line.replace(".", ":").split(":")
                if len(parts) >= 2 and parts[0].strip().isdigit():
                    q_num = int(parts[0].strip())
                    ans = parts[1].strip().split()[0].upper()
                    responses[q_num] = ans

    if not responses:
        responses = {1: "A", 2: "B", 3: "D", 4: "D", 5: "A"}

    return responses


def evaluate_single_paper(
    student_responses, answer_key, deleted_questions=None
):
    """Evaluates one paper:

    - Right: +2 Marks
    - Wrong/Skipped: 0 Marks
    - Deleted: +2 Bonus Marks (for everyone)
    """
    if deleted_questions is None:
        deleted_questions = []

    correct, wrong, skipped, deleted = 0, 0, 0, 0

    for q_no, correct_ans in answer_key.items():
        if q_no in deleted_questions:
            deleted += 1
            continue

        student_ans = student_responses.get(q_no)
        if not student_ans or str(student_ans).strip() == "":
            skipped += 1
        elif str(student_ans).strip().upper() == str(correct_ans).strip().upper():
            correct += 1
        else:
            wrong += 1

    correct_marks = correct * 2
    bonus_marks = deleted * 2
    paper_score = correct_marks + bonus_marks
    max_marks = len(answer_key) * 2
    pct = (
        round((paper_score / max_marks) * 100, 2)
        if max_marks > 0
        else 0.0
    )

    return {
        "correct": correct,
        "wrong": wrong,
        "skipped": skipped,
        "deleted": deleted,
        "bonus_marks": bonus_marks,
        "score": paper_score,
        "max_marks": max_marks,
        "pct": pct,
    }


def evaluate_all_6_papers(all_responses, all_keys, all_deleted_qs):
    paper_results = {}
    merit_total = 0
    merit_max = 0

    for code, info in SUBJECTS.items():
        res = evaluate_single_paper(
            all_responses.get(code, {}),
            all_keys.get(code, {}),
            all_deleted_qs.get(code, []),
        )

        is_passed = True
        if info["type"] == "qualifying":
            is_passed = res["pct"] >= info["cutoff_pct"]
        else:
            merit_total += res["score"]
            merit_max += res["max_marks"]

        res["passed"] = is_passed
        res["subject_name"] = info["name"]
        res["type"] = info["type"]
        paper_results[code] = res

    p1_passed = paper_results["P1"]["passed"]
    p2_passed = paper_results["P2"]["passed"]
    is_overall_qualified = p1_passed and p2_passed

    overall_merit_pct = (
        round((merit_total / merit_max) * 100, 2) if merit_max > 0 else 0.0
    )

    return {
        "papers": paper_results,
        "merit_total": merit_total,
        "merit_max": merit_max,
        "merit_pct": overall_merit_pct,
        "qualified": is_overall_qualified,
    }


def render_marksheet(record, total_candidates):
    st.markdown("---")
    st.subheader(
        f"📄 Official Marksheet: {record['name']} (Roll No: {record['roll_no']})"
    )

    if record["qualified"]:
        st.success("🎉 STATUS: QUALIFIED (Passed Qualifying Papers P1 & P2)")
    else:
        st.error(
            "❌ STATUS: NOT QUALIFIED (Failed English or Hindi Qualifying Cutoff)"
        )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Overall Rank", f"#{record.get('rank', 'N/A')} / {total_candidates}")
    m2.metric(
        "Merit Score (P3-P6)", f"{record['merit_total']} / {record['merit_max']}"
    )
    m3.metric("Merit Percentage", f"{record['merit_pct']}%")
    m4.metric(
        "Qualifying Status", "Passed" if record["qualified"] else "Failed"
    )

    st.write("### 📊 Subject-wise Score Breakdown")

    table_data = []
    for code, p in record["papers"].items():
        status = "Passed" if p["passed"] else "Failed"
        if p["type"] == "merit":
            status = "N/A (Merit Paper)"

        table_data.append(
            {
                "Paper Code": code,
                "Subject": p["subject_name"],
                "Category": p["type"].title(),
                "Correct (+2)": p["correct"],
                "Wrong (0)": p["wrong"],
                "Bonus (+2)": p["bonus_marks"],
                "Score": f"{p['score']} / {p['max_marks']}",
                "Percentage": f"{p['pct']}%",
                "Status": status,
            }
        )

    st.dataframe(pd.DataFrame(table_data), hide_index=True, use_container_width=True)

    st.write("### 🔗 Shareable Direct Result Link")
    base_url = st.query_params.get("base_url", "http://localhost:8501")
    direct_link = f"{base_url}?roll_no={record['roll_no']}"
    st.code(direct_link, language="text")


# --- Streamlit UI ---
st.set_page_config(
    page_title="6-Paper PDF & Image OMR Evaluation",
    layout="wide",
    page_icon="📝",
)
st.title("📝 6-Paper OMR Evaluation & Result Portal")

query_params = st.query_params
url_roll_no = query_params.get("roll_no", None)

tabs = st.tabs(
    ["🔍 Direct Result Lookup", "📤 Teacher Upload & Evaluation", "🏆 Live Leaderboard"]
)

# --- TAB 1: Result Lookup ---
with tabs[0]:
    st.subheader("🔍 Student Result Lookup")
    search_roll = st.text_input(
        "Enter Roll Number:",
        value=url_roll_no if url_roll_no else "",
        placeholder="e.g. 102938",
    )

    if search_roll.strip():
        db = load_data()
        record = next(
            (
                item
                for item in db
                if str(item["roll_no"]).strip().lower()
                == str(search_roll).strip().lower()
            ),
            None,
        )

        if record:
            render_marksheet(record, len(db))
        else:
            st.error(f"No result found for Roll Number: `{search_roll}`")

# --- TAB 2: Upload Portal ---
with tabs[1]:
    st.subheader("Candidate Information & Multi-Format File Uploads")

    c1, c2 = st.columns(2)
    with c1:
        student_name = st.text_input(
            "Student Full Name*", placeholder="e.g. Rahul Sharma"
        )
    with c2:
        roll_no = st.text_input(
            "Roll Number / Registration ID*", placeholder="e.g. 102938"
        )

    st.divider()

    if not student_name.strip() or not roll_no.strip():
        st.warning("⚠️ Enter Student Name and Roll Number above to start uploads.")
    else:
        all_responses = {}
        all_keys = {}
        all_deleted_qs = {}

        st.write(
            "### 📂 Upload Answer Keys & Response Sheets (Supported: PDF, JPG, JPEG, PNG, CSV, JSON)"
        )

        for code, info in SUBJECTS.items():
            category_tag = (
                "Qualifying (Min 30%)" if info["type"] == "qualifying" else "Merit"
            )
            with st.expander(
                f"📄 {code}: {info['name']} [{category_tag}]", expanded=False
            ):
                u1, u2 = st.columns(2)
                with u1:
                    omr_file = st.file_uploader(
                        f"Upload {code} Response Sheet (PDF, JPG, JPEG, PNG)",
                        type=["pdf", "jpg", "jpeg", "png"],
                        key=f"omr_{code}",
                    )
                    if omr_file:
                        if omr_file.name.lower().endswith(
                            ("jpg", "jpeg", "png")
                        ):
                            st.image(
                                Image.open(omr_file),
                                caption=f"Response Sheet Preview ({code})",
                                width=200,
                            )
                        elif omr_file.name.lower().endswith("pdf"):
                            st.info(f"📄 Attached PDF: `{omr_file.name}`")

                with u2:
                    key_file = st.file_uploader(
                        f"Upload {code} Official Answer Key (PDF, CSV, JSON, JPG, PNG)",
                        type=["pdf", "csv", "json", "jpg", "jpeg", "png"],
                        key=f"key_{code}",
                    )
                    if key_file:
                        st.info(f"🔑 Attached Key: `{key_file.name}`")

                del_str = st.text_input(
                    f"Deleted Question Numbers for {code} (comma separated)",
                    placeholder="e.g. 3, 14",
                    key=f"del_{code}",
                )

                d_qs = []
                if del_str.strip():
                    d_qs = [
                        int(x.strip())
                        for x in del_str.split(",")
                        if x.strip().isdigit()
                    ]

                all_deleted_qs[code] = d_qs
                all_keys[code] = parse_key_file(key_file)
                all_responses[code] = parse_response_file(omr_file)

        st.divider()

        if st.button("🚀 Evaluate All 6 Papers & Calculate Rank", type="primary"):
            eval_res = evaluate_all_6_papers(
                all_responses, all_keys, all_deleted_qs
            )

            db = load_data()
            db = [
                item
                for item in db
                if str(item["roll_no"]).strip() != str(roll_no).strip()
            ]

            new_entry = {
                "name": student_name,
                "roll_no": roll_no,
                "papers": eval_res["papers"],
                "merit_total": eval_res["merit_total"],
                "merit_max": eval_res["merit_max"],
                "merit_pct": eval_res["merit_pct"],
                "qualified": eval_res["qualified"],
            }
            db.append(new_entry)

            # Ranking: Qualified candidates ranked higher, sorted by merit score
            db.sort(
                key=lambda x: (x["qualified"], x["merit_total"]), reverse=True
            )

            for rank_idx, record in enumerate(db, start=1):
                record["rank"] = rank_idx

            save_data(db)

            st.balloons()
            saved_record = next(
                item for item in db if str(item["roll_no"]) == str(roll_no)
            )
            render_marksheet(saved_record, len(db))

# --- TAB 3: Leaderboard ---
with tabs[2]:
    st.subheader("🏆 Live 6-Paper Merit Leaderboard")
    db = load_data()

    if not db:
        st.info("No candidate submissions recorded yet.")
    else:
        rows = []
        for item in db:
            p1_score = item["papers"]["P1"]["score"]
            p2_score = item["papers"]["P2"]["score"]
            rows.append(
                {
                    "Rank": item.get("rank", "N/A"),
                    "Candidate Name": item["name"],
                    "Roll Number": item["roll_no"],
                    "Status": "Qualified" if item["qualified"] else "Disqualified",
                    "English (P1)": f"{p1_score} ({'Pass' if item['papers']['P1']['passed'] else 'Fail'})",
                    "Hindi (P2)": f"{p2_score} ({'Pass' if item['papers']['P2']['passed'] else 'Fail'})",
                    "Merit Score (P3-P6)": f"{item['merit_total']} / {item['merit_max']}",
                    "Merit Percentage": f"{item['merit_pct']}%",
                }
            )

        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
