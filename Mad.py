import json
import os
import pandas as pd
import streamlit as st

DATA_FILE = "leaderboard.json"


def load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    return []


def save_data(db):
    with open(DATA_FILE, "w") as f:
        json.dump(db, f, indent=4)


def evaluate_student_responses(
    student_responses, answer_key, deleted_questions=None
):
    """Marking Rules:

    - Right Question: +2 Marks
    - Wrong/Skipped Question: 0 Marks
    - Deleted Question: +2 Bonus Marks (Awarded to all)
    """
    if deleted_questions is None:
        deleted_questions = []

    correct_count = 0
    wrong_count = 0
    skipped_count = 0
    deleted_count = 0

    for q_no, correct_ans in answer_key.items():
        if q_no in deleted_questions:
            deleted_count += 1
            continue

        student_ans = student_responses.get(q_no)
        if not student_ans or str(student_ans).strip() == "":
            skipped_count += 1
        elif str(student_ans).strip().upper() == str(correct_ans).strip().upper():
            correct_count += 1
        else:
            wrong_count += 1

    # Scoring Calculation:
    # Right questions (+2) + Deleted questions bonus (+2 each)
    correct_marks = correct_count * 2
    bonus_marks = deleted_count * 2
    merit_total = correct_marks + bonus_marks

    total_questions = len(answer_key)
    max_possible_marks = total_questions * 2
    pct = (
        round((merit_total / max_possible_marks) * 100, 2)
        if max_possible_marks > 0
        else 0.0
    )

    return {
        "correct": correct_count,
        "wrong": wrong_count,
        "skipped": skipped_count,
        "deleted": deleted_count,
        "bonus_marks": bonus_marks,
        "merit_total": merit_total,
        "max_marks": max_possible_marks,
        "merit_pct": pct,
    }


# UI Setup
st.set_page_config(
    page_title="OMR Evaluation & Leaderboard", layout="wide", page_icon="📝"
)
st.title("📝 Student OMR Evaluation & Live Leaderboard")

tabs = st.tabs(["📤 Student Portal", "🏆 Live Leaderboard"])

with tabs[0]:
    st.subheader("Candidate Information & Response Upload")

    # Ask for Student Name & Roll Number First
    col_a, col_b = st.columns(2)
    with col_a:
        student_name = st.text_input(
            "Student Full Name*", placeholder="e.g. Rahul Sharma"
        )
    with col_b:
        roll_no = st.text_input(
            "Roll Number / Registration ID*", placeholder="e.g. 102938"
        )

    st.divider()

    if not student_name.strip() or not roll_no.strip():
        st.warning(
            "⚠️ Please enter your Name and Roll Number above to unlock the response sheet uploader."
        )
    else:
        st.write("### Upload Documents")
        col_up1, col_up2 = st.columns(2)

        with col_up1:
            uploaded_omr = st.file_uploader(
                "Upload Student OMR / Response Sheet (JPG, JPEG, PNG)",
                type=["jpg", "jpeg", "png"],
            )

        with col_up2:
            uploaded_key = st.file_uploader(
                "Upload Official Answer Key (CSV / JSON)", type=["csv", "json"]
            )

        # Input for deleted questions
        deleted_qs_input = st.text_input(
            "Deleted Question Numbers (comma-separated, awarded +2 bonus marks each)",
            placeholder="e.g. 5, 8",
        )
        deleted_questions = []
        if deleted_qs_input:
            try:
                deleted_questions = [
                    int(x.strip())
                    for x in deleted_qs_input.split(",")
                    if x.strip().isdigit()
                ]
            except ValueError:
                st.error("Please enter valid comma-separated question numbers.")

        if st.button("🚀 Submit & Calculate Score", type="primary"):
            if uploaded_omr is None:
                st.error("Please upload your OMR response sheet/image.")
            elif uploaded_key is None:
                st.error("Please upload the official answer key.")
            else:
                # Sample keys (Replace with actual file processing)
                sample_answer_key = {
                    1: "A",
                    2: "B",
                    3: "C",
                    4: "D",
                    5: "A",
                    6: "B",
                    7: "C",
                    8: "D",
                }
                sample_student_responses = {
                    1: "A",
                    2: "B",
                    3: "D",
                    4: "D",
                    5: "A",
                    6: "C",
                    7: "C",
                }

                # Evaluate Scores
                scores = evaluate_student_responses(
                    sample_student_responses,
                    sample_answer_key,
                    deleted_questions,
                )

                # Update Data Store
                db = load_data()
                db = [item for item in db if str(item["roll_no"]) != str(roll_no)]

                new_entry = {
                    "name": student_name,
                    "roll_no": roll_no,
                    "merit_total": scores["merit_total"],
                    "pct": scores["merit_pct"],
                    "correct": scores["correct"],
                    "wrong": scores["wrong"],
                    "bonus_marks": scores["bonus_marks"],
                    "max_marks": scores["max_marks"],
                }
                db.append(new_entry)

                # Compute Ranks
                db.sort(key=lambda x: x["merit_total"], reverse=True)
                student_rank = 1
                for rank_idx, record in enumerate(db, start=1):
                    record["rank"] = rank_idx
                    if str(record["roll_no"]) == str(roll_no):
                        student_rank = rank_idx

                save_data(db)

                # Display Results
                st.success(
                    f"Evaluation Complete! Final Rank: #{student_rank} out of {len(db)} candidates."
                )

                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Correct Answers", f"{scores['correct']} (+{scores['correct']*2})")
                m2.metric("Wrong Answers", f"{scores['wrong']} (0)")
                m3.metric(
                    "Deleted Questions Bonus",
                    f"{scores['deleted']} (+{scores['bonus_marks']})",
                )
                m4.metric(
                    "Total Merit Score",
                    f"{scores['merit_total']} / {scores['max_marks']}",
                )
                m5.metric("Percentage", f"{scores['merit_pct']}%")

                if scores["deleted"] > 0:
                    st.info(
                        f"ℹ️ {scores['deleted']} question(s) were deleted. +{scores['bonus_marks']} bonus marks (+2 per deleted question) have been added to your score."
                    )

                st.image(
                    uploaded_omr,
                    caption=f"Uploaded OMR Sheet - {student_name} ({roll_no})",
                    width=400,
                )

with tabs[1]:
    st.subheader("🏆 Live Candidate Ranking & Leaderboard")
    db = load_data()

    if not db:
        st.info("No candidate submissions recorded yet.")
    else:
        df_rank = pd.DataFrame(db)[
            [
                "rank",
                "name",
                "roll_no",
                "correct",
                "bonus_marks",
                "merit_total",
                "pct",
            ]
        ].rename(
            columns={
                "rank": "Rank",
                "name": "Candidate Name",
                "roll_no": "Roll Number",
                "correct": "Correct Qs",
                "bonus_marks": "Bonus Marks (Deleted Qs)",
                "merit_total": "Total Score",
                "pct": "Percentage (%)",
            }
        )
        st.dataframe(df_rank, hide_index=True, use_container_width=True)
