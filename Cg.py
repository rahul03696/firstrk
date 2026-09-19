import io
import json
import os
import re
import secrets
from difflib import SequenceMatcher
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

# ============================================================
# OPTIONAL IMPORTS
# ============================================================

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    import pytesseract
except ImportError:
    pytesseract = None

try:
    from pdf2image import convert_from_bytes
except ImportError:
    convert_from_bytes = None

# PyMuPDF is the primary PDF renderer for Streamlit Cloud.
# It does not require a separate Poppler installation.
try:
    import fitz
except ImportError:
    fitz = None


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = "bpsc_leaderboard_6papers.json"

OPTION_MAP = {
    0: "A",
    1: "B",
    2: "C",
    3: "D"
}


# ============================================================
# OFFICIAL BPSC ANSWER KEYS
# ============================================================

OFFICIAL_KEYS = {

    # --------------------------------------------------------
    # P1 - GENERAL ENGLISH
    # --------------------------------------------------------

    "P1": {

        "Set-I (Set-1)": {
            1:"C",2:"C",3:"Deleted",4:"B",5:"A",
            6:"B",7:"D",8:"A",9:"B",10:"B",
            11:"D",12:"C",13:"C",14:"B",15:"B",
            16:"D",17:"C",18:"C",19:"B",20:"B",
            21:"A",22:"B",23:"D",24:"A",25:"A",
            26:"C",27:"B",28:"C",29:"B",30:"A",
            31:"A",32:"C",33:"D",34:"B",35:"C",
            36:"D",37:"B",38:"D",39:"B",40:"A",
            41:"A",42:"B",43:"B",44:"C",45:"C",
            46:"A",47:"A",48:"A",49:"C",50:"B"
        },

        "Set-J (Set-2)": {
            1:"Deleted",2:"A",3:"C",4:"C",5:"B",
            6:"C",7:"A",8:"D",9:"C",10:"B",
            11:"C",12:"B",13:"B",14:"C",15:"B",
            16:"C",17:"D",18:"B",19:"C",20:"D",
            21:"A",22:"B",23:"C",24:"B",25:"B",
            26:"D",27:"D",28:"C",29:"B",30:"C",
            31:"C",32:"C",33:"A",34:"D",35:"D",
            36:"B",37:"D",38:"C",39:"A",40:"B",
            41:"C",42:"B",43:"C",44:"D",45:"D",
            46:"A",47:"A",48:"A",49:"B",50:"C"
        },

        "Set-K (Set-3)": {
            1:"A",2:"A",3:"D",4:"B",5:"C",
            6:"C",7:"A",8:"A",9:"Deleted",10:"B",
            11:"B",12:"D",13:"C",14:"B",15:"B",
            16:"C",17:"B",18:"D",19:"C",20:"C",
            21:"C",22:"C",23:"C",24:"B",25:"D",
            26:"C",27:"D",28:"A",29:"A",30:"D",
            31:"A",32:"B",33:"A",34:"D",35:"B",
            36:"A",37:"D",38:"D",39:"C",40:"C",
            41:"A",42:"A",43:"C",44:"D",45:"D",
            46:"B",47:"C",48:"A",49:"A",50:"A"
        },

        "Set-L (Set-4)": {
            1:"D",2:"A",3:"B",4:"B",5:"B",
            6:"Deleted",7:"B",8:"C",9:"A",10:"C",
            11:"C",12:"B",13:"D",14:"B",15:"D",
            16:"C",17:"B",18:"B",19:"C",20:"C",
            21:"A",22:"D",23:"D",24:"D",25:"C",
            26:"A",27:"D",28:"A",29:"B",30:"B",
            31:"D",32:"B",33:"B",34:"C",35:"A",
            36:"C",37:"D",38:"B",39:"A",40:"D",
            41:"A",42:"B",43:"B",44:"D",45:"A",
            46:"A",47:"B",48:"C",49:"A",50:"A"
        }
    },


    # --------------------------------------------------------
    # P2 - GENERAL HINDI
    # --------------------------------------------------------

    "P2": {

        "Set-A": {
            1:"A",2:"C",3:"A",4:"D",5:"B",
            6:"B",7:"D",8:"C",9:"D",10:"B",
            11:"D",12:"D",13:"D",14:"D",15:"D",
            16:"B",17:"C",18:"D",19:"A",20:"A",
            21:"D",22:"B",23:"B",24:"A",25:"D",
            26:"D",27:"B",28:"C",29:"A",30:"B",
            31:"B",32:"A",33:"A",34:"B",35:"A",
            36:"B",37:"C",38:"A",39:"D",40:"D",
            41:"B",42:"A",43:"B",44:"B",45:"A",
            46:"A",47:"C",48:"D",49:"D",50:"A"
        },

        "Set-B": {
            1:"B",2:"B",3:"D",4:"D",5:"B",
            6:"B",7:"C",8:"A",9:"B",10:"C",
            11:"D",12:"D",13:"B",14:"B",15:"D",
            16:"C",17:"D",18:"B",19:"A",20:"D",
            21:"C",22:"B",23:"D",24:"A",25:"A",
            26:"C",27:"C",28:"C",29:"C",30:"C",
            31:"C",32:"D",33:"A",34:"B",35:"B",
            36:"D",37:"C",38:"A",39:"C",40:"D",
            41:"D",42:"D",43:"B",44:"C",45:"C",
            46:"D",47:"B",48:"C",49:"C",50:"A"
        },

        "Set-C": {
            1:"D",2:"C",3:"C",4:"A",5:"A",
            6:"D",7:"A",8:"A",9:"B",10:"C",
            11:"D",12:"C",13:"D",14:"D",15:"C",
            16:"C",17:"A",18:"C",19:"A",20:"B",
            21:"A",22:"B",23:"A",24:"A",25:"D",
            26:"D",27:"A",28:"A",29:"C",30:"A",
            31:"A",32:"A",33:"C",34:"B",35:"D",
            36:"C",37:"B",38:"D",39:"C",40:"A",
            41:"D",42:"A",43:"A",44:"B",45:"D",
            46:"C",47:"C",48:"D",49:"D",50:"C"
        },

        "Set-D": {
            1:"D",2:"C",3:"A",4:"B",5:"B",
            6:"A",7:"A",8:"D",9:"D",10:"A",
            11:"D",12:"D",13:"A",14:"B",15:"B",
            16:"A",17:"B",18:"B",19:"A",20:"C",
            21:"D",22:"A",23:"D",24:"D",25:"A",
            26:"D",27:"C",28:"A",29:"D",30:"B",
            31:"D",32:"B",33:"C",34:"B",35:"D",
            36:"B",37:"B",38:"A",39:"B",40:"B",
            41:"B",42:"B",43:"C",44:"A",45:"D",
            46:"D",47:"B",48:"C",49:"A",50:"A"
        }
    },


    # --------------------------------------------------------
    # P3 - GENERAL STUDIES III
    # --------------------------------------------------------

    "P3": {

        "Set-E": {
            1:"A",2:"B",3:"D",4:"B",5:"C",
            6:"A",7:"B",8:"D",9:"D",10:"B",
            11:"A",12:"A",13:"A",14:"D",15:"A",
            16:"C",17:"A",18:"A",19:"A",20:"B",
            21:"A",22:"A",23:"D",24:"C",25:"C",
            26:"C",27:"C",28:"D",29:"C",30:"B",
            31:"A",32:"B",33:"D",34:"B",35:"A",
            36:"C",37:"C",38:"C",39:"C",40:"D",
            41:"Deleted",42:"Deleted",43:"A",44:"D",
            45:"A",46:"B",47:"C",48:"A",49:"B",50:"D"
        },

        "Set-F": {
            1:"A",2:"A",3:"C",4:"C",5:"B",
            6:"C",7:"A",8:"B",9:"B",10:"A",
            11:"A",12:"B",13:"A",14:"A",15:"C",
            16:"B",17:"Deleted",18:"B",19:"Deleted",
            20:"C",21:"C",22:"D",23:"D",24:"C",
            25:"C",26:"D",27:"C",28:"B",29:"C",
            30:"D",31:"A",32:"C",33:"D",34:"D",
            35:"B",36:"C",37:"A",38:"C",39:"C",
            40:"C",41:"D",42:"B",43:"B",44:"A",
            45:"B",46:"D",47:"C",48:"D",49:"D",50:"C"
        },

        "Set-G": {
            1:"C",2:"D",3:"D",4:"B",5:"D",
            6:"A",7:"D",8:"B",9:"B",10:"B",
            11:"A",12:"B",13:"C",14:"B",15:"A",
            16:"C",17:"B",18:"A",19:"C",20:"Deleted",
            21:"B",22:"A",23:"C",24:"B",25:"D",
            26:"C",27:"A",28:"B",29:"D",30:"B",
            31:"D",32:"C",33:"A",34:"D",35:"D",
            36:"B",37:"A",38:"Deleted",39:"D",40:"D",
            41:"B",42:"B",43:"C",44:"D",45:"B",
            46:"C",47:"A",48:"C",49:"B",50:"B"
        },

        "Set-H": {
            1:"A",2:"D",3:"D",4:"C",5:"D",
            6:"A",7:"A",8:"C",9:"B",10:"C",
            11:"A",12:"Deleted",13:"Deleted",14:"A",
            15:"B",16:"D",17:"A",18:"D",19:"C",
            20:"D",21:"A",22:"D",23:"C",24:"A",
            25:"B",26:"A",27:"D",28:"A",29:"B",
            30:"D",31:"C",32:"B",33:"D",34:"D",
            35:"C",36:"D",37:"D",38:"D",39:"C",
            40:"C",41:"B",42:"B",43:"D",44:"C",
            45:"B",46:"A",47:"B",48:"A",49:"D",50:"D"
        }
    },


    # --------------------------------------------------------
    # P4 - ENGINEERING SCIENCE IV
    # --------------------------------------------------------

    "P4": {

        "Set-A": {
            1:"A",2:"Deleted",3:"C",4:"C",5:"D",
            6:"B",7:"A",8:"C",9:"D",10:"B",
            11:"B",12:"C",13:"C",14:"B",15:"D",
            16:"B",17:"C",18:"D",19:"B",20:"A",
            21:"B",22:"C",23:"A",24:"C",25:"B",
            26:"A",27:"Deleted",28:"D",29:"A",30:"A",
            31:"A",32:"A",33:"B",34:"D",35:"C",
            36:"D",37:"D",38:"A",39:"B",40:"B",
            41:"B",42:"A",43:"Deleted",44:"B",
            45:"B",46:"B",47:"D",48:"C",49:"B",50:"C"
        },

        "Set-B": {
            1:"A",2:"D",3:"C",4:"Deleted",5:"A",
            6:"D",7:"D",8:"D",9:"D",10:"B",
            11:"B",12:"D",13:"A",14:"A",15:"B",
            16:"A",17:"D",18:"C",19:"D",20:"B",
            21:"C",22:"A",23:"Deleted",24:"D",
            25:"C",26:"D",27:"B",28:"D",29:"B",
            30:"D",31:"D",32:"A",33:"D",34:"C",
            35:"D",36:"A",37:"D",38:"B",39:"C",
            40:"C",41:"B",42:"A",43:"D",44:"B",
            45:"C",46:"B",47:"C",48:"C",49:"C",50:"Deleted"
        },

        "Set-C": {
            1:"A",2:"A",3:"C",4:"D",5:"D",
            6:"C",7:"B",8:"C",9:"C",10:"D",
            11:"A",12:"C",13:"A",14:"C",15:"A",
            16:"Deleted",17:"A",18:"A",19:"D",20:"C",
            21:"D",22:"A",23:"D",24:"B",25:"B",
            26:"A",27:"B",28:"Deleted",29:"D",30:"B",
            31:"D",32:"B",33:"A",34:"A",35:"C",
            36:"B",37:"B",38:"C",39:"D",40:"A",
            41:"A",42:"C",43:"A",44:"D",45:"C",
            46:"D",47:"A",48:"A",49:"Deleted",50:"D"
        },

        "Set-D": {
            1:"D",2:"D",3:"B",4:"C",5:"D",
            6:"Deleted",7:"A",8:"D",9:"B",10:"A",
            11:"C",12:"D",13:"D",14:"B",15:"B",
            16:"B",17:"Deleted",18:"C",19:"D",20:"B",
            21:"D",22:"A",23:"B",24:"A",25:"Deleted",
            26:"B",27:"D",28:"D",29:"C",30:"C",
            31:"C",32:"C",33:"D",34:"C",35:"D",
            36:"A",37:"D",38:"B",39:"A",40:"B",
            41:"D",42:"A",43:"D",44:"A",45:"A",
            46:"A",47:"D",48:"B",49:"B",50:"D"
        }
    },


    # --------------------------------------------------------
    # P5 - CIVIL ENGINEERING V
    # --------------------------------------------------------

    "P5": {

        # Paper-V official final answer key, Advt. No. 29/2025
        # Set-I
        "Set-I": {
            1:"B",2:"D",3:"C",4:"A",5:"B",
            6:"A",7:"A",8:"D",9:"D",10:"D",
            11:"B",12:"B",13:"C",14:"B",15:"B",
            16:"C",17:"A",18:"A",19:"D",20:"B",
            21:"B",22:"A",23:"C",24:"A",25:"A",
            26:"D",27:"C",28:"B",29:"B",30:"C",
            31:"D",32:"B",33:"B",34:"D",35:"B",
            36:"D",37:"D",38:"A",39:"B",40:"D",
            41:"A",42:"C",43:"A",44:"B",45:"A",
            46:"A",47:"D",48:"C",49:"B",50:"Deleted"
        },

        # Set-J
        "Set-J": {
            1:"B",2:"B",3:"C",4:"C",5:"D",
            6:"B",7:"A",8:"A",9:"A",10:"C",
            11:"A",12:"C",13:"D",14:"B",15:"C",
            16:"B",17:"C",18:"B",19:"A",20:"C",
            21:"D",22:"D",23:"A",24:"B",25:"Deleted",
            26:"D",27:"D",28:"C",29:"D",30:"B",
            31:"D",32:"A",33:"A",34:"D",35:"A",
            36:"C",37:"A",38:"B",39:"B",40:"C",
            41:"C",42:"C",43:"A",44:"D",45:"C",
            46:"B",47:"B",48:"D",49:"B",50:"C"
        },

        # Set-K
        "Set-K": {
            1:"B",2:"B",3:"C",4:"B",5:"C",
            6:"C",7:"B",8:"C",9:"A",10:"A",
            11:"D",12:"D",13:"A",14:"A",15:"B",
            16:"B",17:"D",18:"C",19:"B",20:"B",
            21:"C",22:"A",23:"A",24:"A",25:"C",
            26:"C",27:"B",28:"D",29:"D",30:"D",
            31:"A",32:"B",33:"D",34:"B",35:"A",
            36:"B",37:"C",38:"B",39:"B",40:"B",
            41:"C",42:"B",43:"D",44:"A",45:"Deleted",
            46:"C",47:"A",48:"A",49:"C",50:"A"
        },

        # Set-L
        "Set-L": {
            1:"A",2:"D",3:"D",4:"D",5:"A",
            6:"A",7:"B",8:"C",9:"B",10:"C",
            11:"D",12:"A",13:"B",14:"C",15:"A",
            16:"A",17:"B",18:"A",19:"B",20:"C",
            21:"B",22:"D",23:"A",24:"D",25:"A",
            26:"B",27:"D",28:"C",29:"A",30:"C",
            31:"C",32:"C",33:"D",34:"B",35:"A",
            36:"D",37:"D",38:"B",39:"A",40:"D",
            41:"C",42:"A",43:"D",44:"C",45:"C",
            46:"C",47:"A",48:"A",49:"Deleted",50:"D"
        }
    },


    # --------------------------------------------------------
    # P6 - CIVIL ENGINEERING VI
    # --------------------------------------------------------

    "P6": {

        "Set-E": {
            1:"D",2:"A",3:"B",4:"A",5:"Deleted",
            6:"Deleted",7:"B",8:"Deleted",9:"A",10:"Deleted",
            11:"D",12:"A",13:"A",14:"B",15:"B",
            16:"A",17:"D",18:"A",19:"Deleted",20:"D",
            21:"C",22:"Deleted",23:"D",24:"D",25:"D",
            26:"D",27:"D",28:"C",29:"A",30:"B",
            31:"D",32:"C",33:"C",34:"A",35:"D",
            36:"B",37:"A",38:"A",39:"C",40:"C",
            41:"D",42:"A",43:"B",44:"B",45:"A",
            46:"A",47:"A",48:"A",49:"C",50:"Deleted"
        },

        "Set-F": {
            1:"Deleted",2:"B",3:"B",4:"C",5:"B",
            6:"B",7:"Deleted",8:"B",9:"C",10:"A",
            11:"D",12:"C",13:"B",14:"C",15:"B",
            16:"Deleted",17:"A",18:"A",19:"D",20:"B",
            21:"C",22:"D",23:"C",24:"C",25:"D",
            26:"Deleted",27:"A",28:"Deleted",29:"Deleted",
            30:"C",31:"D",32:"D",33:"B",34:"A",
            35:"C",36:"A",37:"B",38:"Deleted",39:"B",
            40:"D",41:"A",42:"B",43:"C",44:"C",
            45:"A",46:"B",47:"C",48:"B",49:"D",50:"B"
        },

        "Set-G": {
            1:"D",2:"D",3:"Deleted",4:"C",5:"D",
            6:"Deleted",7:"D",8:"D",9:"C",10:"A",
            11:"D",12:"C",13:"C",14:"D",15:"B",
            16:"B",17:"D",18:"A",19:"D",20:"D",
            21:"C",22:"A",23:"C",24:"B",25:"Deleted",
            26:"A",27:"B",28:"A",29:"C",30:"A",
            31:"A",32:"A",33:"A",34:"A",35:"Deleted",
            36:"Deleted",37:"Deleted",38:"C",39:"A",
            40:"B",41:"D",42:"B",43:"B",44:"D",
            45:"C",46:"D",47:"A",48:"D",49:"D",50:"Deleted"
        },

        "Set-H": {
            1:"D",2:"B",3:"D",4:"C",5:"C",
            6:"B",7:"A",8:"Deleted",9:"D",10:"A",
            11:"Deleted",12:"C",13:"C",14:"Deleted",15:"D",
            16:"C",17:"Deleted",18:"A",19:"B",20:"Deleted",
            21:"A",22:"B",23:"D",24:"C",25:"C",
            26:"C",27:"D",28:"D",29:"D",30:"C",
            31:"B",32:"A",33:"D",34:"D",35:"B",
            36:"Deleted",37:"Deleted",38:"C",39:"B",
            40:"D",41:"C",42:"C",43:"B",44:"B",
            45:"A",46:"C",47:"D",48:"C",49:"A",50:"C"
        }
    }
}


# ============================================================
# SUBJECT INFORMATION
# ============================================================

SUBJECT_META = {

    "P1": {
        "name": "General English",
        "type": "qualifying",
        "cutoff_pct": 30.0
    },

    "P2": {
        "name": "General Hindi",
        "type": "qualifying",
        "cutoff_pct": 30.0
    },

    "P3": {
        "name": "General Studies III",
        "type": "merit"
    },

    "P4": {
        "name": "General Engineering Science IV",
        "type": "merit"
    },

    "P5": {
        "name": "Civil Engineering V",
        "type": "merit"
    },

    "P6": {
        "name": "Civil Engineering VI",
        "type": "merit"
    }
}


# ============================================================
# MARKING ENGINE
# ============================================================

class OMREvaluationEngine:

    def __init__(self, answer_key, marking_scheme=None):

        self.answer_key = answer_key

        self.marking_scheme = marking_scheme or {
            "positive": 2.0,
            "negative": 0.0,
            "bonus": 2.0
        }


    def evaluate_responses(self, student_responses):

        correct = 0
        wrong = 0
        skipped = 0
        deleted = 0

        score = 0.0

        response_breakdown = []

        for q_no in range(1, 51):

            q_key = f"Q{q_no}" if f"Q{q_no}" in self.answer_key else q_no

            official_ans = self.answer_key.get(q_key, "")

            user_ans = student_responses.get(
                q_no,
                student_responses.get(str(q_no), "")
            )


            # Deleted question
            if str(official_ans).strip().lower() == "deleted":

                deleted += 1

                status = "Deleted (+2 Bonus)"

                marks_awarded = self.marking_scheme["bonus"]

                score += marks_awarded


            # Unattempted
            elif (
                not user_ans
                or str(user_ans).strip() == ""
                or str(user_ans).strip().lower() == "unattempted"
            ):

                skipped += 1

                status = "Unattempted (0 Marks)"

                marks_awarded = 0.0


            # Correct
            elif (
                str(user_ans).strip().upper()
                == str(official_ans).strip().upper()
            ):

                correct += 1

                status = "Correct (+2 Marks)"

                marks_awarded = self.marking_scheme["positive"]

                score += marks_awarded


            # Wrong
            else:

                wrong += 1

                status = "Incorrect (0 Marks)"

                marks_awarded = -self.marking_scheme["negative"]

                score += marks_awarded


            response_breakdown.append({

                "Q. No": q_no,

                "Your Answer":
                    user_ans if user_ans else "Unattempted",

                "Official Key":
                    official_ans,

                "Result":
                    status,

                "Marks":
                    marks_awarded
            })


        max_marks = (
            len(self.answer_key)
            * self.marking_scheme["positive"]
        )

        bonus_marks = (
            deleted
            * self.marking_scheme["bonus"]
        )

        pct = (
            round((score / max_marks) * 100, 2)
            if max_marks > 0
            else 0.0
        )


        return {

            "correct": correct,

            "wrong": wrong,

            "skipped": skipped,

            "deleted_count": deleted,

            "bonus_marks": bonus_marks,

            "score": round(score, 2),

            "max_marks": max_marks,

            "pct": pct,

            "itemized_breakdown":
                response_breakdown
        }


# ============================================================
# DATABASE
# ============================================================

def load_data():

    if os.path.exists(DATA_FILE):

        with open(DATA_FILE, "r") as f:

            return json.load(f)

    return []


def save_data(db):

    with open(DATA_FILE, "w") as f:

        json.dump(
            db,
            f,
            indent=4
        )


def generate_unique_public_id(db):
    """Generate a persistent random numeric ID for public display.

    The real OMR roll number remains private and is never used in the
    public leaderboard. The generated ID is stored with the record so
    it remains stable until that record is replaced.
    """
    used = {
        str(item.get("public_id", "")).strip()
        for item in db
        if item.get("public_id")
    }

    while True:
        candidate = str(secrets.randbelow(900000) + 100000)
        if candidate not in used:
            return candidate


def ensure_public_ids(db):
    """Backfill unique public IDs for older verified records."""
    used = set()
    changed = False

    for item in db:
        public_id = str(item.get("public_id", "")).strip()
        if public_id and public_id not in used:
            used.add(public_id)
        else:
            item["public_id"] = generate_unique_public_id(
                [{"public_id": value} for value in used]
            )
            used.add(item["public_id"])
            changed = True

    return changed


# ============================================================
# BPSC OMR TEMPLATE
# ============================================================

OMR_REFERENCE_WIDTH = 1191.0

OMR_REFERENCE_HEIGHT = 1684.0


# Bubble X positions.
#
# Five blocks.
# Each block contains 10 questions.
#
# Every question has A/B/C/D.

OMR_OPTION_X = [

    [194, 216, 239, 262],

    [376, 398, 421, 444],

    [558, 580, 603, 626],

    [740, 762, 785, 807],

    [921, 944, 966, 989]
]


# Ten rows.

OMR_ROW_Y = [
    513,
    547,
    581,
    615,
    649,
    683,
    716,
    750,
    784,
    818
]


# ============================================================
# BUBBLE DARKNESS DETECTION
# ============================================================

def _dark_fraction(
    gray,
    cx,
    cy,
    radius
):

    h, w = gray.shape[:2]

    cx = int(round(cx))

    cy = int(round(cy))

    radius = max(
        2,
        int(round(radius))
    )


    x1 = max(
        0,
        cx - radius
    )

    x2 = min(
        w,
        cx + radius + 1
    )

    y1 = max(
        0,
        cy - radius
    )

    y2 = min(
        h,
        cy + radius + 1
    )


    patch = gray[
        y1:y2,
        x1:x2
    ]


    if patch.size == 0:

        return 0.0


    yy, xx = np.ogrid[
        :patch.shape[0],
        :patch.shape[1]
    ]


    local_cx = cx - x1

    local_cy = cy - y1


    # Ignore outer printed bubble ring.

    r = max(
        2,
        int(radius * 0.68)
    )


    mask = (
        (xx - local_cx) ** 2
        +
        (yy - local_cy) ** 2
        <= r ** 2
    )


    values = patch[mask]


    if values.size == 0:

        return 0.0


    return float(
        np.mean(values < 120)
    )


# ============================================================
# BUBBLE SCORE WITH ALIGNMENT TOLERANCE
# ============================================================

def _best_bubble_score(
    gray,
    x,
    y,
    scale
):

    radius = max(
        5,
        int(round(8 * scale))
    )


    drift = max(
        1,
        int(round(4 * scale))
    )


    best = 0.0


    for dx in (
        -drift,
        0,
        drift
    ):

        for dy in (
            -drift,
            0,
            drift
        ):

            score = _dark_fraction(
                gray,
                x + dx,
                y + dy,
                radius
            )


            best = max(
                best,
                score
            )


    return best


# ============================================================
# MAIN OMR READER
# ============================================================

def process_image_cv_omr(
    pil_img,
    return_diagnostics=False
):

    responses = {}


    diagnostics = {

        "source":
            "BPSC Page-2 template reader",

        "confidence":
            {},

        "ambiguous":
            [],

        "unattempted":
            [],

        "detected_count":
            0,

        "quality":
            0.0
    }


    try:

        rgb = np.array(
            pil_img.convert("RGB")
        )


        gray = cv2.cvtColor(
            rgb,
            cv2.COLOR_RGB2GRAY
        )


        h, w = gray.shape[:2]


        sx = (
            w
            /
            OMR_REFERENCE_WIDTH
        )


        sy = (
            h
            /
            OMR_REFERENCE_HEIGHT
        )


        scale = min(
            sx,
            sy
        )


        aspect = (
            w
            /
            float(h)
        )


        reference_aspect = (
            OMR_REFERENCE_WIDTH
            /
            OMR_REFERENCE_HEIGHT
        )


        if abs(
            aspect
            -
            reference_aspect
        ) > 0.08:

            diagnostics["source"] = (
                "BPSC template reader "
                "(non-standard page ratio)"
            )


        # ----------------------------------------------------
        # FIVE BLOCKS
        # ----------------------------------------------------

        for block_idx in range(5):

            # ------------------------------------------------
            # TEN QUESTIONS PER BLOCK
            # ------------------------------------------------

            for row_idx in range(10):

                q_no = (
                    block_idx * 10
                    +
                    row_idx
                    +
                    1
                )


                y = (
                    OMR_ROW_Y[row_idx]
                    *
                    sy
                )


                scores = []


                # --------------------------------------------
                # A/B/C/D
                # --------------------------------------------

                for x_ref in OMR_OPTION_X[block_idx]:

                    x = (
                        x_ref
                        *
                        sx
                    )


                    score = _best_bubble_score(
                        gray,
                        x,
                        y,
                        scale
                    )


                    scores.append(score)


                # --------------------------------------------
                # FIND HIGHEST SCORE
                # --------------------------------------------

                order = np.argsort(
                    scores
                )[::-1]


                best_idx = int(
                    order[0]
                )


                best = float(
                    scores[best_idx]
                )


                second = float(
                    scores[
                        int(order[1])
                    ]
                )


                margin = (
                    best
                    -
                    second
                )


                diagnostics[
                    "confidence"
                ][q_no] = {

                    "scores":
                        [
                            round(v, 3)
                            for v in scores
                        ],

                    "best":
                        round(best, 3),

                    "margin":
                        round(margin, 3)
                }


                # --------------------------------------------
                # BLANK QUESTION
                # --------------------------------------------

                if best < 0.30:

                    diagnostics[
                        "unattempted"
                    ].append(q_no)


                # --------------------------------------------
                # MULTIPLE MARKS
                # --------------------------------------------

                elif (
                    second >= 0.30
                    and
                    (best - second) < 0.18
                ):

                    diagnostics[
                        "ambiguous"
                    ].append(q_no)


                # --------------------------------------------
                # LOW CONFIDENCE
                # --------------------------------------------

                elif margin < 0.18:

                    diagnostics[
                        "ambiguous"
                    ].append(q_no)


                # --------------------------------------------
                # VALID ANSWER
                # --------------------------------------------

                else:

                    responses[q_no] = (
                        OPTION_MAP[
                            best_idx
                        ]
                    )

                    diagnostics[
                        "detected_count"
                    ] += 1


        # ----------------------------------------------------
        # QUALITY SCORE
        # ----------------------------------------------------

        if diagnostics[
            "detected_count"
        ]:

            conf_values = [

                v["best"]

                for v
                in diagnostics[
                    "confidence"
                ].values()

                if v["best"] >= 0.30
            ]


            diagnostics[
                "quality"
            ] = round(

                float(
                    np.mean(
                        conf_values
                    )
                )
                if conf_values
                else 0.0,

                3
            )


    except Exception as exc:

        diagnostics[
            "error"
        ] = str(exc)


    if return_diagnostics:

        return (
            responses,
            diagnostics
        )


    return responses


# ============================================================
# PDF RENDERING
# ============================================================

def _render_pdf_pages(
    file_bytes
):
    """
    Render every PDF page at high resolution.

    PyMuPDF is deliberately used first because it works on Streamlit
    servers without requiring a separate Poppler system package.
    """

    # --------------------------------------------------------
    # PRIMARY: PyMuPDF
    # --------------------------------------------------------

    if fitz is not None:

        try:

            doc = fitz.open(
                stream=file_bytes,
                filetype="pdf"
            )

            pages = []

            # 300 DPI is enough for this BPSC OMR template.
            zoom = 300.0 / 72.0

            matrix = fitz.Matrix(
                zoom,
                zoom
            )

            for page in doc:

                pix = page.get_pixmap(
                    matrix=matrix,
                    alpha=False
                )

                pages.append(
                    Image.frombytes(
                        "RGB",
                        [
                            pix.width,
                            pix.height
                        ],
                        pix.samples
                    )
                )

            doc.close()

            if pages:
                return pages

        except Exception:
            pass

    # --------------------------------------------------------
    # SECONDARY: pdf2image
    # --------------------------------------------------------

    if convert_from_bytes is not None:

        try:

            return convert_from_bytes(
                file_bytes,
                dpi=300,
                fmt="png"
            )

        except Exception:
            pass

    return []


# ============================================================
# OCR FALLBACK
# ============================================================

def _extract_text_responses(
    extracted_text
):

    responses = {}


    if not extracted_text:

        return responses


    lines = [
        line.strip()
        for line
        in extracted_text.splitlines()
        if line.strip()
    ]


    # --------------------------------------------------------
    # PIPE/TABLE FORMAT
    # --------------------------------------------------------

    for i in range(
        len(lines) - 1
    ):

        q_nums = re.findall(

            r'(?:^|\||\s)'
            r'(\d{1,2})'
            r'(?=\s*\||\s*$)',

            lines[i]
        )


        answers = re.findall(

            r'(?:^|\||\s)'
            r'([A-Da-d])'
            r'(?=\s*\||\s*$)',

            lines[i + 1]
        )


        if (
            len(q_nums) >= 2
            and
            len(q_nums) == len(answers)
        ):

            for q_str, ans in zip(
                q_nums,
                answers
            ):

                q_num = int(
                    q_str
                )


                if 1 <= q_num <= 50:

                    responses[
                        q_num
                    ] = ans.upper()


    # --------------------------------------------------------
    # STANDARD FORMAT
    # --------------------------------------------------------

    if not responses:

        standard_matches = re.findall(

            r'(?:^|\b|\s)'
            r'(\d{1,2})'
            r'[\s\.\:\-\|]+'
            r'([A-Da-d])'
            r'(?:\b|\s|$)',

            extracted_text
        )


        for q_str, ans in standard_matches:

            q_num = int(
                q_str
            )


            if 1 <= q_num <= 50:

                responses[
                    q_num
                ] = ans.upper()


    return responses


# ============================================================
# OMR IDENTITY EXTRACTION
# ============================================================


def _normalize_identity_name(value):
    """Normalize OCR name text for cross-sheet comparison."""
    if not value:
        return ""

    value = str(value).upper()
    value = re.sub(r"[^A-Z0-9 ]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _normalize_roll_number(value):
    """Keep only digits so OCR punctuation/spaces do not break matching."""
    if not value:
        return ""
    return re.sub(r"\D", "", str(value))


def _name_from_targeted_ocr(image):
    """Read the handwritten candidate name from the fixed OMR name box.

    The supplied OMR has red printed labels and black handwritten text.  OCR
    works much better when the black handwriting is isolated from the red
    template before recognition.
    """
    if pytesseract is None or image is None:
        return ""

    w, h = image.size
    box = (
        int(w * 0.105),
        int(h * 0.076),
        int(w * 0.825),
        int(h * 0.097),
    )
    crop = image.crop(box).convert("RGB")
    # The application's PDF renderer uses 300 DPI.  Upsampling also helps
    # when the user uploads a lower-resolution PNG/JPEG.
    crop = crop.resize((crop.width * 2, crop.height * 2))
    rgb = np.array(crop)
    r, g, b = cv2.split(rgb)

    # Keep dark neutral/black ink and suppress the red printed template.
    ink_mask = ((r < 120) & (g < 120) & (b < 120)).astype(np.uint8) * 255
    candidates = []
    for arr in (ink_mask, 255 - ink_mask):
        for psm in (6, 7):
            try:
                text = pytesseract.image_to_string(
                    arr,
                    config=(
                        f"--psm {psm} "
                        "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz "
                    ),
                )
            except Exception:
                continue

            cleaned = _normalize_identity_name(text)
            # Remove very short OCR debris, but preserve the student's words.
            words = [w for w in cleaned.split() if len(re.sub(r"[^A-Z]", "", w)) >= 2]
            cleaned = " ".join(words)
            letters = re.sub(r"[^A-Z]", "", cleaned)
            if len(letters) >= 5:
                candidates.append(cleaned)

    if not candidates:
        return ""

    # Prefer a multi-word/long candidate.  Cross-sheet fuzzy matching below
    # handles small OCR differences such as RAHULKOUMAR vs FRAHULKUMMAR.
    candidates.sort(
        key=lambda x: (
            1 if len(x.split()) >= 2 else 0,
            len(re.sub(r"[^A-Z]", "", x)),
        ),
        reverse=True,
    )
    return candidates[0]


def _name_similarity(a, b):
    """Return a 0-1 similarity score for OCR names."""
    a = _normalize_identity_name(a).replace(" ", "")
    b = _normalize_identity_name(b).replace(" ", "")
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _extract_roll_from_bubbles(image):
    """Read the six-digit roll number from the OMR bubble grid.

    The sample sheets have a six-column x ten-row roll grid.  This deliberately
    ignores the printed 'OMR Sheet No.' because that number changes from paper
    to paper for the same candidate.
    """
    if image is None:
        return ""

    rgb = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape[:2]

    # Relative positions measured from the supplied OMR template.
    x0 = 0.1635 * w
    dx = 0.0380 * w
    y0 = 0.1735 * h
    dy = 0.0103 * h

    digits = []
    confidence = []

    for col in range(6):
        x = x0 + col * dx
        scores = []
        for row in range(10):
            y = y0 + row * dy
            cx, cy = int(round(x)), int(round(y))
            radius = max(5, int(round(0.006 * min(w, h))))
            patch = gray[
                max(0, cy - radius):min(h, cy + radius + 1),
                max(0, cx - radius):min(w, cx + radius + 1),
            ]
            if patch.size == 0:
                scores.append(0)
                continue
            # Filled bubbles contain substantially more dark pixels than the
            # outlined bubbles and their printed row numbers.
            scores.append(int(np.sum(patch < 100)))

        order = np.argsort(scores)[::-1]
        best = int(order[0])
        best_score = scores[best]
        second_score = scores[int(order[1])] if len(order) > 1 else 0
        digits.append(str(best))
        confidence.append((best_score, second_score))

    # Filled circles in this template are normally ~60-90 dark pixels in the
    # sampling patch, while empty circles are far lower.
    if not all(best >= 25 and best >= second + 15 for best, second in confidence):
        return ""

    return "".join(digits)


def _extract_name_from_ocr(text):
    """Legacy full-page fallback for templates where targeted OCR is unavailable."""
    if not text:
        return ""

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    label_pattern = re.compile(
        r"(?:CANDIDATE\s+)?NAME(?:\s*OF\s*CANDIDATE)?",
        re.I,
    )

    for i, line in enumerate(lines):
        match = label_pattern.search(line)
        if not match:
            continue

        value = line[match.end():].strip(" :-_=|\t")
        value = re.sub(r"^(?:MR|MS|MRS|MISS)\.?\s+", "", value, flags=re.I)
        value = re.sub(r"[^A-Za-z .'-]", " ", value)
        value = re.sub(r"\s+", " ", value).strip()
        if len(re.sub(r"[^A-Za-z]", "", value)) >= 3:
            return _normalize_identity_name(value)

        if i + 1 < len(lines):
            value = re.sub(r"[^A-Za-z .'-]", " ", lines[i + 1])
            value = re.sub(r"\s+", " ", value).strip()
            if len(re.sub(r"[^A-Za-z]", "", value)) >= 3:
                return _normalize_identity_name(value)

    return ""


def _extract_booklet_set(image, valid_sets=None):
    """Read the printed Question Booklet Series from the OMR sheet.

    The response sheet contains a large booklet-series letter (for example J)
    near the top-center of the page.  This is deliberately read from the OMR
    itself so a user cannot accidentally score a Set-J response sheet against
    a different answer-key set.
    """
    if image is None or pytesseract is None:
        return ""

    valid_sets = valid_sets or []
    valid_letters = {
        str(item).strip().upper()[-1]
        for item in valid_sets
        if str(item).strip()
    }
    if not valid_letters:
        valid_letters = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    rgb = image.convert("RGB")
    w, h = rgb.size

    # The supplied BPSC template places the large booklet letter in this box.
    crop = rgb.crop((
        int(0.47 * w),
        int(0.16 * h),
        int(0.66 * w),
        int(0.31 * h),
    ))
    # Upscale the large booklet letter for more reliable OCR.
    crop = crop.resize((crop.width * 3, crop.height * 3))

    variants = [crop]
    gray = cv2.cvtColor(np.array(crop), cv2.COLOR_RGB2GRAY)
    variants.append(Image.fromarray(gray))
    _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(Image.fromarray(threshold))

    candidates = []
    for variant in variants:
        for psm in (6, 10, 11, 13):
            try:
                text = pytesseract.image_to_string(
                    variant,
                    config=f"--psm {psm}",
                ).upper()
            except Exception:
                continue

            # Prefer an isolated valid set letter.
            tokens = re.findall(r"[A-Z]", text)
            for token in tokens:
                if token in valid_letters:
                    candidates.append(token)

    if not candidates:
        return ""

    # Most frequent OCR candidate wins.
    counts = Counter(candidates)
    return counts.most_common(1)[0][0]


def extract_omr_identity(uploaded_file, valid_sets=None):
    """Extract name, roll number and booklet set from one OMR file."""
    if uploaded_file is None:
        return {"name": "", "roll_no": "", "booklet_set": "", "raw_ocr": ""}

    file_bytes = uploaded_file.getvalue()
    file_ext = uploaded_file.name.rsplit(".", 1)[-1].lower()
    images = []
    extracted_text = ""

    try:
        if file_ext in ["jpg", "jpeg", "png"]:
            images = [Image.open(io.BytesIO(file_bytes)).convert("RGB")]

        elif file_ext == "pdf":
            if pdfplumber is not None:
                try:
                    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                        for page in pdf.pages:
                            page_text = page.extract_text()
                            if page_text:
                                extracted_text += page_text + "\n"
                except Exception:
                    extracted_text = ""

            images = _render_pdf_pages(file_bytes)

        if not images:
            return {"name": "", "roll_no": "", "raw_ocr": extracted_text}

        # Use all rendered pages for identity, not just the page chosen for answers.
        ocr_parts = []
        if extracted_text.strip():
            ocr_parts.append(extracted_text)

        for image in images:
            page_text = _ocr_identity_text(image)
            if page_text:
                ocr_parts.append(page_text)

        combined_text = "\n".join(ocr_parts)

        # The candidate name is handwritten, so OCR only the name box.
        # The roll number is bubble-coded; do NOT use the printed OMR Sheet No.
        # because that number can differ between papers for the same student.
        first_image = images[0]
        detected_name = _name_from_targeted_ocr(first_image)
        detected_roll = _extract_roll_from_bubbles(first_image)
        detected_booklet_set = _extract_booklet_set(
            first_image,
            valid_sets=valid_sets,
        )

        # Keep the old OCR parser only as a name fallback for alternate templates.
        if not detected_name:
            detected_name = _extract_name_from_ocr(combined_text)

        return {
            "name": detected_name,
            "roll_no": detected_roll,
            "booklet_set": detected_booklet_set,
            "raw_ocr": combined_text,
        }

    except Exception as exc:
        return {
            "name": "",
            "roll_no": "",
            "booklet_set": "",
            "raw_ocr": f"IDENTITY OCR ERROR: {exc}",
        }


# ============================================================
# MAIN FILE PARSER
# ============================================================

def parse_omr_file(
    uploaded_file
):

    if uploaded_file is None:

        return {}


    file_bytes = (
        uploaded_file.getvalue()
    )


    file_ext = (
        uploaded_file.name
        .rsplit(".", 1)[-1]
        .lower()
    )


    pil_images = []

    extracted_text = ""


    try:

        # ----------------------------------------------------
        # IMAGE FILE
        # ----------------------------------------------------

        if file_ext in [
            "jpg",
            "jpeg",
            "png"
        ]:

            pil_images = [

                Image.open(
                    io.BytesIO(
                        file_bytes
                    )
                ).convert("RGB")

            ]


            if pytesseract is not None:

                try:

                    extracted_text = (
                        pytesseract
                        .image_to_string(
                            pil_images[0]
                        )
                    )

                except Exception:

                    extracted_text = ""


        # ----------------------------------------------------
        # PDF FILE
        # ----------------------------------------------------

        elif file_ext == "pdf":

            # -----------------------------------------------
            # PDF TEXT EXTRACTION
            # -----------------------------------------------

            if pdfplumber is not None:

                try:

                    with pdfplumber.open(
                        io.BytesIO(
                            file_bytes
                        )
                    ) as pdf:

                        for page in pdf.pages:

                            text = (
                                page.extract_text()
                            )


                            if text:

                                extracted_text += (
                                    text
                                    +
                                    "\n"
                                )

                except Exception:

                    extracted_text = ""


            # -----------------------------------------------
            # RENDER PDF
            # -----------------------------------------------

            pil_images = _render_pdf_pages(
                file_bytes
            )


            if not pil_images:

                st.error(
                    f"❌ PDF rendering failed for "
                    f"`{uploaded_file.name}`. "
                    "The server needs PyMuPDF (fitz). "
                    "Check that `PyMuPDF` is present in requirements.txt."
                )

                return {}


            # -----------------------------------------------
            # OCR
            # -----------------------------------------------

            if (
                pil_images
                and
                pytesseract is not None
                and
                not extracted_text.strip()
            ):

                try:

                    extracted_text = (
                        pytesseract
                        .image_to_string(
                            pil_images[0]
                        )
                    )

                except Exception:

                    extracted_text = ""


        # ====================================================
        # PRIMARY OMR READER
        # ====================================================

        if pil_images:

            best_responses = {}
            best_diag = None

            # Test every rendered page and keep the page with
            # the highest number of confidently detected answers.
            for page_index, page_image in enumerate(pil_images):

                page_responses, page_diag = (
                    process_image_cv_omr(
                        page_image,
                        return_diagnostics=True
                    )
                )

                if (
                    best_diag is None
                    or
                    page_diag.get("detected_count", 0)
                    >
                    best_diag.get("detected_count", 0)
                ):
                    best_responses = page_responses
                    best_diag = page_diag

            if best_diag is not None:

                detected = best_diag.get(
                    "detected_count",
                    0
                )

                ambiguous = best_diag.get(
                    "ambiguous",
                    []
                )

                unattempted = best_diag.get(
                    "unattempted",
                    []
                )

                # ------------------------------------------------
                # SUCCESSFUL OMR READ
                # ------------------------------------------------

                if detected >= 40:

                    if ambiguous:

                        st.warning(
                            "⚠️ OMR reader found ambiguous "
                            "marks in "
                            +
                            ", ".join(
                                f"Q{q}"
                                for q in ambiguous
                            )
                            +
                            ". These questions were not guessed."
                        )

                    if unattempted:

                        st.info(
                            "ℹ️ Blank/unattempted questions: "
                            +
                            ", ".join(
                                f"Q{q}"
                                for q in unattempted
                            )
                        )

                    st.success(
                        f"✅ OMR vision read completed: "
                        f"{detected}/50 responses detected."
                    )

                    # Visible diagnostic table so the reader can
                    # be verified before scoring.
                    detected_rows = []

                    for q_no in range(1, 51):

                        detected_rows.append({
                            "Q": q_no,
                            "Detected Answer":
                                best_responses.get(
                                    q_no,
                                    "Unattempted"
                                ),
                            "Confidence":
                                best_diag.get(
                                    "confidence",
                                    {}
                                ).get(
                                    q_no,
                                    {}
                                ).get(
                                    "best",
                                    0.0
                                )
                        })

                    with st.expander(
                        "🔎 Verify detected OMR answers before scoring",
                        expanded=False
                    ):

                        st.dataframe(
                            pd.DataFrame(
                                detected_rows
                            ),
                            hide_index=True,
                            use_container_width=True
                        )

                    return best_responses

                # ------------------------------------------------
                # LOW DETECTION
                # ------------------------------------------------

                st.warning(
                    f"⚠️ BPSC OMR reader detected only "
                    f"{detected}/50 responses. "
                    "The score will NOT silently treat the "
                    "whole sheet as blank."
                )

                # Do not fall through to a bad legacy result
                # when the actual OMR image was successfully read.
                if detected == 0:
                    st.error(
                        "❌ No OMR bubbles were detected. "
                        "Check PDF rendering/template alignment."
                    )
                    return {}

        # ====================================================
        # OCR FALLBACK
        # ====================================================

        responses = _extract_text_responses(
            extracted_text
        )


        if responses:

            st.info(

                f"ℹ️ Used text/OCR fallback: "
                f"{len(responses)} responses extracted."
            )


            return responses


        # ====================================================
        # LEGACY FALLBACK
        # ====================================================

        if pil_images:

            responses = _legacy_contour_omr(
                pil_images[0]
            )


            if responses:

                st.info(

                    f"ℹ️ Used contour fallback: "
                    f"{len(responses)} responses extracted."
                )


            return responses


    except Exception as e:

        st.error(

            f"Error reading "
            f"{uploaded_file.name}: "
            f"{str(e)}"
        )


    return {}


# ============================================================
# GENERIC CONTOUR FALLBACK
# ============================================================

def _legacy_contour_omr(
    pil_img
):

    responses = {}


    try:

        image = np.array(
            pil_img.convert("RGB")
        )


        gray = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2GRAY
        )


        blur = cv2.GaussianBlur(
            gray,
            (5, 5),
            0
        )


        thresh = cv2.threshold(
            blur,
            0,
            255,
            cv2.THRESH_BINARY_INV
            +
            cv2.THRESH_OTSU
        )[1]


        contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )


        bubbles = []


        for c in contours:

            x, y, w, h = (
                cv2.boundingRect(c)
            )


            if (
                10 <= w <= 70
                and
                10 <= h <= 70
            ):

                ar = (
                    w
                    /
                    float(h)
                )


                if (
                    0.70 <= ar <= 1.30
                ):

                    bubbles.append(
                        (
                            x,
                            y,
                            w,
                            h
                        )
                    )


        bubbles.sort(
            key=lambda b: (
                b[1],
                b[0]
            )
        )


        for q in range(
            1,
            51
        ):

            group = bubbles[
                (q - 1) * 4:
                q * 4
            ]


            if len(group) != 4:

                continue


            scores = []


            for x, y, w, h in group:

                roi = thresh[
                    y:y+h,
                    x:x+w
                ]


                scores.append(

                    cv2.countNonZero(
                        roi
                    )
                    /
                    float(
                        max(
                            1,
                            roi.size
                        )
                    )
                )


            idx = int(
                np.argmax(scores)
            )


            if scores[idx] > 0.20:

                responses[q] = (
                    OPTION_MAP[idx]
                )


    except Exception:

        return {}


    return responses


# ============================================================
# EVALUATION
# ============================================================

def evaluate_paper(
    student_responses,
    official_key
):

    engine = OMREvaluationEngine(
        official_key
    )

    return engine.evaluate_responses(
        student_responses
    )


# ============================================================
# MARKSHEET DISPLAY
# ============================================================

def render_marksheet(
    record,
    total_candidates
):

    st.markdown("---")


    st.subheader(

        f"📄 BPSC Official Scorecard: "
        f"{record['name']} "
        f"(Roll No: {record['roll_no']})"
    )


    if record["qualified"]:

        st.success(

            "🎉 STATUS: QUALIFIED "
            "(Passed English & Hindi "
            "Qualifying Cutoff >= 30%)"
        )

    else:

        st.error(

            "❌ STATUS: DISQUALIFIED "
            "(Failed English or Hindi "
            "Qualifying Cutoff of 30%)"
        )


    m1, m2, m3, m4 = st.columns(4)


    m1.metric(

        "Overall Rank",

        f"#{record.get('rank', 'N/A')} "
        f"/ {total_candidates}"
    )


    m2.metric(

        "Merit Score (P3-P6)",

        f"{record['merit_total']} "
        f"/ {record['merit_max']}"
    )


    m3.metric(

        "Merit Percentage",

        f"{record['merit_pct']}%"
    )


    m4.metric(

        "Qualifying Status",

        "Passed"
        if record["qualified"]
        else
        "Failed"
    )


    # ========================================================
    # SUBJECT TABLE
    # ========================================================

    st.write(
        "### 📊 Subject-wise Scorecard Breakdown"
    )


    table_data = []


    for code, p in record["papers"].items():

        status = (
            "Pass"
            if p["passed"]
            else
            "Fail"
        )


        if p["type"] == "merit":

            status = "Merit Subject"


        table_data.append({

            "Paper Code":
                code,

            "Subject Name":
                p["subject_name"],

            "Selected OMR Set":
                p["set_name"],

            "Correct (+2)":
                p["correct"],

            "Wrong (0)":
                p["wrong"],

            "Unattempted":
                p["skipped"],

            "Deleted Questions":
                f"{p['deleted_count']} Qs "
                f"(+{p['bonus_marks']} Bonus)",

            "Total Score":
                f"{p['score']} / "
                f"{p['max_marks']}",

            "Percentage":
                f"{p['pct']}%",

            "Status":
                status
        })


    st.dataframe(

        pd.DataFrame(
            table_data
        ),

        hide_index=True,

        use_container_width=True
    )


    # ========================================================
    # QUESTION-BY-QUESTION AUDIT
    # ========================================================

    st.write(
        "### 🔍 Detailed Question-by-Question Response Audit"
    )


    for code, p in record["papers"].items():

        with st.expander(

            f"Inspect Responses for "
            f"{code}: "
            f"{p['subject_name']} "
            f"({p['set_name']})",

            expanded=False
        ):

            df_itemized = pd.DataFrame(
                p["itemized_breakdown"]
            )


            st.dataframe(

                df_itemized,

                hide_index=True,

                use_container_width=True
            )


    # ========================================================
    # SHAREABLE LINK
    # ========================================================

    st.write(
        "### 🔗 Shareable Direct Scorecard Link"
    )


    base_url = st.query_params.get(
        "base_url",
        "http://localhost:8501"
    )


    direct_link = (
        f"{base_url}"
        f"?roll_no="
        f"{record['roll_no']}"
    )


    st.code(
        direct_link,
        language="text"
    )


# ============================================================
# STREAMLIT PAGE CONFIG
# ============================================================

st.set_page_config(

    page_title=
        "BPSC OMR Evaluator & Merit Leaderboard",

    layout="wide",

    page_icon="📝"
)


st.title(
    "📝 BPSC Student OMR "
    "Marksheet Generator & Merit Portal"
)


query_params = st.query_params


url_roll_no = query_params.get(
    "roll_no",
    None
)


# ============================================================
# TABS
# ============================================================

tabs = st.tabs([

    "📤 Student OMR Portal",

    "🔍 Direct Scorecard Lookup",

    "🏆 Live Merit Leaderboard"
])


# ============================================================
# TAB 1
# ============================================================

with tabs[0]:

    st.subheader(
        "Candidate Identity & OMR Sheet Submission"
    )


    st.info(
        "ℹ️ Candidate name and roll number are read automatically "
        "from the OMR sheets. You do not need to enter them manually. "
        "All six OMR sheets are required, and all six must contain the "
        "same name and roll number before a marksheet can be published."
    )

    st.divider()

    # The upload section is always enabled; identity comes only from OMR.
    # All candidate identity data is obtained from the uploaded sheets.

    st.write(

        "### 📂 Upload Your OMR "
        "Response Sheets & Select OMR Set Code"
    )


    omr_files = {}

    omr_sets = {}


    # ====================================================
    # SIX PAPERS
    # ====================================================

    for code, meta in SUBJECT_META.items():

        category_tag = (

            "Qualifying Paper (Min 30%)"

            if meta["type"] == "qualifying"

            else

            "Merit Paper"
        )


        available_sets = list(
            OFFICIAL_KEYS[code].keys()
        )


        with st.expander(

            f"📄 {code}: "
            f"{meta['name']} "
            f"[{category_tag}]",

            expanded=False
        ):

            col_s, col_f = st.columns(
                [1, 2]
            )


            with col_s:

                omr_sets[code] = (
                    st.selectbox(

                        f"Select OMR Set for {code}",

                        options=
                            available_sets,

                        key=
                            f"set_{code}"
                    )
                )


            with col_f:

                omr_files[code] = (
                    st.file_uploader(

                        f"Upload {code} "
                        "OMR Sheet "
                        "(PDF / JPG / PNG)",

                        type=[
                            "pdf",
                            "jpg",
                            "jpeg",
                            "png"
                        ],

                        key=
                            f"omr_{code}"
                    )
                )


    st.divider()


    # ====================================================
    # CALCULATE
    # ====================================================

    if st.button(

        "🚀 Verify All Six OMR Sheets & Generate Marksheet",

        type="primary"
    ):

        # =================================================
        # HARD GATE: EXACTLY SIX REQUIRED OMR SHEETS
        # =================================================

        missing_codes = [
            code
            for code in SUBJECT_META
            if omr_files.get(code) is None
        ]

        if missing_codes:
            st.error(
                "❌ Marksheet NOT published. All six OMR sheets are required. "
                f"Missing: {', '.join(missing_codes)}."
            )
            st.stop()

        # =================================================
        # HARD GATE: NAME + ROLL MUST MATCH ON ALL SIX SHEETS
        # =================================================

        identity_results = {}
        for code in SUBJECT_META:
            identity_results[code] = extract_omr_identity(
                omr_files[code],
                valid_sets=list(OFFICIAL_KEYS[code].keys()),
            )

        normalized_identities = {}
        identity_rows = []

        for code, identity in identity_results.items():
            normalized_identities[code] = {
                "name": _normalize_identity_name(identity.get("name", "")),
                "roll_no": _normalize_roll_number(identity.get("roll_no", "")),
            }
            identity_rows.append({
                "Paper": code,
                "Name detected from OMR": identity.get("name") or "NOT DETECTED",
                "Roll number detected from OMR": identity.get("roll_no") or "NOT DETECTED",
                "Booklet set detected from OMR": identity.get("booklet_set") or "NOT DETECTED",
                "Answer key selected": omr_sets[code],
            })

        st.write("### 🔎 OMR Identity Verification")
        st.dataframe(
            pd.DataFrame(identity_rows),
            hide_index=True,
            use_container_width=True,
        )

        identity_values = list(normalized_identities.values())
        all_identity_present = all(
            item["name"] and item["roll_no"]
            for item in identity_values
        )

        # Roll number must match exactly because it is read from the OMR
        # bubbles.  Handwritten-name OCR can contain small character errors,
        # so compare names fuzzily rather than rejecting the same student for
        # OCR differences.
        reference_name = identity_values[0]["name"] if identity_values else ""
        name_matches = all(
            _name_similarity(reference_name, item["name"]) >= 0.70
            for item in identity_values
        ) if all_identity_present else False

        same_identity = (
            all_identity_present
            and name_matches
            and len({item["roll_no"] for item in identity_values}) == 1
        )

        if not same_identity:
            st.error(
                "❌ Marksheet NOT published. The six OMR sheets do not have "
                "the same valid student identity. The bubbled roll number must "
                "match exactly, and the handwritten name must match after OCR "
                "normalization."
            )
            st.warning(
                "Please re-upload/correct the OMR sheets. No score was saved "
                "to the database and no rank was generated."
            )
            st.stop()

        # =================================================
        # HARD GATE: ANSWER-KEY SET MUST MATCH OMR SET
        # =================================================

        set_mismatches = []
        for code in SUBJECT_META:
            detected_set_letter = str(
                identity_results[code].get("booklet_set", "")
            ).strip().upper()
            selected_set = str(omr_sets[code]).strip()
            selected_set_letter = selected_set[-1:].upper()

            if not detected_set_letter:
                set_mismatches.append(
                    f"{code}: OMR booklet set could not be detected "
                    f"(selected answer key: {selected_set})"
                )
            elif detected_set_letter != selected_set_letter:
                set_mismatches.append(
                    f"{code}: OMR sheet is Set-{detected_set_letter}, "
                    f"but Set-{selected_set_letter} answer key was selected"
                )

        if set_mismatches:
            st.error(
                "❌ Marksheet NOT published. The OMR response-sheet booklet "
                "set does not match the selected answer-key set."
            )
            st.warning(
                "Each paper must be scored only against the answer key for "
                "the same Question Booklet Series printed on the OMR sheet."
            )
            st.dataframe(
                pd.DataFrame({"Set mismatch": set_mismatches}),
                hide_index=True,
                use_container_width=True,
            )
            st.stop()

        # Identity and answer-key set are now both verified before scoring.
        # Keep the display name as detected from the first OMR sheet; use
        # normalized values only for matching/validation.
        first_code = next(iter(SUBJECT_META))
        student_name = identity_results[first_code]["name"]
        roll_no = identity_values[0]["roll_no"]

        paper_results = {}

        merit_total = 0

        merit_max = 0


        # =================================================
        # PROCESS EVERY PAPER
        # =================================================

        for code, meta in SUBJECT_META.items():

            selected_set = (
                omr_sets[code]
            )


            official_key = (
                OFFICIAL_KEYS[
                    code
                ][
                    selected_set
                ]
            )


            parsed_responses = (
                parse_omr_file(
                    omr_files.get(code)
                )
            )

            if not parsed_responses:
                st.error(
                    f"❌ Marksheet NOT published. {code} OMR answers could not be read."
                )
                st.warning(
                    "No database record or rank was created. Please upload a clearer "
                    "OMR sheet and try again."
                )
                st.stop()


            eval_res = evaluate_paper(

                parsed_responses,

                official_key
            )


            # ---------------------------------------------
            # QUALIFYING
            # ---------------------------------------------

            is_passed = True


            if meta["type"] == "qualifying":

                is_passed = (

                    eval_res["pct"]
                    >=
                    meta["cutoff_pct"]
                )


            # ---------------------------------------------
            # MERIT
            # ---------------------------------------------

            else:

                merit_total += (
                    eval_res["score"]
                )


                merit_max += (
                    eval_res["max_marks"]
                )


            eval_res["passed"] = (
                is_passed
            )


            eval_res["subject_name"] = (
                meta["name"]
            )


            eval_res["type"] = (
                meta["type"]
            )


            eval_res["set_name"] = (
                selected_set
            )


            paper_results[code] = (
                eval_res
            )


        # =================================================
        # OVERALL QUALIFICATION
        # =================================================

        is_qualified = (

            paper_results["P1"]["passed"]

            and

            paper_results["P2"]["passed"]
        )


        merit_pct = (

            round(
                (
                    merit_total
                    /
                    merit_max
                )
                * 100,
                2
            )

            if merit_max > 0

            else

            0.0
        )


        # =================================================
        # DATABASE
        # =================================================

        db = load_data()


        # Remove previous result for same roll number.

        db = [

            item

            for item in db

            if str(
                item["roll_no"]
            ).strip()
            !=
            str(
                roll_no
            ).strip()
        ]


        new_entry = {

            "name":
                student_name,

            "roll_no":
                roll_no,

            # Public anonymous identifier. The actual roll number is
            # retained only for private identity verification/lookup.
            "public_id":
                generate_unique_public_id(db),

            "papers":
                paper_results,

            "merit_total":
                merit_total,

            "merit_max":
                merit_max,

            "merit_pct":
                merit_pct,

            "qualified":
                is_qualified,

            # This record was created only after all six OMR sheets
            # supplied the same OCR-verified identity.
            "identity_verified":
                True
        }


        db.append(
            new_entry
        )


        # =================================================
        # RANKING
        # =================================================

        # Only records verified from all six OMR sheets can appear
        # in the new merit ranking. Legacy/manual records are not ranked.
        verified_records = [
            item
            for item in db
            if item.get("identity_verified", False) is True
        ]

        unverified_records = [
            item
            for item in db
            if item.get("identity_verified", False) is not True
        ]

        verified_records.sort(

            key=lambda x: (

                x["qualified"],

                x["merit_total"]

            ),

            reverse=True
        )


        for rank_idx, record in enumerate(

            verified_records,

            start=1
        ):

            record["rank"] = (
                rank_idx
            )

        # Explicitly remove stale ranks from legacy records.
        for record in unverified_records:
            record.pop("rank", None)

        db = verified_records + unverified_records


        save_data(db)


        st.balloons()


        saved_record = next(

            item

            for item in db

            if str(
                item["roll_no"]
            )
            ==
            str(
                roll_no
            )
        )


        verified_count = sum(
            1
            for item in db
            if item.get("identity_verified", False) is True
        )

        render_marksheet(

            saved_record,

            verified_count
        )

# ============================================================
# TAB 2
# ============================================================

with tabs[1]:

    st.subheader(
        "🔍 Search Candidate Scorecard"
    )


    search_roll = st.text_input(

        "Enter Candidate Roll Number:",

        value=
            url_roll_no
            if url_roll_no
            else "",

        placeholder=
            "e.g. 10843"
    )


    if search_roll.strip():

        db = load_data()
        ids_changed = ensure_public_ids(db)
        if ids_changed:
            save_data(db)

        verified_count = sum(
            1
            for item in db
            if item.get("identity_verified", False) is True
        )


        record = next(

            (

                item

                for item in db

                if str(
                    item["roll_no"]
                ).strip().lower()
                ==
                str(
                    search_roll
                ).strip().lower()
            ),

            None
        )


        if record:

            render_marksheet(

                record,

                verified_count
            )


        else:

            st.error(

                f"No marksheets found "
                f"for Roll Number: "
                f"`{search_roll}`"
            )


# ============================================================
# TAB 3
# ============================================================

with tabs[2]:

    st.subheader(
        "🏆 Official Live Merit Leaderboard"
    )


    all_db = load_data()
    ids_changed = ensure_public_ids(all_db)
    if ids_changed:
        save_data(all_db)

    db = [
        item
        for item in all_db
        if item.get("identity_verified", False) is True
    ]


    if not db:

        st.info(

            "No candidates with six matching, identity-verified OMR sheets "
            "have been published yet."
        )


    else:

        rows = []


        for item in db:

            p1 = item[
                "papers"
            ]["P1"]


            p2 = item[
                "papers"
            ]["P2"]


            rows.append({

                "Rank":
                    item.get(
                        "rank",
                        "N/A"
                    ),

                "Candidate Name":
                    item["name"],

                # Never expose the real OMR roll number in the public
                # leaderboard. This random ID is unique across all
                # published records and is not derived from the roll number.
                "Anonymous ID":
                    item.get("public_id", "N/A"),

                "Status":
                    (
                        "Qualified"
                        if item["qualified"]
                        else
                        "Disqualified"
                    ),

                "English (P1) Marks":
                    (
                        f"{p1['score']} "
                        f"("
                        f"{'Pass' if p1['passed'] else 'Fail'}"
                        f")"
                    ),

                "Hindi (P2) Marks":
                    (
                        f"{p2['score']} "
                        f"("
                        f"{'Pass' if p2['passed'] else 'Fail'}"
                        f")"
                    ),

                "Merit Total (P3-P6)":
                    (
                        f"{item['merit_total']} "
                        f"/ "
                        f"{item['merit_max']}"
                    ),

                "Merit Percentage":
                    (
                        f"{item['merit_pct']}%"
                    )
            })


        st.dataframe(

            pd.DataFrame(rows),

            hide_index=True,

            use_container_width=True
        )
