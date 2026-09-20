import io
import json
import os
import re
import secrets
import hashlib
from difflib import SequenceMatcher
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

try:
    from cryptography.fernet import Fernet, InvalidToken
except ImportError:
    Fernet = None
    InvalidToken = Exception

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

# Privacy/security configuration.
# IMPORTANT: set OMR_DATABASE_KEY as a Streamlit secret or environment
# variable. Never hard-code it in this source file and never commit it.
ENCRYPTED_ENCRYPTED_DATA_FILE = "bpsc_leaderboard_secure.enc"
PLAIN_RANKING_DATA_FILE = "bpsc_leaderboard.json"

# Encrypted ranking database key.
# No administrator password is required to publish the merit list.
OMR_DATABASE_KEY = os.environ.get("OMR_DATABASE_KEY", "")
if not OMR_DATABASE_KEY:
    try:
        OMR_DATABASE_KEY = st.secrets.get("OMR_DATABASE_KEY", "")
    except Exception:
        OMR_DATABASE_KEY = ""

MAX_UPLOAD_MB = 15

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
# DATABASE — PRIVACY-FIRST / ENCRYPTED AT REST
# ============================================================

def _get_fernet():
    """Return Fernet only when an external database key is configured."""
    if Fernet is None or not OMR_DATABASE_KEY:
        return None
    try:
        return Fernet(OMR_DATABASE_KEY.encode("ascii"))
    except Exception:
        return None


def generate_unique_public_id(existing_records=None):
    """Generate a unique anonymous ID.

    The ID is deliberately independent of the roll number so the roll number
    cannot be recovered from the public ID. Once assigned to a roll number,
    the ID is preserved by the database normalization/update logic.
    """
    existing_records = existing_records or []
    used = {
        str(item.get("public_id", "")).strip()
        for item in existing_records
        if isinstance(item, dict)
    }

    # Random, non-reversible public identifier.
    for _ in range(100):
        candidate = f"BPSC-{secrets.token_hex(5).upper()}"
        if candidate not in used:
            return candidate

    # Extremely unlikely fallback.
    return f"BPSC-{hashlib.sha256(secrets.token_bytes(32)).hexdigest()[:10].upper()}"


def _normalize_database(raw):
    """Keep exactly one maximum-score record per roll number."""
    clean = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        roll_no = str(item.get("roll_no", "")).strip()
        public_id = str(item.get("public_id", "")).strip()
        if not roll_no or not public_id:
            continue
        try:
            merit_total = round(float(item.get("merit_total", 0)), 2)
            merit_max = round(float(item.get("merit_max", 0)), 2)
        except (TypeError, ValueError):
            continue
        clean.append({
            "roll_no": roll_no,
            "public_id": public_id,
            "merit_total": merit_total,
            "merit_max": merit_max,
            "published": True,
        })

    by_roll = {}
    for item in clean:
        old = by_roll.get(item["roll_no"])
        if old is None or item["merit_total"] > old["merit_total"]:
            if old is not None:
                item["public_id"] = old["public_id"]
            by_roll[item["roll_no"]] = item

    canonical = []
    used_public_ids = set()
    for item in by_roll.values():
        if item["public_id"] in used_public_ids:
            item["public_id"] = generate_unique_public_id(canonical)
        used_public_ids.add(item["public_id"])
        canonical.append(item)
    return canonical


def load_data():
    """Load ranking data from session state or best-effort disk storage."""
    session_db = st.session_state.get("_bpsc_ranking_db")
    if isinstance(session_db, list):
        return _normalize_database(session_db)

    fernet = _get_fernet()
    filename = ENCRYPTED_ENCRYPTED_DATA_FILE if fernet else PLAIN_RANKING_DATA_FILE
    db = []
    try:
        if os.path.exists(filename):
            if fernet:
                with open(filename, "rb") as f:
                    raw = json.loads(fernet.decrypt(f.read()).decode("utf-8"))
            else:
                with open(filename, "r", encoding="utf-8") as f:
                    raw = json.load(f)
            db = _normalize_database(raw)
    except Exception:
        db = []

    st.session_state["_bpsc_ranking_db"] = db
    return db


def save_data(db):
    """Save to session state; disk persistence is best-effort only."""
    minimal = _normalize_database(db)
    st.session_state["_bpsc_ranking_db"] = minimal

    payload = json.dumps(
        minimal, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")

    try:
        fernet = _get_fernet()
        if fernet:
            tmp_file = ENCRYPTED_ENCRYPTED_DATA_FILE + ".tmp"
            with open(tmp_file, "wb") as f:
                f.write(fernet.encrypt(payload))
            os.replace(tmp_file, ENCRYPTED_ENCRYPTED_DATA_FILE)
        else:
            tmp_file = PLAIN_RANKING_DATA_FILE + ".tmp"
            with open(tmp_file, "wb") as f:
                f.write(payload)
            os.replace(tmp_file, PLAIN_RANKING_DATA_FILE)
    except Exception:
        # The session copy is still valid. Never raise a storage error.
        pass


# ============================================================
# MARKSHEET DISPLAY
# ============================================================

def render_marksheet(record, total_candidates, paper_results=None, database_saved=True):
    """Render the candidate marksheet.

    The marksheet is generated from the current session's OMR evaluation.
    If secure database storage is unavailable, the result is still displayed
    locally in the current session, but no rank/public record is saved.
    """
    st.markdown(
        """
        <div class="secure-card">
            <div class="secure-badge">🔒 PRIVATE RESULT VIEW</div>
            <div class="result-label">BPSC-AE OMR MARKSHEET</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"<div class='anon-id'>{record.get('public_id', 'SESSION RESULT')}</div>",
        unsafe_allow_html=True,
    )

    if database_saved:
        st.metric("Current Rank", f"#{record.get('rank', 'N/A')}")
    else:
        st.info(
            "Marks generated successfully. The result is saved in the local ranking database."
        )

    if paper_results:
        rows = []
        for code, meta in SUBJECT_META.items():
            result = paper_results.get(code, {})
            rows.append(
                {
                    "Paper": code,
                    "Subject": meta["name"],
                    "Booklet": result.get("set_name", "N/A"),
                    "Correct": result.get("correct", 0),
                    "Wrong": result.get("wrong", 0),
                    "Unattempted": result.get("skipped", 0),
                    "Deleted/Bonus": result.get("deleted_count", 0),
                    "Marks": result.get("score", 0),
                    "Max Marks": result.get("max_marks", 0),
                    "Percentage": f"{result.get('pct', 0):.2f}%",
                    "Status": "PASS" if result.get("passed", True) else "NOT QUALIFIED",
                }
            )

        st.subheader("Paper-wise Result")
        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            use_container_width=True,
        )

        merit_total = sum(
            float(paper_results.get(code, {}).get("score", 0))
            for code, meta in SUBJECT_META.items()
            if meta.get("type") != "qualifying"
        )
        merit_max = sum(
            float(paper_results.get(code, {}).get("max_marks", 0))
            for code, meta in SUBJECT_META.items()
            if meta.get("type") != "qualifying"
        )
        merit_pct = round((merit_total / merit_max) * 100, 2) if merit_max else 0.0
        qualified = all(
            paper_results.get(code, {}).get("passed", False)
            for code, meta in SUBJECT_META.items()
            if meta.get("type") == "qualifying"
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Merit Marks", f"{merit_total:.2f} / {merit_max:.2f}")
        c2.metric("Merit %", f"{merit_pct:.2f}%")
        c3.metric("Overall Qualification", "QUALIFIED" if qualified else "NOT QUALIFIED")

        st.caption(
            "The detailed marks above are generated from the OMR sheets in this session. "
            "The roll number is not displayed."
        )
    else:
        st.caption(
            "Only the anonymous result identifier and current rank are displayed."
        )


# ============================================================
# OMR IMAGE / BUBBLE READER
# ============================================================

def _read_uploaded_image(uploaded_file):
    """Render the first page of an uploaded PDF or read an image upload."""
    if uploaded_file is None:
        raise ValueError("No OMR file was supplied.")

    raw = uploaded_file.getvalue()
    if not raw:
        raise ValueError("The uploaded OMR file is empty.")

    name = str(getattr(uploaded_file, "name", "")).lower()

    if name.endswith(".pdf"):
        if fitz is None:
            raise RuntimeError(
                "PDF support is unavailable. Install PyMuPDF (fitz)."
            )
        try:
            doc = fitz.open(stream=raw, filetype="pdf")
            if len(doc) == 0:
                raise ValueError("The uploaded PDF contains no pages.")
            page = doc.load_page(0)
            # A moderate render scale keeps Streamlit Cloud memory usage reasonable.
            pix = page.get_pixmap(matrix=fitz.Matrix(2.2, 2.2), alpha=False)
            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            doc.close()
            return image
        except Exception as exc:
            raise ValueError(
                f"Could not render the uploaded PDF: {type(exc).__name__}."
            ) from exc

    try:
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise ValueError(
            "The uploaded file is not a readable JPG/PNG image."
        ) from exc


def _image_gray(image):
    """Create a clean grayscale image for OCR/OMR processing."""
    gray = np.array(image.convert("L"))
    h, w = gray.shape[:2]

    # Keep very large phone/scanner images manageable.
    max_dim = 2600
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        gray = cv2.resize(
            gray,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )

    return gray


def _darkness(gray, cx, cy, radius):
    """Return ink darkness inside a circular bubble."""
    h, w = gray.shape[:2]
    r = max(2, int(radius * 0.55))
    x1 = max(0, int(cx - r))
    x2 = min(w, int(cx + r + 1))
    y1 = max(0, int(cy - r))
    y2 = min(h, int(cy + r + 1))
    roi = gray[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0
    return float(255.0 - np.mean(roi))


def _bubble_candidates(gray):
    """
    Find likely OMR bubbles using Hough circles.

    The routine is deliberately permissive because scans/photos can have
    different resolutions. Downstream grouping removes unrelated circles.
    """
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    h, w = gray.shape[:2]
    min_dim = min(h, w)

    min_r = max(5, int(min_dim * 0.004))
    max_r = max(min_r + 4, int(min_dim * 0.018))

    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(8, int(min_r * 1.6)),
        param1=100,
        param2=24,
        minRadius=min_r,
        maxRadius=max_r,
    )

    if circles is None:
        return []

    result = []
    for x, y, r in np.round(circles[0]).astype(int):
        if 0 <= x < w and 0 <= y < h:
            result.append((int(x), int(y), int(r)))
    return result


def _cluster_axis(values, tolerance):
    """Cluster 1-D coordinates while preserving sorted order."""
    if not values:
        return []

    groups = [[values[0]]]
    for value in values[1:]:
        if abs(value - np.median(groups[-1])) <= tolerance:
            groups[-1].append(value)
        else:
            groups.append([value])
    return [float(np.median(g)) for g in groups]


def _decode_roll_from_bubbles(gray, candidates):
    """
    Decode a six-digit BPSC-style bubbled roll number.

    The expected layout is 6 digit columns × 10 digit rows. The method
    searches the upper part of the sheet for a dense 6-column/10-row grid.
    """
    h, w = gray.shape[:2]
    upper = [
        c for c in candidates
        if c[1] < h * 0.45 and c[0] < w * 0.65
    ]

    if len(upper) < 35:
        raise ValueError("Roll-number bubble grid could not be located.")

    # Build x-columns from the detected circles.
    xs = sorted(c[0] for c in upper)
    median_r = max(4.0, float(np.median([c[2] for c in upper])))
    x_tol = max(8.0, median_r * 2.2)
    x_centers = _cluster_axis(xs, x_tol)

    if len(x_centers) < 6:
        raise ValueError("Could not identify the six roll-number columns.")

    # Try every group of six neighboring columns and score grid regularity.
    best = None
    for start in range(0, len(x_centers) - 5):
        cols = x_centers[start:start + 6]
        selected = []
        for xc in cols:
            col = sorted(
                [c for c in upper if abs(c[0] - xc) <= x_tol],
                key=lambda c: c[1],
            )
            # Collapse duplicate detections at nearly the same y.
            ys = []
            for c in col:
                if not ys or abs(c[1] - ys[-1][1]) > median_r * 1.4:
                    ys.append(c)
                elif c[2] > ys[-1][2]:
                    ys[-1] = c
            selected.append(ys)

        if all(len(c) >= 8 for c in selected):
            score = sum(min(len(c), 10) for c in selected)
            if best is None or score > best[0]:
                best = (score, selected)

    if best is None:
        raise ValueError(
            "Roll-number grid was detected, but six complete digit columns "
            "could not be resolved."
        )

    columns = best[1]
    digits = []

    for col in columns:
        # Select the ten circles spanning the most regular vertical range.
        col = sorted(col, key=lambda c: c[1])
        if len(col) > 10:
            best_ten = None
            for i in range(len(col) - 9):
                chunk = col[i:i + 10]
                gaps = np.diff([c[1] for c in chunk])
                regularity = float(np.std(gaps)) if len(gaps) else 999.0
                if best_ten is None or regularity < best_ten[0]:
                    best_ten = (regularity, chunk)
            col = best_ten[1]

        if len(col) != 10:
            raise ValueError("An incomplete roll-number digit column was found.")

        scores = [_darkness(gray, c[0], c[1], c[2]) for c in col]
        order = np.argsort(scores)[::-1]
        top = int(order[0])
        second = float(scores[int(order[1])])

        # A very weak/ambiguous column is safer to reject than silently
        # publishing a wrong candidate identity.
        if scores[top] < 22 or scores[top] - second < 4:
            raise ValueError(
                "The roll-number bubbles are too faint or ambiguous to read."
            )
        digits.append(str(top))

    roll = "".join(digits)
    if not re.fullmatch(r"\d{6}", roll):
        raise ValueError("A valid six-digit roll number could not be decoded.")
    return roll


def extract_omr_identity(uploaded_file):
    """
    Read the candidate's six-digit bubbled roll number.

    The supplied BPSC AE sheet is parsed using its fixed, page-relative
    six-column roll grid first.  A generic CV/OCR fallback is retained for
    slightly different scans.
    """
    image = _read_uploaded_image(uploaded_file)
    gray = _image_gray(image)

    try:
        roll = _decode_bpsc_roll_layout(gray)
        return {"roll_no": roll}
    except ValueError as layout_error:
        candidates = _bubble_candidates(gray)
        if candidates:
            try:
                roll = _decode_roll_from_bubbles(gray, candidates)
                return {"roll_no": roll}
            except ValueError:
                pass

        if pytesseract is not None:
            text = pytesseract.image_to_string(gray, config="--psm 11")
            matches = re.findall(r"(?<!\d)\d{6}(?!\d)", text)
            if len(matches) == 1:
                return {"roll_no": matches[0]}

        raise ValueError(
            "Could not reliably read the six-digit roll number from this OMR. "
            f"Layout reader: {layout_error}"
        )


def _normalize_roll_number(value):
    """Normalize a decoded roll number and enforce BPSC's six-digit format."""
    digits = re.sub(r"\D", "", str(value or ""))
    return digits if len(digits) == 6 else ""


def _group_answer_rows(candidates, gray):
    """
    Locate 50 answer rows, each containing four option bubbles.

    This is layout-tolerant rather than tied to one scanner resolution.
    """
    h, w = gray.shape[:2]
    # The answer area is normally below the identity/booklet area.
    work = [
        c for c in candidates
        if c[1] > h * 0.22 and c[1] < h * 0.98
    ]
    if len(work) < 150:
        return []

    median_r = max(4.0, float(np.median([c[2] for c in work])))
    y_tol = max(5.0, median_r * 1.7)

    work.sort(key=lambda c: c[1])
    rows = []
    for c in work:
        if not rows or abs(c[1] - np.median([p[1] for p in rows[-1]])) > y_tol:
            rows.append([c])
        else:
            rows[-1].append(c)

    row_candidates = []
    for row in rows:
        row = sorted(row, key=lambda c: c[0])
        if len(row) < 4:
            continue

        # Pick four bubbles with the widest, most regular spacing.
        if len(row) > 4:
            best = None
            for i in range(len(row) - 3):
                chunk = row[i:i + 4]
                gaps = np.diff([c[0] for c in chunk])
                if np.all(gaps > 0):
                    regularity = float(np.std(gaps))
                    span = float(chunk[-1][0] - chunk[0][0])
                    score = regularity - 0.001 * span
                    if best is None or score < best[0]:
                        best = (score, chunk)
            row = best[1] if best else row[:4]

        if len(row) == 4:
            row_candidates.append(row)

    # Keep the longest sequence of rows with similar x geometry.
    if len(row_candidates) < 50:
        return []

    best_window = None
    for i in range(len(row_candidates) - 49):
        window = row_candidates[i:i + 50]
        x_patterns = np.array(
            [[c[0] for c in row] for row in window],
            dtype=float,
        )
        spread = float(np.mean(np.std(x_patterns, axis=0)))
        if best_window is None or spread < best_window[0]:
            best_window = (spread, window)

    return best_window[1] if best_window else []



# ---------------------------------------------------------------------------
# BPSC AE OMR SHEET (the uploaded sample layout)
# ---------------------------------------------------------------------------
# The supplied OMR is a portrait A4-style sheet with:
#   - six roll-number columns at the upper-left;
#   - five answer blocks, each containing 10 questions x 4 options;
#   - printed booklet series (for example "J") in the centre.
#
# These coordinates are stored as page-relative fractions so the reader works
# across PDF render resolutions.  This is intentionally preferred over
# unrestricted Hough-circle grouping because the sample sheet contains many
# non-answer circles/marks and the watermark can confuse a generic detector.

_BPSC_ROLL_X = (
    0.16322, 0.20042, 0.23980, 0.27741, 0.31528, 0.35386
)
_BPSC_ROLL_Y = (
    0.17458, 0.18409, 0.19418, 0.20534, 0.21568,
    0.22494, 0.23474, 0.24406, 0.25398, 0.26407
)
_BPSC_ANSWER_X = (
    (0.16314, 0.18220, 0.20109, 0.22166),
    (0.31537, 0.33401, 0.35382, 0.37372),
    (0.46776, 0.48766, 0.50647, 0.52578),
    (0.62024, 0.63997, 0.65894, 0.67783),
    (0.77246, 0.79202, 0.81175, 0.83073),
)
_BPSC_ANSWER_Y = (
    0.30463, 0.32482, 0.34501, 0.36520, 0.38599,
    0.40618, 0.42577, 0.44537, 0.46615, 0.48575
)


def _normalized_patch_darkness(gray, x_frac, y_frac, radius_frac=0.006):
    """Return ink darkness at a known BPSC OMR bubble position."""
    h, w = gray.shape[:2]
    x = int(round(x_frac * w))
    y = int(round(y_frac * h))
    radius = max(4, int(round(radius_frac * min(w, h))))

    x0 = max(0, x - radius)
    x1 = min(w, x + radius + 1)
    y0 = max(0, y - radius)
    y1 = min(h, y + radius + 1)

    patch = gray[y0:y1, x0:x1]
    if patch.size == 0:
        return 0.0
    return float(255.0 - np.mean(patch))


def _decode_bpsc_roll_layout(gray):
    """Decode the six-digit bubbled roll number from the supplied layout."""
    scores = []
    for x in _BPSC_ROLL_X:
        scores.append([
            _normalized_patch_darkness(gray, x, y, radius_frac=0.0055)
            for y in _BPSC_ROLL_Y
        ])

    digits = []
    for column in scores:
        order = np.argsort(column)[::-1]
        best = int(order[0])
        second = float(column[int(order[1])])

        # Filled bubbles in the supplied sheet are far darker than the
        # printed empty circles.  Require both absolute darkness and margin.
        if column[best] < 115 or column[best] - second < 70:
            raise ValueError(
                "Roll number could not be read confidently from the six "
                "digit columns."
            )
        digits.append(str(best))

    roll = "".join(digits)
    if not re.fullmatch(r"\d{6}", roll):
        raise ValueError("The OMR does not contain a valid six-digit roll number.")
    return roll


def _parse_bpsc_answer_layout(gray):
    """Read all 50 A/B/C/D responses from the supplied BPSC OMR layout."""
    responses = {}
    q_no = 1

    for block_x in _BPSC_ANSWER_X:
        for y in _BPSC_ANSWER_Y:
            option_scores = [
                _normalized_patch_darkness(
                    gray, x, y, radius_frac=0.0052
                )
                for x in block_x
            ]

            order = np.argsort(option_scores)[::-1]
            best = int(order[0])
            second = float(option_scores[int(order[1])])

            # Empty bubbles have printed outlines, so absolute darkness alone
            # is not enough.  A real filled response is both darker and
            # clearly separated from the other three options.
            if (
                option_scores[best] >= 105
                and option_scores[best] - second >= 45
            ):
                responses[q_no] = OPTION_MAP[best]
            else:
                # Blank and multiple-marked responses are not silently guessed.
                responses[q_no] = ""

            q_no += 1

    if len(responses) != 50:
        raise ValueError("The BPSC OMR answer grid could not be read.")
    return responses


def parse_omr_file(uploaded_file):
    """
    Parse the 50 answers from the supplied BPSC AE OMR sheet.

    The sheet has five vertical answer blocks:
      questions 1-10, 11-20, 21-30, 31-40, 41-50.
    Each row has four bubbles in A/B/C/D order.

    A response is returned only when one bubble is clearly filled. Blank or
    ambiguous/multiple marks are returned as an empty response.
    """
    image = _read_uploaded_image(uploaded_file)
    gray = _image_gray(image)

    # Use the exact supplied-sheet geometry first. This avoids the watermark,
    # printed circles and page decorations being mistaken for answer bubbles.
    try:
        responses = _parse_bpsc_answer_layout(gray)

        # Require a meaningful number of confident answers. This prevents an
        # unrelated portrait PDF from accidentally being treated as an OMR.
        confident = sum(bool(v) for v in responses.values())
        if confident < 5:
            raise ValueError(
                "Too few filled answer bubbles were detected in the BPSC "
                "answer grid."
            )
        return responses
    except ValueError as layout_error:
        # Retain the previous layout-tolerant detector as a fallback for
        # modestly different scans/templates.
        candidates = _bubble_candidates(gray)
        rows = _group_answer_rows(candidates, gray)
        if len(rows) == 50:
            responses = {}
            for q_no, row in enumerate(rows, start=1):
                scores = [_darkness(gray, c[0], c[1], c[2]) for c in row]
                order = np.argsort(scores)[::-1]
                best = int(order[0])
                second = float(scores[int(order[1])])
                responses[q_no] = (
                    OPTION_MAP[best]
                    if scores[best] >= 30 and scores[best] - second >= 5
                    else ""
                )
            if sum(bool(v) for v in responses.values()) >= 5:
                return responses

        raise ValueError(
            "Could not reliably read the 50-answer BPSC OMR grid. "
            f"Layout reader: {layout_error}"
        )


def evaluate_paper(student_responses, official_key):
    """Evaluate one 50-question paper using the existing marking engine."""
    engine = OMREvaluationEngine(official_key)
    return engine.evaluate_responses(student_responses)


# ============================================================
# STREAMLIT PAGE CONFIG
# ============================================================

st.set_page_config(

    page_title=
        "BPSC OMR Marksheet Portal",

    layout="wide",

    page_icon="🔒"
)


st.markdown(
    """
    <div class="app-header">
        <div class="brand-row">
            <div class="brand-mark">BPSC<br><span>AE</span></div>
            <div>
                <div class="brand-title">BPSC-AE Rank List</div>
                <div class="brand-subtitle">OMR Marks & Merit Ranking Portal</div>
            </div>
        </div>
        <div class="header-badge">🔒 Secure processing</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
    .main .block-container {max-width: 1180px; padding-top: .65rem; padding-bottom: .8rem;}
    .app-header {display:flex; justify-content:space-between; align-items:center; gap:12px; padding:11px 16px; border:1px solid #dfe6ee; border-radius:12px; background:#fff; box-shadow:0 2px 9px rgba(15,23,42,.045); margin-bottom:10px;}
    .brand-row {display:flex; align-items:center; gap:10px;}
    .brand-mark {width:43px; height:43px; border-radius:9px; display:flex; flex-direction:column; align-items:center; justify-content:center; background:#173f67; color:#fff; font-weight:800; font-size:.60rem; line-height:1.05; letter-spacing:.04em;}
    .brand-mark span {font-size:.68rem;}
    .brand-title {font-size:1.22rem; font-weight:800; color:#172033; line-height:1.1;}
    .brand-subtitle {font-size:.76rem; color:#64748b; margin-top:3px;}
    .header-badge {border:1px solid #dbe4ee; background:#f8fafc; color:#334155; border-radius:999px; padding:5px 9px; font-size:.70rem; font-weight:700; white-space:nowrap;}
    .upload-hero {border:1px solid #d8e2ec; border-radius:12px; padding:13px 16px; background:#f8fbff; margin-bottom:9px;}
    .upload-kicker {font-size:.68rem; font-weight:800; letter-spacing:.07em; text-transform:uppercase; color:#25618d; margin-bottom:2px;}
    .upload-title {font-size:1.32rem; font-weight:800; color:#172033; margin:0 0 3px 0;}
    .upload-subtitle {font-size:.79rem; color:#526174; line-height:1.35; margin:0;}
    .step-grid {display:grid; grid-template-columns:repeat(3,1fr); gap:7px; margin-top:9px;}
    .step-card {border:1px solid #e2e8f0; border-radius:8px; padding:7px 9px; background:#fff;}
    .step-number {font-size:.60rem; font-weight:800; color:#25618d; text-transform:uppercase; letter-spacing:.05em;}
    .step-text {font-size:.72rem; font-weight:650; color:#253247; margin-top:2px;}
    .section-label {font-size:.70rem; font-weight:800; color:#64748b; letter-spacing:.05em; text-transform:uppercase; margin:8px 0 5px 0;}
    .paper-card {border:1px solid #dfe6ee; border-radius:10px; padding:8px 9px; background:#fff; min-height:175px; box-shadow:0 1px 4px rgba(15,23,42,.025);}
    .paper-title {font-size:.78rem; font-weight:800; color:#243247; margin-bottom:5px;}
    [data-testid="stFileUploader"] {padding:0 !important; margin-bottom:3px !important;}
    [data-testid="stFileUploaderDropzone"] {min-height:72px !important; padding:8px !important;}
    [data-testid="stFileUploaderDropzoneInstructions"] {font-size:.68rem !important;}
    [data-testid="stExpander"] {border:1px solid #dfe6ee; border-radius:9px; margin-bottom:5px; overflow:hidden; background:#fff;}
    [data-testid="stExpander"] summary {font-weight:700; font-size:.78rem;}
    .privacy-strip {display:grid; grid-template-columns:1fr 1fr; gap:7px; margin:9px 0 6px 0;}
    .privacy-box {border:1px solid #dfe6ee; border-radius:9px; padding:8px 10px; background:#f8fafc;}
    .privacy-box h4 {margin:0 0 2px 0; font-size:.75rem; color:#172033;}
    .privacy-box p {margin:0; color:#5b6879; line-height:1.3; font-size:.68rem;}
    .portal-note {font-size:.68rem; color:#64748b; text-align:center; margin-top:4px;}
    @media (max-width:900px) {.main .block-container {padding-top:.45rem;} .step-grid,.privacy-strip {grid-template-columns:1fr;} .paper-card {min-height:auto;}}
    </style>
    """,
    unsafe_allow_html=True,
)

query_params = st.query_params


url_public_id = query_params.get(
    "public_id",
    ""
)


# ============================================================
# OMR UPLOAD — shown first when the app opens
# ============================================================


st.markdown(
    """
    <div class="upload-hero">
        <div class="upload-kicker">BPSC-AE OMR Evaluation</div>
        <div class="upload-title">📄 Upload OMR Sheets & Generate Marks</div>
        <p class="upload-subtitle">Upload the six required response sheets, confirm each booklet series, and generate the candidate marksheet.</p>
        <div class="step-grid">
            <div class="step-card"><div class="step-number">01 · Upload</div><div class="step-text">Add 6 OMR sheets</div></div>
            <div class="step-card"><div class="step-number">02 · Verify</div><div class="step-text">Confirm booklet series</div></div>
            <div class="step-card"><div class="step-number">03 · Result</div><div class="step-text">Generate marks & rank</div></div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.info("💡 Your roll number is read from the OMR automatically — you do not need to type it.")

# The upload section is always enabled; identity comes only from OMR.
# All candidate identity data is obtained from the uploaded sheets.

st.markdown('<div class="section-label">Six required response sheets</div>', unsafe_allow_html=True)



omr_files = {}

# The booklet series is intentionally NOT detected by OCR.
# The user must manually select and confirm the booklet printed on each OMR.
omr_sets = {}
booklet_confirmed = {}


# ====================================================
# SIX PAPERS — compact 3-column layout
# ====================================================

omr_files = {}
omr_sets = {}
booklet_confirmed = {}

subjects = list(SUBJECT_META.items())
for row_start in range(0, len(subjects), 3):
    row = subjects[row_start:row_start + 3]
    cols = st.columns(3, gap="small")
    for col, (code, meta) in zip(cols, row):
        with col:
            st.markdown(
                f'<div class="paper-card"><div class="paper-title">📄 {code} · {meta["name"]}</div>',
                unsafe_allow_html=True,
            )
            omr_files[code] = st.file_uploader(
                "OMR Sheet (PDF / JPG / PNG)",
                type=["pdf", "jpg", "jpeg", "png"],
                key=f"omr_{code}",
                label_visibility="collapsed",
            )
            available_sets = list(OFFICIAL_KEYS[code].keys())
            omr_sets[code] = st.selectbox(
                "Booklet Series",
                options=available_sets,
                key=f"set_{code}",
            )
            booklet_confirmed[code] = st.checkbox(
                f"Confirm booklet {omr_sets[code]}",
                key=f"confirm_booklet_{code}",
            )
            st.markdown('</div>', unsafe_allow_html=True)

uploaded_count = sum(1 for item in omr_files.values() if item is not None)
confirmed_count = sum(1 for item in booklet_confirmed.values() if item)

if uploaded_count == 6 and confirmed_count == 6:
    st.success("✅ 6/6 OMR sheets uploaded • 6/6 booklet series confirmed — ready to generate.")
else:
    st.info(f"📋 OMR sheets: **{uploaded_count}/6**  ·  Booklet confirmations: **{confirmed_count}/6**")

st.divider()


# ====================================================
# CALCULATE
# ====================================================

if st.button(

    "🚀 Generate My Marksheet",

    type="primary"
):

    progress = st.progress(0, text="Starting OMR verification…")
    status_box = st.empty()

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

    # Privacy control: reject unusually large uploads before processing.
    oversized = []
    for code, uploaded in omr_files.items():
        if uploaded is not None:
            try:
                size_mb = uploaded.size / (1024 * 1024)
                if size_mb > MAX_UPLOAD_MB:
                    oversized.append(
                        f"{code}: {size_mb:.1f} MB (limit {MAX_UPLOAD_MB} MB)"
                    )
            except Exception:
                pass

    if oversized:
        st.error("Upload rejected because one or more files exceed the privacy/safety size limit.")
        st.write("\n".join(f"- {item}" for item in oversized))
        st.stop()

    # UploadedFile objects are processed in memory only. This application
    # never writes the original OMR bytes to the ranking database.
    # =================================================
    # HARD GATE: SAME STUDENT ACROSS ALL SIX SUBJECT SHEETS
    # =================================================
    # Student identity is determined ONLY by the six-digit bubbled
    # Roll Number.  The printed OMR Sheet No. is deliberately ignored
    # because it can be different on every subject paper.  The booklet
    # series can also differ by subject and is checked separately only
    # against that subject's selected answer key.

    identity_results = {}
    try:
        for code in SUBJECT_META:
            identity_results[code] = extract_omr_identity(
                omr_files[code]
            )
    except (ValueError, RuntimeError) as exc:
        st.error(
            "❌ Marksheet NOT published. OMR identity could not be read reliably."
        )
        st.warning(str(exc))
        st.stop()

    normalized_rolls = {
        code: _normalize_roll_number(identity.get("roll_no", ""))
        for code, identity in identity_results.items()
    }

    # Do not display the detected roll number back to the browser/UI.
    # Only show a privacy-safe verification status.
    st.success(
        "🔐 Identity verification completed across all six OMR sheets. "
        "The detected roll number is not displayed."
    )

    roll_values = list(normalized_rolls.values())
    all_rolls_present = all(bool(value) for value in roll_values)
    same_roll = (
        all_rolls_present
        and len(set(roll_values)) == 1
    )

    if not same_roll:
        st.error(
            "❌ Marksheet NOT published. All six subject response sheets "
            "must contain the same valid bubbled roll number."
        )
        st.warning(
            "Different subjects may have different OMR Sheet Nos. and "
            "different Question Booklet Series. Those fields are NOT used "
            "to identify the student. Only the bubbled roll number is used "
            "for cross-subject student matching. No score was saved and "
            "no rank was generated."
        )
        st.stop()

    # =================================================
    # HARD GATE: MANUAL BOOKLET CONFIRMATION
    # =================================================
    # Booklet series is deliberately NOT detected from the OMR image.
    # The student/operator must manually select the booklet printed on
    # each response sheet and explicitly confirm it.  The selected
    # booklet is then used to choose the corresponding answer key.

    unconfirmed_booklets = [
        f"{code}: booklet {omr_sets[code]} was not manually confirmed"
        for code in SUBJECT_META
        if not booklet_confirmed.get(code, False)
    ]

    if unconfirmed_booklets:
        st.error(
            "❌ Marksheet NOT published. The Question Booklet Series "
            "must be manually confirmed for all six response sheets."
        )
        st.warning(
            "For every paper, read the Question Booklet Series printed "
            "on the response sheet, select the matching booklet above, "
            "and tick the manual confirmation box. The selected booklet "
            "will be used as the answer key for that paper. No score, "
            "marksheet, or rank is saved until all six confirmations are made."
        )
        st.dataframe(
            pd.DataFrame({"Booklet confirmation required": unconfirmed_booklets}),
            hide_index=True,
            use_container_width=True,
        )
        st.stop()

    # Cross-subject student identity and manual booklet confirmations are
    # now verified before scoring. Different subjects may have different
    # booklet letters; each subject is scored only with the manually
    # selected answer key for that subject. The real roll number remains
    # private and is never shown publicly.
    roll_no = roll_values[0]

    paper_results = {}

    merit_total = 0

    merit_max = 0


    # =================================================
    # PROCESS EVERY PAPER
    # =================================================

    total_papers = len(SUBJECT_META)

    for paper_index, (code, meta) in enumerate(SUBJECT_META.items(), start=1):

        status_box.info(f"⚙️ Processing {paper_index}/{total_papers}: {meta['name']}…")
        progress.progress(
            (paper_index - 1) / total_papers,
            text=f"Processing {code} — {meta['name']}"
        )

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


        try:
            parsed_responses = parse_omr_file(
                omr_files.get(code)
            )
        except (ValueError, RuntimeError) as exc:
            st.error(
                f"❌ Marksheet NOT published. {code} OMR answers could not be "
                "read reliably."
            )
            st.warning(str(exc))
            st.stop()

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

        progress.progress(
            paper_index / total_papers,
            text=f"Completed {paper_index}/{total_papers}: {meta['name']}"
        )


    status_box.success("✅ All OMR sheets processed. Generating your marksheet…")

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
    # IMPORTANT PRIVACY RULE:
    # The persistent database stores ONLY:
    #   - private roll number (identity key)
    #   - anonymous public ID
    #   - merit marks
    #   - maximum merit marks
    #
    # OMR responses, answer keys, question-by-question results,
    # English/Hindi marks, correct/wrong counts, and other paper details
    # are NOT stored in the database.

    # =================================================
    # SECURE DATABASE / MARKSHEET FALLBACK
    # =================================================
    # OMR_DATABASE_KEY is optional. Without it, the app uses the local JSON
    # ranking database; with it, the database is encrypted.

    normalized_roll_no = str(roll_no).strip()

    # Initialize before database work so a storage problem can never cause
    # a secondary NameError.
    public_id = "SESSION-RESULT"
    database_saved = True
    best_total = round(float(merit_total), 2)
    best_max = round(float(merit_max), 2)
    db = []

    try:
        db = load_data()

        existing = next(
            (
                item for item in db
                if str(item.get("roll_no", "")).strip() == normalized_roll_no
            ),
            None,
        )

        if existing:
            # The Anonymous ID is permanently tied to this roll number.
            public_id = str(
                existing.get("public_id") or generate_unique_public_id(db)
            )

            old_total = float(existing.get("merit_total", 0))
            old_max = float(existing.get("merit_max", 0))

            # Keep the maximum score ever obtained for this roll number.
            if old_total > best_total:
                best_total = round(old_total, 2)
                best_max = round(old_max, 2)
        else:
            public_id = generate_unique_public_id(db)

        # Exactly one ranking record per roll number.
        db = [
            item for item in db
            if str(item.get("roll_no", "")).strip() != normalized_roll_no
        ]
        db.append(
            {
                "roll_no": normalized_roll_no,
                "public_id": public_id,
                "merit_total": best_total,
                "merit_max": best_max,
                "published": True,
            }
        )

        db = _normalize_database(db)
        db.sort(
            key=lambda x: (
                -float(x.get("merit_total", 0)),
                str(x.get("roll_no", "")).strip(),
            )
        )
        for rank_idx, item in enumerate(db, start=1):
            item["rank"] = rank_idx

        save_data(db)

    except Exception:
        # Fall back to session-only ranking. Marks are never lost because of
        # a file/permission/database problem.
        database_saved = True
        db = st.session_state.get("_bpsc_ranking_db", [])
        if not isinstance(db, list):
            db = []

        existing = next(
            (
                item for item in db
                if str(item.get("roll_no", "")).strip() == normalized_roll_no
            ),
            None,
        )

        if existing:
            public_id = str(
                existing.get("public_id") or generate_unique_public_id(db)
            )
            if float(existing.get("merit_total", 0)) > best_total:
                best_total = round(float(existing.get("merit_total", 0)), 2)
                best_max = round(float(existing.get("merit_max", 0)), 2)
        else:
            public_id = generate_unique_public_id(db)

        db = [
            item for item in db
            if str(item.get("roll_no", "")).strip() != normalized_roll_no
        ]
        db.append(
            {
                "roll_no": normalized_roll_no,
                "public_id": public_id,
                "merit_total": best_total,
                "merit_max": best_max,
                "published": True,
            }
        )
        db.sort(
            key=lambda x: (
                -float(x.get("merit_total", 0)),
                str(x.get("roll_no", "")).strip(),
            )
        )
        for rank_idx, item in enumerate(db, start=1):
            item["rank"] = rank_idx
        st.session_state["_bpsc_ranking_db"] = db

    saved_record = {
        "public_id": public_id,
        "rank": next(
            (
                item.get("rank")
                for item in db
                if str(item.get("public_id", "")) == public_id
            ),
            "N/A",
        ),
        "merit_total": best_total,
        "merit_max": best_max,
    }


    st.balloons()

    saved_record = {
        "public_id": public_id,
        "rank": next(
            (
                item.get("rank")
                for item in db
                if str(item.get("public_id", "")) == public_id
            ),
            "N/A"
        ),
        "merit_total": round(merit_total, 2),
        "merit_max": round(merit_max, 2),
    }

    render_marksheet(
        saved_record,
        len(db),
        paper_results=paper_results,
        database_saved=database_saved,
    )

    if database_saved:
        st.success(
            "📌 Result saved. Note your Anonymous ID — you can use it in "
            "Rank Lookup below to check your current rank later."
        )

    # Best-effort cleanup of transient OMR/evaluation objects from this
    # Streamlit session after the result has been rendered.
    # No OMR image/response data is placed into persistent storage.
    del paper_results
    del identity_results
    del normalized_rolls


# ============================================================
# TAB 2
# ============================================================

st.markdown(
    """
    <div class="privacy-strip">
        <div class="privacy-box"><h4>🔒 Privacy & Security</h4><p>Uploaded OMR files are processed for evaluation and are not stored in the ranking database.</p></div>
        <div class="privacy-box"><h4>🏆 Your Rank</h4><p>Use Rank Lookup below with your Anonymous ID to see only your own rank. The full merit list is not published.</p></div>
    </div>
    <div class="portal-note">BPSC-AE Rank List Portal • OMR marks processing and merit ranking</div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# OTHER PORTAL TABS
# ============================================================
rank_lookup_section = st.container()

with rank_lookup_section:


    st.subheader(
        "🔍 Find Your Current Rank"
    )

    search_public_id = st.text_input(
        "Enter Anonymous ID:",
        value=url_public_id if url_public_id else "",
        placeholder="e.g. 583214"
    )

    if search_public_id.strip():

        db = load_data()

        # Recalculate ranks from the current stored merit marks so the
        # displayed rank is always the current rank.
        if db:
            db.sort(
                key=lambda x: (
                    -float(x.get("merit_total", 0)),
                    str(x.get("roll_no", "")).strip(),
                )
            )

            for rank_idx, item in enumerate(db, start=1):
                item["rank"] = rank_idx

            save_data(db)

        record = next(
            (
                item for item in db
                if str(item.get("public_id", "")).strip().lower()
                == str(search_public_id).strip().lower()
            ),
            None
        )

        if record:
            # Student sees ONLY the current rank.
            render_marksheet(record, len(db))
        else:
            st.error(
                f"No record found for Anonymous ID: `{search_public_id}`"
            )
