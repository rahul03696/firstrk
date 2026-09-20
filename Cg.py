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
    """Return the encryption object or fail closed.

    The encryption key is deliberately NOT generated or stored by this app.
    It must be supplied externally (environment variable or Streamlit secret).
    This prevents the source code from containing the decryption key.
    """
    if Fernet is None:
        st.error(
            "Secure storage is unavailable: the 'cryptography' package is not installed."
        )
        st.stop()

    if not OMR_DATABASE_KEY:
        st.error(
            "Secure storage is locked. The administrator must configure "
            "OMR_DATABASE_KEY as a secret/environment variable before the app can run."
        )
        st.stop()

    try:
        key = OMR_DATABASE_KEY.encode("ascii")
        return Fernet(key)
    except Exception:
        st.error(
            "Secure storage is locked because OMR_DATABASE_KEY is invalid. "
            "Use a valid Fernet key."
        )
        st.stop()


def load_data():
    """Load only the minimal encrypted ranking database.

    The file contains no OMR images, answer selections, answer keys,
    question-level results, or English/Hindi marks.
    """
    if not os.path.exists(ENCRYPTED_ENCRYPTED_DATA_FILE):
        return []

    try:
        encrypted = open(ENCRYPTED_ENCRYPTED_DATA_FILE, "rb").read()
        decrypted = _get_fernet().decrypt(encrypted)
        raw = json.loads(decrypted.decode("utf-8"))

        # Defensive normalization: discard anything outside the approved
        # minimal schema, including legacy fields if an old record is present.
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
            })
        return clean

    except InvalidToken:
        st.error(
            "The encrypted database could not be opened with the configured key. "
            "No data was loaded."
        )
        st.stop()
    except Exception as exc:
        st.error(
            f"Secure database could not be read. No data was loaded. ({type(exc).__name__})"
        )
        st.stop()


def save_data(db):
    """Encrypt the minimal database before writing it to disk."""
    minimal = []
    for item in db:
        minimal.append({
            "roll_no": str(item.get("roll_no", "")).strip(),
            "public_id": str(item.get("public_id", "")).strip(),
            "merit_total": round(float(item.get("merit_total", 0)), 2),
            "merit_max": round(float(item.get("merit_max", 0)), 2),
        })

    payload = json.dumps(
        minimal,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")

    encrypted = _get_fernet().encrypt(payload)

    # Atomic replacement reduces the chance of leaving a partially-written
    # database if the process stops during a write.
    tmp_file = ENCRYPTED_ENCRYPTED_DATA_FILE + ".tmp"
    with open(tmp_file, "wb") as f:
        f.write(encrypted)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass
    os.replace(tmp_file, ENCRYPTED_ENCRYPTED_DATA_FILE)


def generate_unique_public_id(db):
    """Generate a random anonymous numeric ID."""
    used = {
        str(item.get("public_id", "")).strip()
        for item in db
    }

    while True:
        candidate = f"{secrets.randbelow(1_000_000):06d}"
        if candidate not in used:
            return candidate


# ============================================================
# MARKSHEET DISPLAY
# ============================================================

def render_marksheet(record, total_candidates):
    """Privacy-minimal student result: Anonymous ID + current rank only."""
    st.markdown(
        """
        <div class="secure-card">
            <div class="secure-badge">🔒 PRIVATE RESULT VIEW</div>
            <div class="result-label">Anonymous ID</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div class='anon-id'>{record.get('public_id', 'N/A')}</div>",
        unsafe_allow_html=True,
    )
    st.metric(
        "Current Rank",
        f"#{record.get('rank', 'N/A')}",
    )
    st.caption(
        "Only your current rank is displayed. Marks, OMR responses, "
        "answer keys and question-level details are not displayed."
    )


# ============================================================
# STREAMLIT PAGE CONFIG
# ============================================================

st.set_page_config(

    page_title=
        "BPSC OMR Marksheet Portal",

    layout="wide",

    page_icon="🔒"
)


st.title(
    "📝 BPSC OMR "
    "Marksheet Portal"
)

st.markdown(
    """
    <style>
    .main .block-container {max-width: 1180px; padding-top: 2rem;}
    .secure-card {
        border: 1px solid rgba(49, 51, 63, .18);
        border-radius: 18px;
        padding: 18px 20px 8px 20px;
        background: rgba(250, 250, 252, .72);
        margin-bottom: 8px;
    }
    .secure-badge {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 999px;
        font-size: .78rem;
        font-weight: 700;
        letter-spacing: .04em;
    }
    .result-label {font-size: .9rem; opacity: .72; margin-top: 12px;}
    .anon-id {font-size: 2rem; font-weight: 800; letter-spacing: .08em;
              margin: 0 0 12px 0;}
    .privacy-note {
        border-left: 4px solid #888;
        padding: 12px 16px;
        border-radius: 8px;
        background: rgba(128,128,128,.08);
    }
    .privacy-hero {
        border: 1px solid rgba(49, 51, 63, .16);
        border-radius: 22px;
        padding: 28px 28px 24px 28px;
        margin: 10px 0 20px 0;
        background: linear-gradient(135deg, rgba(245,248,255,.95), rgba(250,250,252,.95));
        text-align: center;
    }
    .privacy-lock {
        font-size: 2.4rem;
        margin-bottom: 4px;
    }
    .privacy-hero h2 {
        margin: 0;
        font-size: 1.8rem;
    }
    .privacy-lead {
        max-width: 760px;
        margin: 10px auto 0 auto;
        font-size: 1rem;
        line-height: 1.6;
        opacity: .82;
    }
    .privacy-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 14px;
        margin-bottom: 18px;
    }
    .privacy-box {
        border: 1px solid rgba(49, 51, 63, .14);
        border-radius: 16px;
        padding: 18px;
        background: rgba(250,250,252,.7);
    }
    .privacy-box h4 {
        margin: 4px 0 7px 0;
    }
    .privacy-box p {
        margin: 0;
        line-height: 1.55;
        opacity: .78;
        font-size: .92rem;
    }
    .privacy-icon {
        font-size: 1.55rem;
    }
    @media (max-width: 700px) {
        .privacy-grid {grid-template-columns: 1fr;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="privacy-hero">
        <div class="privacy-lock">🔒</div>
        <h2>Upload OMR → Get Your Marksheet</h2>
        <p class="privacy-lead">
            Upload your six OMR sheets below. The system reads them, checks the
            booklet series, calculates your marks and generates your result.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="privacy-grid">
        <div class="privacy-box">
            <div class="privacy-icon">🔐</div>
            <h4>Private & Secure</h4>
            <p>Your uploaded OMR files are processed for evaluation and are not stored in the ranking database.</p>
        </div>
        <div class="privacy-box">
            <div class="privacy-icon">⚡</div>
            <h4>Quick Result</h4>
            <p>Upload the required sheets, verify the booklet series, and generate your marksheet in one step.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.info(
    "📌 **Only 2 things to remember:** Upload your OMR sheets and select the "
    "correct booklet series printed on each sheet. Then tap **Generate Marksheet**."
)


query_params = st.query_params


url_public_id = query_params.get(
    "public_id",
    ""
)


# ============================================================
# TABS
# ============================================================

tabs = st.tabs([

    "📤 Student OMR Portal",

    "🔍 Rank Lookup",

    "🏆 Merit Rank List"
])


# ============================================================
# TAB 1
# ============================================================

with tabs[0]:

    st.subheader("📤 Upload Your OMR Sheets")

    st.markdown(
        """
        ### Follow these 3 simple steps
        **1. Upload** all 6 OMR response sheets (PDF/JPG/PNG).  
        **2. Select** the Question Booklet Series printed on each sheet.  
        **3. Tap** **🚀 Generate Marksheet** — your result is calculated automatically.
        """
    )

    st.caption(
        "Your roll number is read automatically from the OMR. You do not need to type it."
    )

    # The upload section is always enabled; identity comes only from OMR.
    # All candidate identity data is obtained from the uploaded sheets.

    st.write(

        "### 📂 Upload Your OMR "
        "Response Sheets & Select OMR Set Code"
    )


    omr_files = {}

    # The booklet series is intentionally NOT detected by OCR.
    # The user must manually select and confirm the booklet printed on each OMR.
    omr_sets = {}
    booklet_confirmed = {}


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

                omr_sets[code] = st.selectbox(
                    f"Select Question Booklet Series for {code}",
                    options=available_sets,
                    key=f"set_{code}",
                    help=(
                        "Look at the Question Booklet Series printed on this "
                        "response sheet and select the matching booklet here. "
                        "This selection is used to choose the answer key."
                    ),
                )

                booklet_confirmed[code] = st.checkbox(
                    f"I manually confirm that this response sheet is "
                    f"booklet {omr_sets[code]}",
                    key=f"confirm_booklet_{code}",
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


    uploaded_count = sum(1 for item in omr_files.values() if item is not None)
    confirmed_count = sum(1 for item in booklet_confirmed.values() if item)

    if uploaded_count == 6 and confirmed_count == 6:
        st.success("✅ All 6 OMR sheets are uploaded and all booklet series are confirmed. You are ready to generate your marksheet.")
    else:
        st.info(
            f"📋 Upload status: **{uploaded_count}/6 OMR sheets** • "
            f"Booklet confirmation: **{confirmed_count}/6**"
        )

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
        for code in SUBJECT_META:
            identity_results[code] = extract_omr_identity(
                omr_files[code]
            )

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

        db = load_data()

        normalized_roll_no = str(roll_no).strip()

        # If this roll number already exists, update that record instead of
        # creating another historical record. Keep the existing Anonymous ID.
        existing = next(
            (
                item for item in db
                if str(item.get("roll_no", "")).strip() == normalized_roll_no
            ),
            None
        )

        if existing:
            public_id = str(
                existing.get("public_id") or generate_unique_public_id(db)
            )
        else:
            public_id = generate_unique_public_id(db)

        new_entry = {
            "roll_no": normalized_roll_no,
            "public_id": public_id,
            "merit_total": round(merit_total, 2),
            "merit_max": round(merit_max, 2),
        }

        # Remove any old record for this roll number, then save only the
        # current result.
        db = [
            item for item in db
            if str(item.get("roll_no", "")).strip() != normalized_roll_no
        ]
        db.append(new_entry)

        # Current merit ranking: highest merit marks = rank 1.
        # Recalculate every rank after each new/update submission.
        db.sort(
            key=lambda x: float(x.get("merit_total", 0)),
            reverse=True
        )

        for rank_idx, record in enumerate(db, start=1):
            record["rank"] = rank_idx

        # save_data() strips the transient rank and any unexpected fields.
        save_data(db)

        st.balloons()

        saved_record = next(
            item for item in db
            if str(item.get("public_id", "")) == public_id
        )

        render_marksheet(
            saved_record,
            len(db)
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

with tabs[1]:

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
        db.sort(
            key=lambda x: float(x.get("merit_total", 0)),
            reverse=True
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

# ============================================================
# TAB 3
# ============================================================

with tabs[2]:

    st.subheader(
        "🏆 Live Merit Rank List"
    )
    st.caption(
        "Public view contains only Rank, Anonymous ID and Merit Marks. "
        "No roll numbers or OMR details are published."
    )

    all_db = load_data()

    if not all_db:
        st.info(
            "No merit records have been published yet."
        )
    else:
        # Current rank is always calculated from current merit marks.
        all_db.sort(
            key=lambda x: float(x.get("merit_total", 0)),
            reverse=True
        )

        for rank_idx, item in enumerate(all_db, start=1):
            item["rank"] = rank_idx

        save_data(all_db)

        rows = [
            {
                "Rank": item.get("rank", "N/A"),
                "Anonymous ID": item.get("public_id", "N/A"),
                "Merit Marks": (
                    f"{item.get('merit_total', 0)} / "
                    f"{item.get('merit_max', 0)}"
                ),
            }
            for item in all_db
        ]

        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            use_container_width=True
        )

