import io
import json
import os
import re
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

# Safe imports for PDF processing, OCR, and image conversion
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

DATA_FILE = "bpsc_leaderboard_6papers.json"

# Option map for contour index
OPTION_MAP = {0: "A", 1: "B", 2: "C", 3: "D"}

# --- OFFICIAL BPSC ANSWER KEYS FOR ALL PAPERS & SETS ---
OFFICIAL_KEYS = {
    "P1": {  # General English
        "Set-I (Set-1)": {1: "C", 2: "C", 3: "Deleted", 4: "B", 5: "A", 6: "B", 7: "D", 8: "A", 9: "B", 10: "B", 11: "D", 12: "C", 13: "C", 14: "B", 15: "B", 16: "D", 17: "C", 18: "C", 19: "B", 20: "B", 21: "A", 22: "B", 23: "D", 24: "A", 25: "A", 26: "C", 27: "B", 28: "C", 29: "B", 30: "A", 31: "A", 32: "C", 33: "D", 34: "B", 35: "C", 36: "D", 37: "B", 38: "D", 39: "B", 40: "A", 41: "A", 42: "B", 43: "B", 44: "C", 45: "C", 46: "A", 47: "A", 48: "A", 49: "C", 50: "B"},
        "Set-J (Set-2)": {1: "Deleted", 2: "A", 3: "C", 4: "C", 5: "B", 6: "C", 7: "A", 8: "D", 9: "C", 10: "B", 11: "C", 12: "B", 13: "B", 14: "C", 15: "B", 16: "C", 17: "D", 18: "B", 19: "C", 20: "D", 21: "A", 22: "B", 23: "C", 24: "B", 25: "B", 26: "D", 27: "D", 28: "C", 29: "B", 30: "C", 31: "C", 32: "C", 33: "A", 34: "D", 35: "D", 36: "B", 37: "D", 38: "C", 39: "A", 40: "B", 41: "C", 42: "B", 43: "C", 44: "D", 45: "D", 46: "A", 47: "A", 48: "A", 49: "B", 50: "C"},
        "Set-K (Set-3)": {1: "A", 2: "A", 3: "D", 4: "B", 5: "C", 6: "C", 7: "A", 8: "A", 9: "Deleted", 10: "B", 11: "B", 12: "D", 13: "C", 14: "B", 15: "B", 16: "C", 17: "B", 18: "D", 19: "C", 20: "C", 21: "C", 22: "C", 23: "C", 24: "B", 25: "D", 26: "C", 27: "D", 28: "A", 29: "A", 30: "D", 31: "A", 32: "B", 33: "A", 34: "D", 35: "B", 36: "A", 37: "D", 38: "D", 39: "C", 40: "C", 41: "A", 42: "A", 43: "C", 44: "D", 45: "D", 46: "B", 47: "C", 48: "A", 49: "A", 50: "A"},
        "Set-L (Set-4)": {1: "D", 2: "A", 3: "B", 4: "B", 5: "B", 6: "Deleted", 7: "B", 8: "C", 9: "A", 10: "C", 11: "C", 12: "B", 13: "D", 14: "B", 15: "D", 16: "C", 17: "B", 18: "B", 19: "C", 20: "C", 21: "A", 22: "D", 23: "D", 24: "D", 25: "C", 26: "A", 27: "D", 28: "A", 29: "B", 30: "B", 31: "D", 32: "B", 33: "B", 34: "C", 35: "A", 36: "C", 37: "D", 38: "B", 39: "A", 40: "D", 41: "A", 42: "B", 43: "B", 44: "D", 45: "A", 46: "A", 47: "B", 48: "C", 49: "A", 50: "A"}
    },
    "P2": {  # General Hindi
        "Set-A": {1: "A", 2: "C", 3: "A", 4: "D", 5: "B", 6: "B", 7: "D", 8: "C", 9: "D", 10: "B", 11: "D", 12: "D", 13: "D", 14: "D", 15: "D", 16: "B", 17: "C", 18: "D", 19: "A", 20: "A", 21: "D", 22: "B", 23: "B", 24: "A", 25: "D", 26: "D", 27: "B", 28: "C", 29: "A", 30: "B", 31: "B", 32: "A", 33: "A", 34: "B", 35: "A", 36: "B", 37: "C", 38: "A", 39: "D", 40: "D", 41: "B", 42: "A", 43: "B", 44: "B", 45: "A", 46: "A", 47: "C", 48: "D", 49: "D", 50: "A"},
        "Set-B": {1: "B", 2: "B", 3: "D", 4: "D", 5: "B", 6: "B", 7: "C", 8: "A", 9: "B", 10: "C", 11: "D", 12: "D", 13: "B", 14: "B", 15: "D", 16: "C", 17: "D", 18: "B", 19: "A", 20: "D", 21: "C", 22: "B", 23: "D", 24: "A", 25: "A", 26: "C", 27: "C", 28: "C", 29: "C", 30: "C", 31: "C", 32: "D", 33: "A", 34: "B", 35: "B", 36: "D", 37: "C", 38: "A", 39: "C", 40: "D", 41: "D", 42: "D", 43: "B", 44: "C", 45: "C", 46: "D", 47: "B", 48: "C", 49: "C", 50: "A"},
        "Set-C": {1: "D", 2: "C", 3: "C", 4: "A", 5: "A", 6: "D", 7: "A", 8: "A", 9: "B", 10: "C", 11: "D", 12: "C", 13: "D", 14: "D", 15: "C", 16: "C", 17: "A", 18: "C", 19: "A", 20: "B", 21: "A", 22: "B", 23: "A", 24: "A", 25: "D", 26: "D", 27: "A", 28: "A", 29: "C", 30: "A", 31: "A", 32: "A", 33: "C", 34: "B", 35: "D", 36: "C", 37: "B", 38: "D", 39: "C", 40: "A", 41: "D", 42: "A", 43: "A", 44: "B", 45: "D", 46: "C", 47: "C", 48: "D", 49: "D", 50: "C"},
        "Set-D": {1: "D", 2: "C", 3: "A", 4: "B", 5: "B", 6: "A", 7: "A", 8: "D", 9: "D", 10: "A", 11: "D", 12: "D", 13: "A", 14: "B", 15: "B", 16: "A", 17: "B", 18: "B", 19: "A", 20: "C", 21: "D", 22: "A", 23: "D", 24: "D", 25: "A", 26: "D", 27: "C", 28: "A", 29: "D", 30: "B", 31: "D", 32: "B", 33: "C", 34: "B", 35: "D", 36: "B", 37: "B", 38: "A", 39: "B", 40: "B", 41: "B", 42: "B", 43: "C", 44: "A", 45: "D", 46: "D", 47: "B", 48: "C", 49: "A", 50: "A"}
    },
    "P3": {  # General Studies III
        "Set-E": {1: "A", 2: "B", 3: "D", 4: "B", 5: "C", 6: "A", 7: "B", 8: "D", 9: "D", 10: "B", 11: "A", 12: "A", 13: "A", 14: "D", 15: "A", 16: "C", 17: "A", 18: "A", 19: "A", 20: "B", 21: "A", 22: "A", 23: "D", 24: "C", 25: "C", 26: "C", 27: "C", 28: "D", 29: "C", 30: "B", 31: "A", 32: "B", 33: "D", 34: "B", 35: "A", 36: "C", 37: "C", 38: "C", 39: "C", 40: "D", 41: "Deleted", 42: "Deleted", 43: "A", 44: "D", 45: "A", 46: "B", 47: "C", 48: "A", 49: "B", 50: "D"},
        "Set-F": {1: "A", 2: "A", 3: "C", 4: "C", 5: "B", 6: "C", 7: "A", 8: "B", 9: "B", 10: "A", 11: "A", 12: "B", 13: "A", 14: "A", 15: "C", 16: "B", 17: "Deleted", 18: "B", 19: "Deleted", 20: "C", 21: "C", 22: "D", 23: "D", 24: "C", 25: "C", 26: "D", 27: "C", 28: "B", 29: "C", 30: "D", 31: "A", 32: "C", 33: "D", 34: "D", 35: "B", 36: "C", 37: "A", 38: "C", 39: "C", 40: "C", 41: "D", 42: "B", 43: "B", 44: "A", 45: "B", 46: "D", 47: "C", 48: "D", 49: "D", 50: "C"},
        "Set-G": {1: "C", 2: "D", 3: "D", 4: "B", 5: "D", 6: "A", 7: "D", 8: "B", 9: "B", 10: "B", 11: "A", 12: "B", 13: "C", 14: "B", 15: "A", 16: "C", 17: "B", 18: "A", 19: "C", 20: "Deleted", 21: "B", 22: "A", 23: "C", 24: "B", 25: "D", 26: "C", 27: "A", 28: "B", 29: "D", 30: "B", 31: "D", 32: "C", 33: "A", 34: "D", 35: "D", 36: "B", 37: "A", 38: "Deleted", 39: "D", 40: "D", 41: "B", 42: "B", 43: "C", 44: "D", 45: "B", 46: "C", 47: "A", 48: "C", 49: "B", 50: "B"},
        "Set-H": {1: "A", 2: "D", 3: "D", 4: "C", 5: "D", 6: "A", 7: "A", 8: "C", 9: "B", 10: "C", 11: "A", 12: "Deleted", 13: "Deleted", 14: "A", 15: "B", 16: "D", 17: "A", 18: "D", 19: "C", 20: "D", 21: "A", 22: "D", 23: "C", 24: "A", 25: "B", 26: "A", 27: "D", 28: "A", 29: "B", 30: "D", 31: "C", 32: "B", 33: "D", 34: "D", 35: "C", 36: "D", 37: "D", 38: "D", 39: "C", 40: "C", 41: "B", 42: "B", 43: "D", 44: "C", 45: "B", 46: "A", 47: "B", 48: "A", 49: "D", 50: "D"}
    },
    "P4": {  # General Engineering Science IV
        "Set-A": {1: "A", 2: "Deleted", 3: "C", 4: "C", 5: "D", 6: "B", 7: "A", 8: "C", 9: "D", 10: "B", 11: "B", 12: "C", 13: "C", 14: "B", 15: "D", 16: "B", 17: "C", 18: "D", 19: "B", 20: "A", 21: "B", 22: "C", 23: "A", 24: "C", 25: "B", 26: "A", 27: "Deleted", 28: "D", 29: "A", 30: "A", 31: "A", 32: "A", 33: "B", 34: "D", 35: "C", 36: "D", 37: "D", 38: "A", 39: "B", 40: "B", 41: "B", 42: "A", 43: "Deleted", 44: "B", 45: "B", 46: "B", 47: "D", 48: "C", 49: "B", 50: "C"},
        "Set-B": {1: "A", 2: "D", 3: "C", 4: "Deleted", 5: "A", 6: "D", 7: "D", 8: "D", 9: "D", 10: "B", 11: "B", 12: "D", 13: "A", 14: "A", 15: "B", 16: "A", 17: "D", 18: "C", 19: "D", 20: "B", 21: "C", 22: "A", 23: "Deleted", 24: "D", 25: "C", 26: "D", 27: "B", 28: "D", 29: "B", 30: "D", 31: "D", 32: "A", 33: "D", 34: "C", 35: "D", 36: "A", 37: "D", 38: "B", 39: "C", 40: "C", 41: "B", 42: "A", 43: "D", 44: "B", 45: "C", 46: "B", 47: "C", 48: "C", 49: "C", 50: "Deleted"},
        "Set-C": {1: "A", 2: "A", 3: "C", 4: "D", 5: "D", 6: "C", 7: "B", 8: "C", 9: "C", 10: "D", 11: "A", 12: "C", 13: "A", 14: "C", 15: "A", 16: "Deleted", 17: "A", 18: "A", 19: "D", 20: "C", 21: "D", 22: "A", 23: "D", 24: "B", 25: "B", 26: "A", 27: "B", 28: "Deleted", 29: "D", 30: "B", 31: "D", 32: "B", 33: "A", 34: "A", 35: "C", 36: "B", 37: "B", 38: "C", 39: "D", 40: "A", 41: "A", 42: "C", 43: "A", 44: "D", 45: "C", 46: "D", 47: "A", 48: "A", 49: "Deleted", 50: "D"},
        "Set-D": {1: "D", 2: "D", 3: "B", 4: "C", 5: "D", 6: "Deleted", 7: "A", 8: "D", 9: "B", 10: "A", 11: "C", 12: "D", 13: "D", 14: "B", 15: "B", 16: "B", 17: "Deleted", 18: "C", 19: "D", 20: "B", 21: "D", 22: "A", 23: "B", 24: "A", 25: "Deleted", 26: "B", 27: "D", 28: "D", 29: "C", 30: "C", 31: "C", 32: "C", 33: "D", 34: "C", 35: "D", 36: "A", 37: "D", 38: "B", 39: "A", 40: "B", 41: "D", 42: "A", 43: "D", 44: "A", 45: "A", 46: "A", 47: "D", 48: "B", 49: "B", 50: "D"}
    },
    "P5": {  # Civil Engineering V
        "Set-I": {1: "B", 2: "D", 3: "C", 4: "A", 5: "B", 6: "A", 7: "A", 8: "D", 9: "D", 10: "D", 11: "B", 12: "B", 13: "C", 14: "B", 15: "B", 16: "C", 17: "A", 18: "A", 19: "D", 20: "B", 21: "B", 22: "A", 23: "C", 24: "A", 25: "A", 26: "D", 27: "C", 28: "B", 29: "D", 30: "C", 31: "D", 32: "B", 33: "B", 34: "D", 35: "B", 36: "D", 37: "D", 38: "A", 39: "B", 40: "D", 41: "A", 42: "C", 43: "A", 44: "B", 45: "A", 46: "A", 47: "D", 48: "C", 49: "B", 50: "Deleted"},
        "Set-J": {1: "B", 2: "B", 3: "C", 4: "C", 5: "D", 6: "B", 7: "A", 8: "A", 9: "A", 10: "C", 11: "A", 12: "C", 13: "D", 14: "B", 15: "B", 16: "B", 17: "C", 18: "B", 19: "A", 20: "C", 21: "D", 22: "D", 23: "A", 24: "B", 25: "Deleted", 26: "D", 27: "D", 28: "C", 29: "D", 30: "B", 31: "D", 32: "A", 33: "A", 34: "D", 35: "A", 36: "C", 37: "A", 38: "B", 39: "B", 40: "C", 41: "C", 42: "C", 43: "A", 44: "D", 45: "C", 46: "B", 47: "B", 48: "D", 49: "B", 50: "C"},
        "Set-K": {1: "B", 2: "B", 3: "C", 4: "B", 5: "C", 6: "C", 7: "B", 8: "C", 9: "A", 10: "A", 11: "D", 12: "D", 13: "A", 14: "A", 15: "B", 16: "B", 17: "D", 18: "C", 19: "B", 20: "A", 21: "C", 22: "A", 23: "A", 24: "A", 25: "C", 26: "C", 27: "B", 28: "D", 29: "D", 30: "D", 31: "A", 32: "B", 33: "D", 34: "B", 35: "A", 36: "B", 37: "C", 38: "B", 39: "B", 40: "B", 41: "C", 42: "B", 43: "D", 44: "A", 45: "Deleted", 46: "C", 47: "A", 48: "A", 49: "C", 50: "A"},
        "Set-L": {1: "A", 2: "D", 3: "D", 4: "D", 5: "A", 6: "A", 7: "B", 8: "C", 9: "B", 10: "C", 11: "D", 12: "A", 13: "B", 14: "C", 15: "A", 16: "A", 17: "B", 18: "A", 19: "B", 20: "C", 21: "B", 22: "D", 23: "A", 24: "D", 25: "A", 26: "B", 27: "D", 28: "C", 29: "A", 30: "C", 31: "C", 32: "C", 33: "D", 34: "B", 35: "A", 36: "D", 37: "D", 38: "B", 39: "A", 40: "D", 41: "C", 42: "A", 43: "D", 44: "C", 45: "C", 46: "C", 47: "A", 48: "A", 49: "Deleted", 50: "D"}
    },
    "P6": {  # Civil Engineering VI
        "Set-E": {1: "D", 2: "A", 3: "B", 4: "A", 5: "Deleted", 6: "Deleted", 7: "B", 8: "Deleted", 9: "A", 10: "Deleted", 11: "D", 12: "A", 13: "A", 14: "B", 15: "B", 16: "A", 17: "D", 18: "A", 19: "Deleted", 20: "D", 21: "C", 22: "Deleted", 23: "D", 24: "D", 25: "D", 26: "D", 27: "D", 28: "C", 29: "A", 30: "B", 31: "D", 32: "C", 33: "C", 34: "A", 35: "D", 36: "B", 37: "A", 38: "A", 39: "C", 40: "C", 41: "D", 42: "A", 43: "B", 44: "B", 45: "A", 46: "A", 47: "A", 48: "A", 49: "C", 50: "Deleted"},
        "Set-F": {1: "Deleted", 2: "B", 3: "B", 4: "C", 5: "B", 6: "B", 7: "Deleted", 8: "B", 9: "C", 10: "A", 11: "D", 12: "C", 13: "B", 14: "C", 15: "B", 16: "Deleted", 17: "A", 18: "A", 19: "D", 20: "B", 21: "C", 22: "D", 23: "C", 24: "C", 25: "D", 26: "Deleted", 27: "A", 28: "Deleted", 29: "Deleted", 30: "C", 31: "D", 32: "D", 33: "B", 34: "A", 35: "C", 36: "A", 37: "B", 38: "Deleted", 39: "B", 40: "D", 41: "A", 42: "B", 43: "C", 44: "C", 45: "A", 46: "B", 47: "C", 48: "B", 49: "D", 50: "B"},
        "Set-G": {1: "D", 2: "D", 3: "Deleted", 4: "C", 5: "D", 6: "Deleted", 7: "D", 8: "D", 9: "C", 10: "A", 11: "D", 12: "C", 13: "C", 14: "D", 15: "B", 16: "B", 17: "D", 18: "A", 19: "D", 20: "D", 21: "C", 22: "A", 23: "C", 24: "B", 25: "Deleted", 26: "A", 27: "B", 28: "A", 29: "C", 30: "A", 31: "A", 32: "A", 33: "A", 34: "A", 35: "Deleted", 36: "Deleted", 37: "Deleted", 38: "C", 39: "A", 40: "B", 41: "D", 42: "B", 43: "B", 44: "D", 45: "C", 46: "D", 47: "A", 48: "D", 49: "D", 50: "Deleted"},
        "Set-H": {1: "D", 2: "B", 3: "D", 4: "C", 5: "C", 6: "B", 7: "A", 8: "Deleted", 9: "D", 10: "A", 11: "Deleted", 12: "C", 13: "C", 14: "Deleted", 15: "D", 16: "C", 17: "Deleted", 18: "A", 19: "B", 20: "Deleted", 21: "A", 22: "B", 23: "D", 24: "C", 25: "C", 26: "C", 27: "D", 28: "D", 29: "D", 30: "C", 31: "B", 32: "A", 33: "D", 34: "D", 35: "B", 36: "Deleted", 37: "Deleted", 38: "C", 39: "B", 40: "D", 41: "C", 42: "C", 43: "B", 44: "B", 45: "A", 46: "C", 47: "D", 48: "C", 49: "A", 50: "C"}
    }
}

SUBJECT_META = {
    "P1": {"name": "General English", "type": "qualifying", "cutoff_pct": 30.0},
    "P2": {"name": "General Hindi", "type": "qualifying", "cutoff_pct": 30.0},
    "P3": {"name": "General Studies III", "type": "merit"},
    "P4": {"name": "General Engineering Science IV", "type": "merit"},
    "P5": {"name": "Civil Engineering V", "type": "merit"},
    "P6": {"name": "Civil Engineering VI", "type": "merit"},
}


class OMREvaluationEngine:

    def __init__(self, answer_key, marking_scheme=None):
        """
        Initializes the evaluation engine with an answer key and default marking scheme.
        Default marking scheme: +2 for correct, 0 for wrong/unattempted, +2 for deleted questions.
        """
        self.answer_key = answer_key
        self.marking_scheme = marking_scheme or {
            "positive": 2.0,
            "negative": 0.0,
            "bonus": 2.0,
        }

    def evaluate_responses(self, student_responses):
        """
        Evaluates candidate's student responses against the official answer key.
        Performs itemized breakdown and total score calculation.
        """
        correct, wrong, skipped, deleted = 0, 0, 0, 0
        score = 0.0
        response_breakdown = []

        for q_no in range(1, 51):
            q_key = f"Q{q_no}" if f"Q{q_no}" in self.answer_key else q_no
            official_ans = self.answer_key.get(q_key, "")
            user_ans = student_responses.get(q_no, student_responses.get(str(q_no), ""))

            if str(official_ans).strip().lower() == "deleted":
                deleted += 1
                status = "Deleted (+2 Bonus)"
                marks_awarded = self.marking_scheme["bonus"]
                score += marks_awarded
            elif not user_ans or str(user_ans).strip() == "" or str(user_ans).strip().lower() == "unattempted":
                skipped += 1
                status = "Unattempted (0 Marks)"
                marks_awarded = 0.0
            elif str(user_ans).strip().upper() == str(official_ans).strip().upper():
                correct += 1
                status = "Correct (+2 Marks)"
                marks_awarded = self.marking_scheme["positive"]
                score += marks_awarded
            else:
                wrong += 1
                status = "Incorrect (0 Marks)"
                marks_awarded = -self.marking_scheme["negative"]
                score += marks_awarded

            response_breakdown.append({
                "Q. No": q_no,
                "Your Answer": user_ans if user_ans else "Unattempted",
                "Official Key": official_ans,
                "Result": status,
                "Marks": marks_awarded
            })

        max_marks = len(self.answer_key) * self.marking_scheme["positive"]
        bonus_marks = deleted * self.marking_scheme["bonus"]
        pct = round((score / max_marks) * 100, 2) if max_marks > 0 else 0.0

        return {
            "correct": correct,
            "wrong": wrong,
            "skipped": skipped,
            "deleted_count": deleted,
            "bonus_marks": bonus_marks,
            "score": round(score, 2),
            "max_marks": max_marks,
            "pct": pct,
            "itemized_breakdown": response_breakdown
        }


def load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    return []


def save_data(db):
    with open(DATA_FILE, "w") as f:
        json.dump(db, f, indent=4)


def process_image_cv_omr(pil_img):
    """
    OpenCV computer vision fallback to detect marked bubbles on image OMRs.
    """
    responses = {}
    try:
        open_cv_image = np.array(pil_img.convert("RGB"))
        gray = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bubble_contours = []

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            aspect_ratio = w / float(h)
            if 12 <= w <= 60 and 12 <= h <= 60 and 0.75 <= aspect_ratio <= 1.25:
                bubble_contours.append((x, y, w, h, c))

        bubble_contours = sorted(bubble_contours, key=lambda b: (b[1] // 20, b[0]))

        for q in range(1, 51):
            q_bubbles = bubble_contours[(q - 1) * 4 : q * 4]
            if len(q_bubbles) == 4:
                pixel_counts = []
                for (x, y, w, h, _) in q_bubbles:
                    mask = np.zeros(thresh.shape, dtype="uint8")
                    cv2.circle(mask, (x + w // 2, y + h // 2), int(w // 2), 255, -1)
                    mask = cv2.bitwise_and(thresh, thresh, mask=mask)
                    pixel_counts.append(cv2.countNonZero(mask))
                
                max_idx = np.argmax(pixel_counts)
                if pixel_counts[max_idx] > 100:
                    responses[q] = OPTION_MAP[max_idx]
    except Exception:
        pass
    return responses


def parse_omr_file(uploaded_file):
    """
    Reads OMR responses from PDFs or Images, combining text-parsing, OCR,
    and OpenCV vision bubble detection.
    """
    responses = {}
    if uploaded_file is None:
        return responses

    extracted_text = ""
    file_bytes = uploaded_file.getvalue()
    file_ext = uploaded_file.name.split(".")[-1].lower()
    pil_images = []

    try:
        # Step 1: Extract Text & Prepare Images from PDF or Image
        if file_ext in ["jpg", "jpeg", "png"]:
            img = Image.open(io.BytesIO(file_bytes))
            pil_images.append(img)
            if pytesseract is not None:
                extracted_text = pytesseract.image_to_string(img)

        elif file_ext == "pdf":
            if pdfplumber is not None:
                with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                    for page in pdf.pages:
                        text = page.extract_text()
                        if text:
                            extracted_text += text + "\n"

            # Fallback to OCR if pdfplumber returns empty text (scanned page)
            if convert_from_bytes is not None:
                pil_images = convert_from_bytes(file_bytes)
                if not extracted_text.strip() and pytesseract is not None:
                    for img in pil_images:
                        extracted_text += pytesseract.image_to_string(img) + "\n"

        # Step 2: Extract Answers from Pipe-Separated Header/Value Blocks
        lines = [line.strip() for line in extracted_text.splitlines() if line.strip()]
        for i in range(len(lines) - 1):
            q_nums = re.findall(r'(?:^|\||\s)(\d{1,2})(?=\s*\||\s*$)', lines[i])
            answers = re.findall(r'(?:^|\||\s)([A-Da-d])(?=\s*\||\s*$)', lines[i + 1])

            if len(q_nums) > 1 and len(q_nums) == len(answers):
                for q_str, ans in zip(q_nums, answers):
                    q_num = int(q_str)
                    if 1 <= q_num <= 50:
                        responses[q_num] = ans.upper()

        # Step 3: Fallback Regex for Standard Formats ("1: A", "1. A", "1 - A")
        if not responses:
            standard_matches = re.findall(r'(?:^|\b|\s)(\d{1,2})[\s\.\:\-\|]+([A-Da-d])(?:\b|\s|$)', extracted_text)
            for q_str, ans in standard_matches:
                q_num = int(q_str)
                if 1 <= q_num <= 50:
                    responses[q_num] = ans.upper()

        # Step 4: OpenCV Bubble Contour Detection (Fallback if text extraction failed)
        if not responses and pil_images:
            responses = process_image_cv_omr(pil_images[0])

    except Exception as e:
        st.error(f"Error reading {uploaded_file.name}: {str(e)}")

    return responses


def evaluate_paper(student_responses, official_key):
    """
    Evaluates paper using the integrated OMREvaluationEngine module.
    """
    engine = OMREvaluationEngine(official_key)
    return engine.evaluate_responses(student_responses)


def render_marksheet(record, total_candidates):
    st.markdown("---")
    st.subheader(f"📄 BPSC Official Scorecard: {record['name']} (Roll No: {record['roll_no']})")

    if record["qualified"]:
        st.success("🎉 STATUS: QUALIFIED (Passed English & Hindi Qualifying Cutoff >= 30%)")
    else:
        st.error("❌ STATUS: DISQUALIFIED (Failed English or Hindi Qualifying Cutoff of 30%)")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Overall Rank", f"#{record.get('rank', 'N/A')} / {total_candidates}")
    m2.metric("Merit Score (P3-P6)", f"{record['merit_total']} / {record['merit_max']}")
    m3.metric("Merit Percentage", f"{record['merit_pct']}%")
    m4.metric("Qualifying Status", "Passed" if record["qualified"] else "Failed")

    st.write("### 📊 Subject-wise Scorecard Breakdown")

    table_data = []
    for code, p in record["papers"].items():
        status = "Pass" if p["passed"] else "Fail"
        if p["type"] == "merit":
            status = "Merit Subject"

        table_data.append({
            "Paper Code": code,
            "Subject Name": p["subject_name"],
            "Selected OMR Set": p["set_name"],
            "Correct (+2)": p["correct"],
            "Wrong (0)": p["wrong"],
            "Unattempted": p["skipped"],
            "Deleted Questions": f"{p['deleted_count']} Qs (+{p['bonus_marks']} Bonus)",
            "Total Score": f"{p['score']} / {p['max_marks']}",
            "Percentage": f"{p['pct']}%",
            "Status": status
        })

    st.dataframe(pd.DataFrame(table_data), hide_index=True, use_container_width=True)

    st.write("### 🔍 Detailed Question-by-Question Response Audit")
    for code, p in record["papers"].items():
        with st.expander(f"Inspect Responses for {code}: {p['subject_name']} ({p['set_name']})", expanded=False):
            df_itemized = pd.DataFrame(p["itemized_breakdown"])
            st.dataframe(df_itemized, hide_index=True, use_container_width=True)

    st.write("### 🔗 Shareable Direct Scorecard Link")
    base_url = st.query_params.get("base_url", "http://localhost:8501")
    direct_link = f"{base_url}?roll_no={record['roll_no']}"
    st.code(direct_link, language="text")


# --- UI LAYOUT ---
st.set_page_config(page_title="BPSC OMR Evaluator & Merit Leaderboard", layout="wide", page_icon="📝")
st.title("📝 BPSC Student OMR Marksheet Generator & Merit Portal")

query_params = st.query_params
url_roll_no = query_params.get("roll_no", None)

tabs = st.tabs(["📤 Student OMR Portal", "🔍 Direct Scorecard Lookup", "🏆 Live Merit Leaderboard"])

# --- TAB 1: OMR Submission Portal ---
with tabs[0]:
    st.subheader("Candidate Identity & OMR Sheet Submission")

    c1, c2 = st.columns(2)
    with c1:
        student_name = st.text_input("Candidate Full Name*", placeholder="e.g. Rahul Kumar")
    with c2:
        roll_no = st.text_input("Roll Number / Registration No.*", placeholder="e.g. 10843")

    st.divider()

    if not student_name.strip() or not roll_no.strip():
        st.warning("⚠️ Enter Candidate Name and Roll Number above to enable OMR uploads.")
    else:
        st.write("### 📂 Upload Your OMR Response Sheets & Select OMR Set Code")

        omr_files = {}
        omr_sets = {}

        for code, meta in SUBJECT_META.items():
            category_tag = "Qualifying Paper (Min 30%)" if meta["type"] == "qualifying" else "Merit Paper"
            available_sets = list(OFFICIAL_KEYS[code].keys())

            with st.expander(f"📄 {code}: {meta['name']} [{category_tag}]", expanded=False):
                col_s, col_f = st.columns([1, 2])
                with col_s:
                    omr_sets[code] = st.selectbox(
                        f"Select OMR Set for {code}",
                        options=available_sets,
                        key=f"set_{code}"
                    )
                with col_f:
                    omr_files[code] = st.file_uploader(
                        f"Upload {code} OMR Sheet (PDF / JPG / PNG)",
                        type=["pdf", "jpg", "jpeg", "png"],
                        key=f"omr_{code}"
                    )

        st.divider()

        if st.button("🚀 Calculate Scores & Generate Marksheet", type="primary"):
            paper_results = {}
            merit_total = 0
            merit_max = 0

            for code, meta in SUBJECT_META.items():
                selected_set = omr_sets[code]
                official_key = OFFICIAL_KEYS[code][selected_set]
                parsed_responses = parse_omr_file(omr_files.get(code))

                eval_res = evaluate_paper(parsed_responses, official_key)

                is_passed = True
                if meta["type"] == "qualifying":
                    is_passed = eval_res["pct"] >= meta["cutoff_pct"]
                else:
                    merit_total += eval_res["score"]
                    merit_max += eval_res["max_marks"]

                eval_res["passed"] = is_passed
                eval_res["subject_name"] = meta["name"]
                eval_res["type"] = meta["type"]
                eval_res["set_name"] = selected_set
                paper_results[code] = eval_res

            is_qualified = paper_results["P1"]["passed"] and paper_results["P2"]["passed"]
            merit_pct = round((merit_total / merit_max) * 100, 2) if merit_max > 0 else 0.0

            db = load_data()
            db = [item for item in db if str(item["roll_no"]).strip() != str(roll_no).strip()]

            new_entry = {
                "name": student_name,
                "roll_no": roll_no,
                "papers": paper_results,
                "merit_total": merit_total,
                "merit_max": merit_max,
                "merit_pct": merit_pct,
                "qualified": is_qualified
            }
            db.append(new_entry)

            db.sort(key=lambda x: (x["qualified"], x["merit_total"]), reverse=True)

            for rank_idx, record in enumerate(db, start=1):
                record["rank"] = rank_idx

            save_data(db)

            st.balloons()
            saved_record = next(item for item in db if str(item["roll_no"]) == str(roll_no))
            render_marksheet(saved_record, len(db))

# --- TAB 2: Direct Scorecard Lookup ---
with tabs[1]:
    st.subheader("🔍 Search Candidate Scorecard")
    search_roll = st.text_input(
        "Enter Candidate Roll Number:",
        value=url_roll_no if url_roll_no else "",
        placeholder="e.g. 10843"
    )

    if search_roll.strip():
        db = load_data()
        record = next((item for item in db if str(item["roll_no"]).strip().lower() == str(search_roll).strip().lower()), None)
        if record:
            render_marksheet(record, len(db))
        else:
            st.error(f"No marksheets found for Roll Number: `{search_roll}`")

# --- TAB 3: Merit Leaderboard ---
with tabs[2]:
    st.subheader("🏆 Official Live Merit Leaderboard")
    db = load_data()

    if not db:
        st.info("No candidates have uploaded their OMR responses yet.")
    else:
        rows = []
        for item in db:
            p1 = item["papers"]["P1"]
            p2 = item["papers"]["P2"]
            rows.append({
                "Rank": item.get("rank", "N/A"),
                "Candidate Name": item["name"],
                "Roll Number": item["roll_no"],
                "Status": "Qualified" if item["qualified"] else "Disqualified",
                "English (P1) Marks": f"{p1['score']} ({'Pass' if p1['passed'] else 'Fail'})",
                "Hindi (P2) Marks": f"{p2['score']} ({'Pass' if p2['passed'] else 'Fail'})",
                "Merit Total (P3-P6)": f"{item['merit_total']} / {item['merit_max']}",
                "Merit Percentage": f"{item['merit_pct']}%"
            })

        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
