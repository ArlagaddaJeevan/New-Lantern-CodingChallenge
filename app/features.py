"""Feature extraction shared between training and inference.

The extractor is stateful only for the two TF-IDF vectorizers (character n-grams
and word n-grams over normalized study descriptions). Everything else is
deterministic text parsing with regex / keyword buckets, so the same module
produces identical features at train and serve time.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Iterable

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer


# --- normalization -----------------------------------------------------------

_PUNCT_RE = re.compile(r"[^a-z0-9/\s]+")
_WS_RE = re.compile(r"\s+")

# Curated synonym expansion so that "MAMMO", "MAMMOGRAPHY", "MAM" all collapse.
# Keys are matched as whole words (case-insensitive after lower()).
_SYNONYMS: dict[str, str] = {
    "mammo": "mam",
    "mammogram": "mam",
    "mammography": "mam",
    "ultrasound": "us",
    "doppler": "us",
    "xray": "xr",
    "x-ray": "xr",
    "radiograph": "xr",
    "radiographs": "xr",
    "rad": "xr",
    "mr": "mri",
    "mra": "mri",
    "cta": "ct",
    "ctap": "ct",
    "pet/ct": "petct",
    "dexa": "dxa",
    "wo": "without",
    "w/o": "without",
    "wo/w": "without with",
    "w": "with",
    "w/": "with",
    "cntrst": "contrast",
    "cnst": "contrast",
    "cntst": "contrast",
    "cntrast": "contrast",
    "con": "contrast",
    "bilat": "bilateral",
    "bi": "bilateral",
    "lt": "left",
    "rt": "right",
    "abd": "abdomen",
    "pelv": "pelvis",
    "cervicl": "cervical",
    "thorac": "thoracic",
    "lumb": "lumbar",
    "extrem": "extremity",
    "cor": "coronary",
    "cardiac": "heart",
    "myo": "myocardial",
    "perf": "perfusion",
    "le": "lowerextremity",
    "ue": "upperextremity",
    "lower": "lower",
    "upper": "upper",
    "angio": "angiography",
    "transthorac": "transthoracic",
    "screen": "screening",
    "dx": "diagnostic",
    "dxs": "diagnostic",
    "limited": "limited",
    "maxfacial": "maxillofacial",
    "maxfac": "maxillofacial",
    "sinuses": "sinus",
    "pulm": "pulmonary",
    "nasopharynx": "nasopharynx",
    "tte": "transthoracic",
    "mri/mra": "mri",
    "us/doppler": "us",
}


def normalize_desc(desc: str) -> str:
    """Lowercase, strip punctuation, expand common medical-imaging synonyms."""
    if not desc:
        return ""
    s = desc.lower()
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    tokens = [_SYNONYMS.get(t, t) for t in s.split()]
    # re-tokenize after synonym expansion (some expansions contain whitespace)
    return " ".join(" ".join(tokens).split())


# --- modality / anatomy / laterality / contrast -----------------------------

_MODALITY_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("MAM", re.compile(r"\bmam\b")),
    ("DXA", re.compile(r"\bdxa\b|\bbone density\b")),
    ("PETCT", re.compile(r"\bpetct\b|\bpet ct\b|\bpet/ct\b|\bpet\b")),
    ("NM", re.compile(r"\bnm\b|\bspect\b|\bmyocardial\b|\bscintigraphy\b")),
    ("ECHO", re.compile(r"\becho\b|\btransthoracic\b")),
    ("US", re.compile(r"\bus\b|\bvas\b|\bvenous\b")),
    ("MRI", re.compile(r"\bmri\b")),
    ("CT", re.compile(r"\bct\b")),
    ("FL", re.compile(r"\bfluoroscopy\b|\bfluoro\b|\bfl\b")),
    ("XR", re.compile(r"\bxr\b")),
    ("IR", re.compile(r"\bbiopsy\b|\bdrain\b|\baspiration\b|\binjection\b")),
]


def extract_modality(norm_desc: str) -> str:
    for name, pat in _MODALITY_PATTERNS:
        if pat.search(norm_desc):
            return name
    return "OTHER"


_ANATOMY_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("BREAST",   ("breast", "mam")),
    ("HEAD",     ("brain", "head", "skull", "cerebral", "intracranial")),
    ("FACE",     ("face", "maxillofacial", "sinus", "orbit", "nasopharynx")),
    ("NECK",     ("neck", "thyroid", "carotid", "parotid", "salivary")),
    ("CSPINE",   ("cervical",)),
    ("TSPINE",   ("thoracic spine", "tspine")),
    ("LSPINE",   ("lumbar", "lumbosacral", "sacral", "sacrum", "coccyx")),
    ("SPINE",    ("spine", "spinal")),
    ("CHEST",    ("chest", "thorax", "lung", "pulmonary", "airway", "ribs", "rib", "sternum", "mediastinal")),
    ("HEART",    ("heart", "coronary", "aortic valve", "mitral")),
    ("VASCULAR", ("angiography", "artery", "arterial", "vein", "vascular", "aorta", "aaa", "vena")),
    ("ABDOMEN",  ("abdomen", "liver", "kidney", "renal", "gall", "pancreas", "spleen", "bowel", "appendix", "stomach", "hepatic")),
    ("PELVIS",   ("pelvis", "prostate", "uterus", "ovary", "bladder", "scrotal", "testicular", "scrotum", "urinary")),
    ("SHOULDER", ("shoulder",)),
    ("UPPER",    ("humerus", "elbow", "forearm", "wrist", "hand", "finger", "arm", "clavicle", "upperextremity")),
    ("HIP",      ("hip", "pelvic girdle")),
    ("LOWER",    ("femur", "knee", "tibia", "fibula", "ankle", "foot", "toe", "leg", "calf", "lowerextremity")),
    ("WHOLE",    ("whole body", "skulltothigh", "skull to thigh", "wholebody")),
]


def extract_anatomy(norm_desc: str) -> str:
    for name, words in _ANATOMY_KEYWORDS:
        for w in words:
            if w in norm_desc:
                return name
    return "OTHER"


def extract_laterality(norm_desc: str) -> str:
    if "bilateral" in norm_desc:
        return "BI"
    if re.search(r"\bleft\b", norm_desc):
        return "LT"
    if re.search(r"\bright\b", norm_desc):
        return "RT"
    return "NONE"


def extract_contrast(norm_desc: str) -> str:
    has_without = "without" in norm_desc
    has_with = "with" in norm_desc and not re.search(r"\bwithout\b", norm_desc)
    both = "without" in norm_desc and re.search(r"\bwith\b(?! contrast)|wo/w|without with", norm_desc)
    if both:
        return "BOTH"
    if has_without:
        return "WO"
    if has_with:
        return "W"
    return "NONE"


# --- study struct ------------------------------------------------------------

@dataclass
class Study:
    study_id: str
    study_description: str
    study_date: str | None
    norm: str
    modality: str
    anatomy: str
    laterality: str
    contrast: str
    date_parsed: date | None

    @classmethod
    def from_dict(cls, d: dict) -> "Study":
        norm = normalize_desc(d.get("study_description") or "")
        dt_raw = d.get("study_date")
        try:
            dt = date.fromisoformat(dt_raw) if dt_raw else None
        except Exception:
            dt = None
        return cls(
            study_id=str(d.get("study_id") or ""),
            study_description=d.get("study_description") or "",
            study_date=dt_raw,
            norm=norm,
            modality=extract_modality(norm),
            anatomy=extract_anatomy(norm),
            laterality=extract_laterality(norm),
            contrast=extract_contrast(norm),
            date_parsed=dt,
        )


# --- stateful extractor ------------------------------------------------------

MODALITY_VOCAB = ["MR", "CT", "XR", "US", "MAM", "NM", "ECHO", "PETCT", "DXA", "FL", "IR", "OTHER", "MRI"]
ANATOMY_VOCAB = ["HEAD", "FACE", "NECK", "CSPINE", "TSPINE", "LSPINE", "SPINE", "CHEST", "HEART",
                 "BREAST", "VASCULAR", "ABDOMEN", "PELVIS", "SHOULDER", "UPPER", "HIP", "LOWER", "WHOLE", "OTHER"]
LATERALITY_VOCAB = ["NONE", "LT", "RT", "BI"]
CONTRAST_VOCAB = ["NONE", "WO", "W", "BOTH"]

NUMERIC_FEATURES = [
    "days_delta", "log_days", "within_365", "within_1825", "over_3650",
    "cosine_char", "cosine_word", "jaccard", "prefix_match",
    "exact_match", "mod_match", "anat_match", "lat_match", "contrast_match",
    "same_mod_anat", "desc_len_cur", "desc_len_prior", "desc_len_diff",
    "priors_per_case", "recency_rank", "is_most_recent",
]

CATEGORICAL_PREFIXES = [
    ("mod_cur",   MODALITY_VOCAB),
    ("mod_pri",   MODALITY_VOCAB),
    ("anat_cur",  ANATOMY_VOCAB),
    ("anat_pri",  ANATOMY_VOCAB),
    ("lat_cur",   LATERALITY_VOCAB),
    ("lat_pri",   LATERALITY_VOCAB),
    ("con_cur",   CONTRAST_VOCAB),
    ("con_pri",   CONTRAST_VOCAB),
]


def _build_feature_names() -> list[str]:
    names = list(NUMERIC_FEATURES)
    for prefix, vocab in CATEGORICAL_PREFIXES:
        for v in vocab:
            names.append(f"{prefix}={v}")
    return names


FEATURE_NAMES = _build_feature_names()


def _token_set(s: str) -> set[str]:
    return {t for t in s.split() if t}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _onehot(value: str, vocab: list[str]) -> list[int]:
    return [1 if value == v else 0 for v in vocab]


class FeatureExtractor:
    """Owns the fitted TF-IDF vectorizers. Stateless parsing is module-level."""

    def __init__(self, char_ngram=(3, 5), word_ngram=(1, 2),
                 char_max_features=30_000, word_max_features=10_000):
        self.char_vec = TfidfVectorizer(
            analyzer="char_wb", ngram_range=char_ngram,
            max_features=char_max_features, sublinear_tf=True,
        )
        self.word_vec = TfidfVectorizer(
            analyzer="word", ngram_range=word_ngram,
            max_features=word_max_features, sublinear_tf=True,
            token_pattern=r"(?u)\b\w+\b",
        )
        self._fitted = False

    def fit(self, descriptions: Iterable[str]) -> "FeatureExtractor":
        norms = [normalize_desc(d) for d in descriptions]
        self.char_vec.fit(norms)
        self.word_vec.fit(norms)
        self._fitted = True
        return self

    def _cosine_pair(self, matrix: sparse.csr_matrix, cur_idx: int, pri_idx: int) -> float:
        a = matrix[cur_idx]
        b = matrix[pri_idx]
        num = float(a.multiply(b).sum())
        denom = float(np.sqrt(a.multiply(a).sum()) * np.sqrt(b.multiply(b).sum()))
        return num / denom if denom > 0 else 0.0

    def featurize_case(self, current: Study, priors: list[Study]) -> np.ndarray:
        """Return a (len(priors), n_features) float32 matrix for one case."""
        if not priors:
            return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)

        texts = [current.norm] + [p.norm for p in priors]
        char_mat = self.char_vec.transform(texts)
        word_mat = self.word_vec.transform(texts)

        cur_tokens = _token_set(current.norm)
        cur_len = len(current.norm)
        n_priors = len(priors)

        # recency rank: 0 = oldest, n-1 = most recent (by date_parsed).
        priors_sorted = sorted(
            range(n_priors),
            key=lambda i: priors[i].date_parsed or date.min,
        )
        rank = [0] * n_priors
        for r, idx in enumerate(priors_sorted):
            rank[idx] = r

        rows = []
        for i, p in enumerate(priors):
            if current.date_parsed and p.date_parsed:
                delta = (current.date_parsed - p.date_parsed).days
            else:
                delta = 0
            delta = max(delta, 0)
            log_days = float(np.log1p(delta))
            w_365 = 1 if delta <= 365 else 0
            w_1825 = 1 if delta <= 365 * 5 else 0
            o_3650 = 1 if delta > 365 * 10 else 0

            cos_c = self._cosine_pair(char_mat, 0, i + 1)
            cos_w = self._cosine_pair(word_mat, 0, i + 1)
            pri_tokens = _token_set(p.norm)
            jac = _jaccard(cur_tokens, pri_tokens)
            prefix = 1 if current.norm[:3] and current.norm[:3] == p.norm[:3] else 0
            exact = 1 if current.norm == p.norm and current.norm else 0
            mm = 1 if current.modality == p.modality else 0
            am = 1 if current.anatomy == p.anatomy else 0
            lm = 1 if current.laterality == p.laterality else 0
            cm = 1 if current.contrast == p.contrast else 0
            sma = 1 if mm and am else 0
            pri_len = len(p.norm)

            numeric = [
                float(delta), log_days, float(w_365), float(w_1825), float(o_3650),
                cos_c, cos_w, jac, float(prefix),
                float(exact), float(mm), float(am), float(lm), float(cm),
                float(sma), float(cur_len), float(pri_len), float(abs(cur_len - pri_len)),
                float(n_priors), float(rank[i]), float(rank[i] == n_priors - 1),
            ]

            cat = []
            cat.extend(_onehot(current.modality, MODALITY_VOCAB))
            cat.extend(_onehot(p.modality, MODALITY_VOCAB))
            cat.extend(_onehot(current.anatomy, ANATOMY_VOCAB))
            cat.extend(_onehot(p.anatomy, ANATOMY_VOCAB))
            cat.extend(_onehot(current.laterality, LATERALITY_VOCAB))
            cat.extend(_onehot(p.laterality, LATERALITY_VOCAB))
            cat.extend(_onehot(current.contrast, CONTRAST_VOCAB))
            cat.extend(_onehot(p.contrast, CONTRAST_VOCAB))

            rows.append(numeric + cat)

        return np.asarray(rows, dtype=np.float32)
