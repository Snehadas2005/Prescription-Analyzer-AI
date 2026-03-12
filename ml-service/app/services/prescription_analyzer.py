from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import re
import uuid
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np
from PIL import Image
import pytesseract
import cohere

cv2.setNumThreads(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

if not hasattr(Image, "ANTIALIAS"):
    Image.ANTIALIAS = Image.LANCZOS

MAX_IMAGE_PX   = 2000   # raised: handwriting needs more resolution than printed text
USE_EASYOCR    = os.getenv("USE_EASYOCR", "false").lower() == "true"
TESSERACT_PSM  = "--oem 3 --psm 6"   # primary: uniform block
TESSERACT_PSM2 = "--oem 3 --psm 11"  # fallback: sparse text (good for handwriting)


@dataclass
class Patient:
    name: str = ""
    age: str = ""
    gender: str = ""


@dataclass
class Doctor:
    name: str = ""
    specialization: str = ""
    registration_number: str = ""


@dataclass
class Medicine:
    name: str = ""
    dosage: str = ""
    quantity: str = "1"
    frequency: str = ""
    duration: str = ""
    instructions: str = ""
    available: bool = True


@dataclass
class AnalysisResult:
    prescription_id: str
    patient: Patient
    doctor: Doctor
    medicines: List[Medicine]
    diagnosis: List[str]
    confidence_score: float
    raw_text: str
    success: bool = True
    error: str = ""


class PrescriptionAnalyzer:
    _MEDICINE_DB: Dict[str, bool] = {
        "augmentin": True, "amoxicillin": True, "azithromycin": True,
        "ciprofloxacin": True, "cephalexin": True, "doxycycline": True,
        "clarithromycin": True, "paracetamol": True, "acetaminophen": True,
        "ibuprofen": True, "diclofenac": True, "aspirin": True,
        "naproxen": True, "crocin": True, "combiflam": True, "dolo": True,
        "esomeprazole": True, "omeprazole": True, "pantoprazole": True,
        "lansoprazole": True, "ranitidine": False, "cetirizine": True,
        "loratadine": True, "fexofenadine": True, "metformin": True,
        "insulin": True, "glimepiride": True, "vitamin d3": True,
        "vitamin b12": True, "montelukast": True, "losartan": True,
        "amlodipine": True, "atorvastatin": True, "metoprolol": True,
        "salbutamol": True, "prednisolone": True,
    }

    def __init__(self, cohere_api_key: Optional[str] = None):
        self._init_cohere(cohere_api_key)
        self._easyocr = None
        if USE_EASYOCR:
            self._init_easyocr()

    def _init_cohere(self, api_key: Optional[str]) -> None:
        key = api_key or os.getenv("COHERE_API_KEY")
        if not key:
            try:
                from integration.keys import COHERE_API_KEY  # type: ignore
                key = COHERE_API_KEY
            except ImportError:
                pass
        self._cohere: Optional[cohere.Client] = None
        if key:
            try:
                self._cohere = cohere.Client(key)
                logger.info("Cohere client ready")
            except Exception as exc:
                logger.warning("Cohere init failed: %s", exc)

    def _init_easyocr(self) -> None:
        try:
            import easyocr
            self._easyocr = easyocr.Reader(["en"], gpu=False, workers=0)
            logger.info("EasyOCR ready (workers=0)")
        except Exception as exc:
            logger.warning("EasyOCR unavailable: %s", exc)

    def analyze_prescription(self, image_path: str) -> AnalysisResult:
        pid = f"RX{datetime.now().strftime('%Y%m%d%H%M%S')}{uuid.uuid4().hex[:6].upper()}"
        logger.info("Analysing %s [id=%s]", image_path, pid)
        try:
            img = self._load_and_resize(image_path)
            if img is None:
                return self._error(pid, "Could not read image")

            processed = self._preprocess(img)
            raw_text, ocr_conf = self._ocr(processed)
            if not raw_text.strip():
                return self._error(pid, "No text extracted from image")

            logger.info("OCR: %d chars, conf=%.2f", len(raw_text), ocr_conf)
            text = self._clean(raw_text)

            doctor_d, patient_d = self._pattern_extract(text)
            medicines = self._extract_medicines(text)
            diagnosis: List[str] = []

            used_llm = False
            llm_data = self._llm_extract(text)
            if llm_data:
                used_llm = True
                self._merge_llm(llm_data, patient_d, doctor_d)
                if llm_data.get("medicines"):
                    medicines = self._medicines_from_llm(llm_data["medicines"])
                diagnosis = llm_data.get("diagnosis", [])

            patient = Patient(**{k: patient_d.get(k, "") for k in ("name", "age", "gender")})
            doctor = Doctor(
                name=doctor_d.get("name", ""),
                specialization=doctor_d.get("specialization", ""),
                registration_number=doctor_d.get("registration_number", ""),
            )
            confidence = self._confidence(ocr_conf, len(medicines), patient, doctor, used_llm)
            logger.info("Done — patient=%r medicines=%d conf=%.0f%%", patient.name, len(medicines), confidence * 100)
            return AnalysisResult(
                prescription_id=pid, patient=patient, doctor=doctor,
                medicines=medicines, diagnosis=diagnosis,
                confidence_score=confidence, raw_text=text, success=True,
            )
        except Exception as exc:
            logger.exception("Unexpected error")
            return self._error(pid, str(exc))

    def _load_and_resize(self, path: str) -> Optional[np.ndarray]:
        try:
            img = cv2.imread(path)
            if img is None:
                pil = Image.open(path).convert("RGB")
                img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        except Exception as exc:
            logger.error("Cannot read image: %s", exc)
            return None
        h, w = img.shape[:2]
        longest = max(h, w)
        if longest > MAX_IMAGE_PX:
            scale = MAX_IMAGE_PX / longest
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            logger.info("Resized %dx%d -> %dx%d", w, h, int(w*scale), int(h*scale))
        return img

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
        h, w = gray.shape
        if h < 800 or w < 600:
            scale = max(800 / h, 600 / w)
            gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

        # Step 1: denoise (light — preserves ink strokes)
        denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)

        # Step 2: sharpen to make thin handwriting strokes crisper
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        sharpened = cv2.filter2D(denoised, -1, kernel)

        # Step 3: CLAHE — boost local contrast without blowing out ink
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(sharpened)

        # Step 4: adaptive threshold — handles uneven lighting / yellowed paper
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
        )
        return binary

    def _ocr(self, img: np.ndarray) -> Tuple[str, float]:
        if self._easyocr:
            try:
                hits = self._easyocr.readtext(img, detail=1)
                if hits:
                    text = "\n".join(h[1] for h in hits)
                    conf = float(np.mean([h[2] for h in hits]))
                    logger.info("EasyOCR conf=%.2f", conf)
                    return text, conf
            except Exception as exc:
                logger.warning("EasyOCR failed, falling back to Tesseract: %s", exc)
        try:
            text = pytesseract.image_to_string(img, config=TESSERACT_PSM)
            data = pytesseract.image_to_data(img, config=TESSERACT_PSM, output_type=pytesseract.Output.DICT)
            confs = [int(c) for c in data["conf"] if int(c) > 0]
            conf = float(np.mean(confs)) / 100.0 if confs else 0.0
            logger.info("Tesseract PSM6 conf=%.2f", conf)

            # If confidence is poor, try sparse-text mode (better for handwriting)
            if conf < 0.40:
                text2 = pytesseract.image_to_string(img, config=TESSERACT_PSM2)
                data2 = pytesseract.image_to_data(img, config=TESSERACT_PSM2, output_type=pytesseract.Output.DICT)
                confs2 = [int(c) for c in data2["conf"] if int(c) > 0]
                conf2 = float(np.mean(confs2)) / 100.0 if confs2 else 0.0
                logger.info("Tesseract PSM11 conf=%.2f", conf2)
                if conf2 > conf and text2.strip():
                    text, conf = text2, conf2

            return text.strip(), conf
        except Exception as exc:
            logger.error("Tesseract failed: %s", exc)
            return "", 0.0

    def _clean(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\w\s.,:()\-/+&'\"]", "", text)
        for pat, rep in (
            (r"rng\b", "mg"), (r"rnl\b", "ml"),
            (r"\bBd\b", "bd"), (r"\bOd\b", "od"),
            (r"(\d+)\s*mg", r"\1 mg"), (r"(\d+)\s*ml", r"\1 ml"),
        ):
            text = re.sub(pat, rep, text, flags=re.IGNORECASE)
        return text.strip()

    _DR_PATTERNS = [
        r"\b(Dr\.?|Doctor|Prof\.?)\s+([A-Za-z\s.]+)",
        r"([A-Za-z\s.]+)\s+(?:MBBS|MD|MS|DM|MCh|DNB)\b",
    ]
    _SPEC_TERMS = [
        "Cardiologist", "Neurologist", "Orthopedic", "Pediatrician",
        "Dermatologist", "Gynecologist", "Psychiatrist",
        "General Physician", "Internal Medicine", "Consultant",
    ]
    _AGE_PATTERNS = [
        r"\bAge\s*:?\s*(\d{1,3})",
        r"(\d{1,3})\s*(?:years?|yrs?|Y)\b",
        r"(\d{1,3})\s*/\s*[MF]",
    ]
    _NAME_PATTERNS = [
        r"\bPatient\s*:?\s*([A-Za-z\s.]+)",
        r"\bName\s*:?\s*([A-Za-z\s.]+)",
        r"\b(?:Mr|Mrs|Ms|Miss)\.?\s+([A-Za-z\s]+)",
    ]

    def _pattern_extract(self, text: str) -> Tuple[Dict, Dict]:
        doctor:  Dict[str, str] = {"name": "", "specialization": "", "registration_number": ""}
        patient: Dict[str, str] = {"name": "", "age": "", "gender": ""}
        for pat in self._DR_PATTERNS:
            m = re.search(pat, text, re.IGNORECASE)
            if m and not doctor["name"]:
                doctor["name"] = m.group(2 if m.lastindex and m.lastindex >= 2 else 1).strip()
        for term in self._SPEC_TERMS:
            if term.lower() in text.lower() and not doctor["specialization"]:
                doctor["specialization"] = term
        m = re.search(r"\bReg\.?\s*No\.?\s*:?\s*([A-Z0-9]+)", text, re.IGNORECASE)
        if m:
            doctor["registration_number"] = m.group(1)
        for pat in self._AGE_PATTERNS:
            m = re.search(pat, text, re.IGNORECASE)
            if m and not patient["age"]:
                for g in m.groups():
                    if g and g.isdigit() and 0 < int(g) < 120:
                        patient["age"] = g
                        break
        if re.search(r"\b(Male|M/|(?<!\w)M(?!\w))\b", text) and "Female" not in text:
            patient["gender"] = "Male"
        elif re.search(r"\b(Female|F/|(?<!\w)F(?!\w))\b", text):
            patient["gender"] = "Female"
        for pat in self._NAME_PATTERNS:
            m = re.search(pat, text, re.IGNORECASE)
            if m and not patient["name"]:
                candidate = m.group(1).strip()
                if 2 < len(candidate) < 50 and not re.search(r"\d", candidate):
                    patient["name"] = candidate
        return doctor, patient

    def _extract_medicines(self, text: str) -> List[Medicine]:
        medicines: List[Medicine] = []
        seen: set = set()
        for line in text.split("\n"):
            line = line.strip()
            if len(line) < 3:
                continue
            if any(kw in line.lower() for kw in ("patient", "doctor", "date", "age", "gender", "diagnosis")):
                continue
            m = re.search(r"(Tab\.?|Cap\.?|Syp\.?|Inj\.?)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)", line, re.IGNORECASE)
            name = m.group(2).strip() if m else ""
            if not name:
                m2 = re.search(r"([A-Z][a-zA-Z]{3,})(?:\s+[A-Z][a-zA-Z]+)?\s+\d+\s*(?:mg|ml|gm)\b", line, re.IGNORECASE)
                name = m2.group(1).strip() if m2 else ""
            if not name or name.lower() in seen:
                continue
            if name.lower() in {"the", "and", "for", "tab", "cap", "syp", "inj"}:
                continue
            seen.add(name.lower())
            dosage_m = re.search(r"\d+\.?\d*\s*(?:mg|ml|gm|g|mcg)", line, re.IGNORECASE)
            medicines.append(Medicine(
                name=name,
                dosage=dosage_m.group(0) if dosage_m else "As prescribed",
                frequency=self._frequency(line),
                duration=self._duration(line),
                instructions=self._instructions(line),
                available=self._MEDICINE_DB.get(name.lower(), True),
            ))
        return medicines

    def _frequency(self, text: str) -> str:
        t = text.lower()
        for pat, label in (
            (r"\b(once\s+daily|od|qd|1-0-0)\b", "Once daily"),
            (r"\b(twice\s+daily|bd|bid|1-0-1)\b", "Twice daily"),
            (r"\b(thrice\s+daily|tid|1-1-1)\b", "Three times daily"),
            (r"\b(qid)\b", "Four times daily"),
            (r"\b(as\s+needed|sos|prn)\b", "As needed"),
            (r"\b(at\s+bedtime|hs|0-0-1)\b", "At bedtime"),
        ):
            if re.search(pat, t):
                return label
        return "As directed"

    def _duration(self, text: str) -> str:
        m = re.search(r"(\d+\s*(?:day|days|week|weeks|month|months))", text, re.IGNORECASE)
        return m.group(1) if m else "As prescribed"

    def _instructions(self, text: str) -> str:
        t = text.lower()
        if re.search(r"before\s+(?:food|meal)", t): return "Before meals"
        if re.search(r"after\s+(?:food|meal)", t):  return "After meals"
        if re.search(r"with\s+(?:food|meal)", t):   return "With meals"
        return ""

    def _llm_extract(self, text: str) -> Optional[Dict]:
        if not self._cohere:
            return None
        prompt = f"""Extract medical information from this prescription OCR text.
Return ONLY a JSON object with keys: patient, doctor, medicines, diagnosis.
- patient: {{name, age, gender}}
- doctor: {{name, specialization, registration_number}}
- medicines: [{{name, dosage, frequency, duration, instructions}}]
- diagnosis: [string]
Rules: "C/o" = complaints -> diagnosis list. Empty fields = "". No extra text.

OCR Text:
---
{text[:2000]}
---"""
        try:
            resp = self._cohere.chat(model="command-r-plus-08-2024", message=prompt, temperature=0.1)
            raw = resp.text.strip()
            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
            s, e = raw.find("{"), raw.rfind("}")
            if s != -1 and e != -1:
                return json.loads(raw[s: e + 1])
        except Exception as exc:
            logger.warning("LLM extraction failed: %s", exc)
        return None

    @staticmethod
    def _merge_llm(llm: Dict, patient: Dict, doctor: Dict) -> None:
        for f in ("name", "age", "gender"):
            v = (llm.get("patient") or {}).get(f, "")
            if v: patient[f] = v
        for f in ("name", "specialization", "registration_number"):
            v = (llm.get("doctor") or {}).get(f, "")
            if v: doctor[f] = v

    def _medicines_from_llm(self, raw: List[Dict]) -> List[Medicine]:
        return [
            Medicine(
                name=m.get("name", ""),
                dosage=m.get("dosage", ""),
                frequency=m.get("frequency", ""),
                duration=m.get("duration", ""),
                instructions=m.get("instructions", ""),
                available=self._MEDICINE_DB.get(m.get("name", "").lower(), True),
            )
            for m in raw if m.get("name")
        ]

    def _confidence(self, ocr_conf: float, n_med: int, patient: Patient, doctor: Doctor, used_llm: bool) -> float:
        med_s = min(1.0, n_med / 3.0 + 0.4) if n_med else 0.0
        pat_s = (0.5 if patient.name else 0) + (0.25 if patient.age else 0) + (0.25 if patient.gender else 0)
        doc_s = (0.6 if doctor.name else 0) + (0.2 if doctor.specialization else 0) + (0.2 if doctor.registration_number else 0)
        score = 0.20 * ocr_conf + 0.40 * med_s + 0.20 * pat_s + 0.20 * doc_s
        if used_llm: score += 0.10
        return float(min(0.99, max(0.0, score)))

    @staticmethod
    def _error(pid: str, msg: str) -> AnalysisResult:
        return AnalysisResult(
            prescription_id=pid, patient=Patient(), doctor=Doctor(),
            medicines=[], diagnosis=[], confidence_score=0.0,
            raw_text="", success=False, error=msg,
        )


EnhancedPrescriptionAnalyzer = PrescriptionAnalyzer