from __future__ import annotations

import os, re, uuid, json, hashlib, time, logging, pathlib
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from difflib import get_close_matches

import cv2
import numpy as np
from PIL import Image
import pytesseract

# ── NEW Gemini SDK (google-genai) ─────────────────────────────────────────────
try:
    from google import genai as _genai
    from google.genai import types as _genai_types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# ── Cohere ────────────────────────────────────────────────────────────────────
try:
    import cohere as _cohere_lib
    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False

# ── thread caps ───────────────────────────────────────────────────────────────
for _k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
cv2.setNumThreads(1)
if not hasattr(Image, "ANTIALIAS"):
    Image.ANTIALIAS = Image.LANCZOS

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── config ────────────────────────────────────────────────────────────────────
MAX_IMAGE_PX      = 2000
GEMINI_MODEL      = "gemini-2.5-flash-lite"
MAX_OUTPUT_TOKENS = 800
TEMPERATURE       = 0.1
RATE_LIMIT_RPM    = 5
CACHE_MAX_SIZE    = 200


# ── load API keys ─────────────────────────────────────────────────────────────
def _load_key(env_name: str) -> Optional[str]:

    # 1. Already in environment
    val = os.getenv(env_name, "").strip()
    if val:
        logger.info("✓ %s loaded from environment", env_name)
        return val

    # 2. Parse .env file directly (handles "KEY= value" with spaces)
    for dot_env in [
        pathlib.Path(".env"),
        pathlib.Path(__file__).parent.parent.parent / ".env",  # repo root
        pathlib.Path(__file__).parent.parent.parent.parent / ".env",
    ]:
        if dot_env.exists():
            for line in dot_env.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                if k.strip() == env_name:
                    v = v.strip().strip('"').strip("'")
                    if v:
                        # Also inject into os.environ so os.getenv works later
                        os.environ[env_name] = v
                        logger.info("✓ %s loaded from %s", env_name, dot_env)
                        return v

    # 3. Import integration.keys as a module
    try:
        import importlib
        keys_mod = importlib.import_module("integration.keys")
        # Force reload in case it was cached before the key was added
        importlib.reload(keys_mod)
        val = str(getattr(keys_mod, env_name, "") or "").strip()
        if val:
            os.environ[env_name] = val
            logger.info("✓ %s loaded from integration/keys.py (import)", env_name)
            return val
    except Exception:
        pass

    # 4. Parse integration/keys.py as raw text (bypasses import issues)
    for keys_file in [
        pathlib.Path("integration/keys.py"),
        pathlib.Path(__file__).parent.parent / "integration" / "keys.py",
        pathlib.Path(__file__).parent.parent.parent / "integration" / "keys.py",
    ]:
        if keys_file.exists():
            for line in keys_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                if k.strip() == env_name:
                    v = v.strip().strip('"').strip("'")
                    if v:
                        os.environ[env_name] = v
                        logger.info("✓ %s loaded from %s (raw parse)", env_name, keys_file)
                        return v

    logger.warning("✗ %s not found in env, .env, or integration/keys.py", env_name)
    return None


# ── system prompt from file ───────────────────────────────────────────────────
def _load_system_prompt() -> str:
    candidates = [
        pathlib.Path(__file__).parent / "gemini_system_prompt.txt",
        pathlib.Path("gemini_system_prompt.txt"),
        pathlib.Path("app/services/gemini_system_prompt.txt"),
    ]
    for p in candidates:
        if p.exists():
            txt = p.read_text(encoding="utf-8").strip()
            logger.info("Loaded system prompt from %s (%d chars)", p, len(txt))
            return txt
    logger.warning("gemini_system_prompt.txt not found — using inline fallback")
    return (
        "You are an expert at reading handwritten Indian medical prescriptions. "
        "Return ONLY valid JSON, no markdown. "
        "bd=Twice daily, od=Once daily, tid=Three times daily, SOS=As needed. "
        "×2 after medicine = 2 weeks duration. c/o = diagnosis. "
        "BP/weight = vitals not medicines. "
        "Doctor name from top letterhead, exclude title. "
        "Patient name = first handwritten name."
    )


MEDICINE_DB: Dict[str, bool] = {k: True for k in [
    "augmentin","amoxicillin","azithromycin","ciprofloxacin","cephalexin",
    "doxycycline","clarithromycin","metronidazole","tinidazole","levofloxacin",
    "norfloxacin","paracetamol","acetaminophen","ibuprofen","diclofenac",
    "aspirin","naproxen","crocin","combiflam","dolo","meftal","esomeprazole",
    "omeprazole","pantoprazole","lansoprazole","famotidine","ondansetron",
    "domperidone","metoclopramide","dicyclomine",
    "esofag","bifilac","emanzen",
    "cetirizine","loratadine","fexofenadine","levocetirizine","chlorpheniramine",
    "serratiopeptidase","serrapeptase","trypsin","bromelain","chymotrypsin",
    "progesterone","estrogen","clomiphene","letrozole",
    "folic acid","calcium","ferrous","vitamin d3","vitamin b12","vitamin c",
    "metformin","insulin","glimepiride","montelukast","losartan","amlodipine",
    "atorvastatin","metoprolol","salbutamol","prednisolone","deflazacort",
    "cefixime","ceftriaxone",
]}

def _normalize_medicine(name: str) -> str:
    name = name.lower().strip()
    matches = get_close_matches(name, MEDICINE_DB.keys(), n=1, cutoff=0.75)
    if matches:
        return matches[0].title()
    return name.title()

MED_LINE_REGEX = re.compile(
    r"(tab|cap|inj|syp)\s+([a-zA-Z0-9\-\+]+)",
    re.IGNORECASE
)

SPECIALIZATIONS = [
    "Obstetrician & Gynaecologist","Obstetrician","Gynaecologist","Gynecologist",
    "OBGYN","Cardiologist","Neurologist","Orthopedic","Pediatrician",
    "Paediatrician","Dermatologist","Psychiatrist","General Physician",
    "General Practitioner","Internal Medicine","Consultant","Surgeon",
    "Ophthalmologist","ENT Specialist","Urologist","Nephrologist",
    "Pulmonologist","Endocrinologist","Gastroenterologist","Rheumatologist","Oncologist",
]


# ── data classes ──────────────────────────────────────────────────────────────
@dataclass
class Patient:
    name:   str = ""
    age:    str = ""
    gender: str = ""

@dataclass
class Doctor:
    name:                str = ""
    specialization:      str = ""
    registration_number: str = ""

@dataclass
class Medicine:
    name:         str  = ""
    dosage:       str  = ""
    quantity:     str  = "1"
    frequency:    str  = ""
    duration:     str  = ""
    instructions: str  = ""
    available:    bool = True

@dataclass
class AnalysisResult:
    prescription_id:  str
    patient:          Patient
    doctor:           Doctor
    medicines:        List[Medicine]
    diagnosis:        List[str]
    confidence_score: float
    raw_text:         str
    success:          bool = True
    error:            str  = ""


# ── cache ─────────────────────────────────────────────────────────────────────
class _Cache:
    def __init__(self, max_size: int = CACHE_MAX_SIZE):
        self._store: Dict[str, dict] = {}
        self._order: List[str] = []
        self._max   = max_size
        self._hits  = 0
        self._miss  = 0

    def get(self, key: str) -> Optional[dict]:
        if key in self._store:
            self._hits += 1
            logger.info("Cache HIT [%s...]  hits=%d miss=%d", key[:8], self._hits, self._miss)
            return self._store[key]
        self._miss += 1
        return None

    def set(self, key: str, val: dict) -> None:
        if key in self._store:
            return
        if len(self._order) >= self._max:
            del self._store[self._order.pop(0)]
        self._store[key] = val
        self._order.append(key)

    def stats(self) -> str:
        t = self._hits + self._miss
        r = self._hits / t * 100 if t else 0
        return f"Cache: {len(self._store)} stored | {self._hits} hits | {self._miss} misses | {r:.0f}% hit rate"


# ── rate limiter ──────────────────────────────────────────────────────────────
class _RateLimiter:
    def __init__(self, rpm: int = RATE_LIMIT_RPM):
        self._rpm  = rpm
        self._ts: List[float] = []

    def wait_if_needed(self) -> None:
        now = time.monotonic()
        self._ts = [t for t in self._ts if now - t < 60.0]
        if len(self._ts) >= self._rpm:
            wait = 60.0 - (now - self._ts[0]) + 0.2
            if wait > 0:
                logger.info("Rate limit: sleeping %.1fs  (%d req in last 60s)", wait, len(self._ts))
                time.sleep(wait)
        self._ts.append(time.monotonic())


# ═════════════════════════════════════════════════════════════════════════════
class PrescriptionAnalyzer:

    _USER_PROMPT = """
Read this prescription image carefully.

Focus especially on:

1. Doctor name in the printed letterhead
2. Patient name written by hand near the top
3. Age and gender near the name
4. Medicine list
5. Diagnosis

Return ONLY JSON using this schema:

{
  "patient": {"name": "", "age": "", "gender": ""},
  "doctor": {"name": "", "specialization": "", "registration_number": ""},
  "medicines": [],
  "diagnosis": [],
  "vitals": {"bp": "", "weight": ""},
  "date": ""
}

Important rules:

Doctor name is printed in the letterhead.

Patient name is handwritten near the top.

Example:
"Banani 47" → name="Banani", age="47"

Example:
"Sneha Das 17/F" → name="Sneha Das", age="17", gender="Female"
"""

    def __init__(self, cohere_api_key: Optional[str] = None):
        self._cache        = _Cache()
        self._limiter      = _RateLimiter()
        self._sys_prompt   = _load_system_prompt()
        self._gemini_key   = _load_key("GEMINI_API_KEY")
        self._gemini       = self._init_gemini()
        self._cohere       = self._init_cohere(cohere_api_key)

        if self._gemini:
            logger.info("✓  Engine : Gemini 2.0 Flash Vision  (google-genai SDK)")
            logger.info("   Model  : %s | Temp: %.1f | Max tokens: %d",
                        GEMINI_MODEL, TEMPERATURE, MAX_OUTPUT_TOKENS)
            logger.info("   Cache  : enabled (%d slots) | Rate limit: %d req/min",
                        CACHE_MAX_SIZE, RATE_LIMIT_RPM)
        elif self._cohere:
            logger.info("✓  Engine : Cohere LLM + OCR (Gemini key missing)")
        else:
            logger.warning("⚠  Engine : Tesseract + regex only")

    # ── init ──────────────────────────────────────────────────────────────────
    def _init_gemini(self):
        if not GEMINI_AVAILABLE:
            logger.warning("google-genai not installed. Run:\n"
                           "  pip uninstall google-generativeai -y\n"
                           "  pip install google-genai")
            return None
        if not self._gemini_key:
            logger.warning("GEMINI_API_KEY not found after checking env, .env, and integration/keys.py")



            return None
        try:
            client = _genai.Client(api_key=self._gemini_key)
            # List available models once at startup
            try:
                models = list(client.models.list())
                logger.info("Available Gemini models:")
                for m in models[:50]:
                    logger.info(" - %s", m.name)
            except Exception as e:
                logger.warning("Could not list Gemini models: %s", e)

            logger.info("Gemini client ready (%s)", GEMINI_MODEL)
            return client
        except Exception as e:
            logger.warning("Gemini init error: %s", e)
            return None

    def _init_cohere(self, api_key: Optional[str]):
        if not COHERE_AVAILABLE:
            return None
        key = api_key or _load_key("COHERE_API_KEY")
        if not key:
            return None
        try:
            return _cohere_lib.Client(key)
        except Exception as e:
            logger.warning("Cohere init error: %s", e)
            return None

    # ── main entry ────────────────────────────────────────────────────────────
    def analyze_prescription(self, image_path: str) -> AnalysisResult:
        pid = f"RX{datetime.now().strftime('%Y%m%d%H%M%S')}{uuid.uuid4().hex[:6].upper()}"
        logger.info("Analysing %s  [id=%s]", image_path, pid)
        try:
            img = self._load_and_resize(image_path)
            if img is None:
                return self._make_error(pid, "Could not read image file")

            # PATH 1 — Gemini Vision
            if self._gemini:
                img_hash = self._image_hash(img)
                cached   = self._cache.get(img_hash)
                if cached:
                    return self._build_result(pid, cached, "gemini-vision[cached]")
                self._limiter.wait_if_needed()
                
                header, body, meds_region = self._split_prescription_regions(img)
                data_h = self._gemini_extract(header)
                data_b = self._gemini_extract(body)
                data_m = self._gemini_extract(meds_region)
                
                data = self._merge_gemini_results(data_h, data_b, data_m)
                if data:
                    self._cache.set(img_hash, data)
                    logger.info(self._cache.stats())
                    return self._build_result(pid, data, "gemini-vision")

            # PATH 2 — Cohere + OCR
            proc = self._preprocess(img)
            raw_text, ocr_conf = self._run_ocr(proc)
            text = self._clean_text(raw_text)
            logger.info("OCR: %d chars, conf=%.2f", len(text), ocr_conf)

            if self._cohere and text:
                data = self._cohere_extract(text)
                if data:
                    return self._build_result(pid, data, "cohere",
                                              raw_text=text, ocr_conf=ocr_conf)

            # PATH 3 — regex only
            doc_d, pat_d = self._pattern_extract(text)
            meds  = self._pattern_medicines(text)
            diag  = self._pattern_diagnosis(text)
            patient = Patient(**{k: pat_d.get(k,"") for k in ("name","age","gender")})
            doctor  = Doctor(name=doc_d.get("name",""),
                             specialization=doc_d.get("specialization",""),
                             registration_number=doc_d.get("registration_number",""))
            conf = self._score_fallback(ocr_conf, len(meds), patient, doctor)
            return AnalysisResult(prescription_id=pid, patient=patient, doctor=doctor,
                                  medicines=meds, diagnosis=diag, confidence_score=conf,
                                  raw_text=text, success=True)
        except Exception as exc:
            logger.exception("Unhandled error")
            return self._make_error(pid, str(exc))

    # ── Gemini Vision (new SDK) with exponential backoff retry ───────────────
    def _gemini_extract(self, img: np.ndarray) -> Optional[Dict]:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        try:
            response = self._gemini.models.generate_content(
                model=GEMINI_MODEL,
                contents=[self._USER_PROMPT, pil_img],
                config=_genai_types.GenerateContentConfig(
                    system_instruction=self._sys_prompt,
                    temperature=TEMPERATURE,
                    max_output_tokens=MAX_OUTPUT_TOKENS,
                ),
            )

            raw = (response.text or "").strip()
            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()

            s, e = raw.find("{"), raw.rfind("}")
            if s == -1 or e == -1:
                logger.warning("Gemini returned no JSON. Raw: %s", raw[:200])
                return None

            json_str = raw[s:e+1]
            # fix common model formatting errors
            json_str = json_str.replace("\n", " ")
            json_str = re.sub(r",\s*}", "}", json_str)
            json_str = re.sub(r",\s*]", "]", json_str)

            try:
                data = json.loads(json_str)
            except json.JSONDecodeError:
                logger.warning("Gemini JSON repair attempt")
                json_str = re.sub(r"(\w+):", r'"\1":', json_str)
                data = json.loads(json_str)

            data["_source"] = "gemini-vision"
            logger.info(
                "Gemini → patient=%r doctor=%r meds=%d diag=%s",
                data.get("patient", {}).get("name"),
                data.get("doctor", {}).get("name"),
                len(data.get("medicines", [])),
                data.get("diagnosis", []),
            )
            return data

        except Exception as exc:
            msg = str(exc).lower()

            if "404" in msg or "not found" in msg:
                logger.warning("⚠ Gemini model not found: %s", exc)
            elif "403" in msg or "api key" in msg or "permission" in msg:
                logger.warning("⚠ Gemini auth error: %s", exc)
            elif "429" in msg or "quota" in msg or "too many requests" in msg or "rate limit" in msg:
                logger.warning("⚠ Gemini rate limit error: %s", exc)
            else:
                logger.warning("⚠ Gemini error: %s", exc)

            return None


    # ── Cohere ────────────────────────────────────────────────────────────────
    def _cohere_extract(self, text: str) -> Optional[Dict]:
        prompt = (
            "Indian prescription expert. OCR text below may have errors — correct them.\n"
            "bd=Twice daily, od=Once daily, tid=Three times daily, SOS=As needed.\n"
            "×2 = 2 weeks. c/o = diagnosis. BP/weight = vitals. "
            "Doctor name from top letterhead (no title). "
            "Return ONLY valid JSON no markdown:\n"
            '{"patient":{"name":"","age":"","gender":""},'
            '"doctor":{"name":"","specialization":"","registration_number":""},'
            '"medicines":[{"name":"","dosage":"","frequency":"","duration":"",'
            '"instructions":"","quantity":"1"}],"diagnosis":[],'
            '"vitals":{"bp":"","weight":""},"date":""}\n\n'
            f"OCR TEXT:\n---\n{text[:3000]}\n---"
        )
        try:
            resp = self._cohere.chat(model="command-r-plus-08-2024",
                                     message=prompt, temperature=0.0)
            raw = re.sub(r"^```(?:json)?\s*|\s*```$","",
                         resp.text.strip(),flags=re.MULTILINE).strip()
            s,e = raw.find("{"),raw.rfind("}")
            if s==-1 or e==-1: return None
            data = json.loads(raw[s:e+1])
            data["_source"] = "cohere"
            return self._fix_cohere(data, text)
        except Exception as exc:
            logger.warning("Cohere error: %s", exc)
            return None

    def _fix_cohere(self, data: Dict, ocr: str) -> Dict:
        doc = data.get("doctor") or {}
        if not doc.get("name") or len(doc.get("name","")) < 4:
            for line in ocr.split("\n")[:10]:
                m = re.search(
                    r"Dr\.?\s*\(?\s*(?:Mrs?\.?|Miss|Ms\.?)?\s*\)?\s*"
                    r"([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){1,3})",
                    line, re.IGNORECASE)
                if m:
                    cand = m.group(1).strip()
                    skip = {"clinic","hospital","medical","centre","patient","consultant","mbbs","dgo"}
                    if cand.lower().split()[0] not in skip and len(cand) > 5:
                        doc["name"] = cand; data["doctor"] = doc; break
        if not doc.get("specialization"):
            up = ocr.upper()
            if "GYNAEC" in up or "GYNEC" in up:
                doc["specialization"] = ("Obstetrician & Gynaecologist"
                                         if "OBSTET" in up else "Gynaecologist")
                data["doctor"] = doc
        pat = data.get("patient") or {}
        if not pat.get("name"):
            m = re.search(r"^([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})?)\s+\d{1,3}",
                          ocr, re.MULTILINE)
            if m:
                cand = m.group(1).strip()
                skip = {"clinic","hospital","morning","evening","consultant","obstetrician"}
                if cand.lower() not in skip:
                    pat["name"] = cand; data["patient"] = pat
        return data

    # ── build result ──────────────────────────────────────────────────────────
    def _build_result(self, pid: str, data: Dict, source: str,
                      raw_text: str = "", ocr_conf: float = 0.0) -> AnalysisResult:
        p = data.get("patient") or {}
        d = data.get("doctor")  or {}
        patient = Patient(
            name  = str(p.get("name","")   or "").strip(),
            age   = str(p.get("age","")    or "").strip(),
            gender= str(p.get("gender","") or "").strip(),
        )
        doctor = Doctor(
            name                = str(d.get("name","")                or "").strip(),
            specialization      = str(d.get("specialization","")      or "").strip(),
            registration_number = str(d.get("registration_number","") or "").strip(),
        )

        if not patient.name and (data.get("medicines") or []):
            diag_list = data.get("diagnosis", [])
            if diag_list:
                guess = str(diag_list[0]).strip()
                if len(guess.split()) <= 2:
                    patient.name = guess

        medicines = [
            Medicine(
                name        = _normalize_medicine(str(m.get("name","")).strip()),
                dosage      = str(m.get("dosage","")      or "As prescribed").strip(),
                frequency   = str(m.get("frequency","")   or "As directed").strip(),
                duration    = str(m.get("duration","")    or "As prescribed").strip(),
                instructions= str(m.get("instructions","")or "").strip(),
                quantity    = str(m.get("quantity","1")   or "1").strip(),
                available   = MEDICINE_DB.get(_normalize_medicine(str(m.get("name","")).strip()).lower(), True),
            )
            for m in (data.get("medicines") or [])
            if str(m.get("name","")).strip()
        ]
        diag = [str(x).strip() for x in (data.get("diagnosis") or []) if str(x).strip()]
        if "gemini" in source:
            conf = self._score_vision(patient, doctor, medicines)
        elif source == "cohere":
            conf = self._score_cohere(patient, doctor, medicines, ocr_conf)
        else:
            conf = self._score_fallback(ocr_conf, len(medicines), patient, doctor)
        logger.info("[%s] conf=%.0f%%  patient=%r  doctor=%r  meds=%d",
                    source, conf*100, patient.name, doctor.name, len(medicines))
        return AnalysisResult(prescription_id=pid, patient=patient, doctor=doctor,
                              medicines=medicines, diagnosis=diag,
                              confidence_score=conf, raw_text=raw_text, success=True)

    # ── scoring ───────────────────────────────────────────────────────────────
    def _score_vision(self, p: Patient, d: Doctor, m: List[Medicine]) -> float:
        s = 0.85
        if p.name: s+=0.03
        if p.age:  s+=0.02
        if d.name: s+=0.04
        if d.specialization: s+=0.02
        if m: s+=min(0.04, len(m)*0.01)
        return float(min(0.99, round(s,4)))

    def _score_cohere(self, p: Patient, d: Doctor, m: List[Medicine], ocr: float) -> float:
        s = 0.55
        if p.name: s+=0.07
        if p.age:  s+=0.03
        if d.name: s+=0.10
        if d.specialization: s+=0.05
        if m: s+=min(0.12, len(m)*0.03)
        return float(min(0.97, round(s+ocr*0.05,4)))

    def _score_fallback(self, ocr: float, n: int, p: Patient, d: Doctor) -> float:
        ms = min(1.0,(n/3)*0.9+0.3) if n else 0
        ps = (0.5 if p.name else 0)+(0.25 if p.age else 0)
        ds = (0.6 if d.name else 0)+(0.2 if d.specialization else 0)
        return float(min(0.75, max(0.0, round(0.20*ocr+0.35*ms+0.20*ps+0.25*ds,4))))

    # ── image helpers ─────────────────────────────────────────────────────────
    def _load_and_resize(self, path: str) -> Optional[np.ndarray]:
        try:
            img = cv2.imread(path)
            if img is None:
                pil = Image.open(path).convert("RGB")
                img = cv2.cvtColor(np.array(pil), cv2.COLOR_BGR2RGB)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        except Exception as exc:
            logger.error("imread failed: %s", exc); return None
        h,w = img.shape[:2]
        longest = max(h,w)
        if longest > MAX_IMAGE_PX:
            s=MAX_IMAGE_PX/longest
            img=cv2.resize(img,(int(w*s),int(h*s)),interpolation=cv2.INTER_AREA)
        elif longest < 800:
            s=800/longest
            img=cv2.resize(img,(int(w*s),int(h*s)),interpolation=cv2.INTER_CUBIC)
        return img

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
        # denoise
        gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        # contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        # sharpen
        kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
        gray = cv2.filter2D(gray,-1,kernel)
        # adaptive threshold
        th = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 35, 11
        )
        return th

    def _run_ocr(self, img: np.ndarray) -> Tuple[str, float]:
        best=("",0.0)
        for cfg in (
            "--oem 3 --psm 4",
            "--oem 3 --psm 6",
            "--oem 3 --psm 11"
        ):
            try:
                txt=pytesseract.image_to_string(img,config=cfg)
                dct=pytesseract.image_to_data(img,config=cfg,
                        output_type=pytesseract.Output.DICT)
                cs=[int(c) for c in dct["conf"] if int(c)>0]
                cf=float(np.mean(cs))/100.0 if cs else 0.0
                if len(txt)*(0.5+0.5*cf)>len(best[0])*(0.5+0.5*best[1]):
                    best=(txt.strip(),cf)
            except Exception as exc:
                logger.warning("OCR cfg %s: %s",cfg,exc)
        return best

    def _clean_text(self, text: str) -> str:
        lines=[l.strip() for l in text.split("\n") if l.strip()]
        text="\n".join(lines)
        for p,r in ((r"\brng\b","mg"),(r"\brnl\b","ml"),(r"\bBd\b","bd"),
                    (r"\bOd\b","od"),(r"\b0d\b","od"),(r"\b8d\b","bd"),
                    (r"\bc/o\b","C/O"),(r"\bC/o\b","C/O")):
            text=re.sub(p,r,text,flags=re.IGNORECASE)
        return text.strip()

    @staticmethod
    def _image_hash(img: np.ndarray) -> str:
        small=cv2.resize(img,(64,64))
        return hashlib.md5(small.tobytes()).hexdigest()

    # ── pattern fallback ──────────────────────────────────────────────────────
    def _pattern_extract(self, text: str) -> Tuple[Dict, Dict]:
        doc:Dict[str,str]={"name":"","specialization":"","registration_number":""}
        pat:Dict[str,str]={"name":"","age":"","gender":""}
        m=re.search(r"Dr\.?\s*\(?\s*(?:Mrs?\.?|Miss|Ms\.?)?\s*\)?\s*"
                    r"([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,3})",text,re.IGNORECASE)
        if m:
            cand=m.group(1).strip()
            skip={"clinic","hospital","medical","centre","patient","consultant"}
            if cand.lower().split()[0] not in skip and len(cand)>3:
                doc["name"]=cand
        for term in SPECIALIZATIONS:
            if re.search(r'\b'+re.escape(term)+r'\b',text,re.IGNORECASE):
                doc["specialization"]=term; break
        for p in (r"\b(\d{1,3})\s*(?:years?|yrs?)\b",r"\bAge\s*[:\-]?\s*(\d{1,3})\b"):
            m=re.search(p,text,re.IGNORECASE)
            if m:
                try:
                    age=int(m.group(1))
                    if 0<age<120: pat["age"]=str(age); break
                except ValueError: pass
        if re.search(r"\bFemale\b",text,re.IGNORECASE): pat["gender"]="Female"
        elif re.search(r"\bMale\b",text,re.IGNORECASE): pat["gender"]="Male"
        for p in (r"^\s*([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})?)\s+\d{1,3}\s",
                  r"\bPatient\s*[:\-]?\s*([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)"):
            m=re.search(p,text,re.MULTILINE)
            if m:
                cand=m.group(1).strip()
                skip={"clinic","hospital","patient","doctor","morning","evening",
                      "consultant","obstetrician","gynaecologist"}
                if cand.lower() not in skip and 2<len(cand)<50:
                    pat["name"]=cand; break
        return doc,pat

    def _pattern_medicines(self, text: str) -> List[Medicine]:
        meds,seen=[],set()
        skip_kw={"patient","doctor","date","clinic","hospital","morning","evening",
                 "m.b.b.s","m.d.","d.g.o","phone","tel","new delhi","delhi",
                 "daily","sunday","closed","consultation"}
        for line in text.split("\n"):
            line=line.strip()
            if len(line)<3 or any(kw in line.lower() for kw in skip_kw): continue
            name=""
            # Improved regex check
            rm = MED_LINE_REGEX.search(line)
            if rm:
                name = rm.group(2).strip()
            
            if not name:
                m=re.search(r"(?:Tab\.?|Cap\.?|Syp\.?|Inj\.?|Cream|Gel|Oint\.?)\s+([A-Za-z][A-Za-z0-9\-\+\s]{1,24}?)"
                            r"(?:\s+\d|\s+once|\s+bd|\s+od|\s+tid|\s+tds|\s+x\b|\s*$)",
                            line,re.IGNORECASE)
                if m: name=m.group(1).strip().rstrip(" ,-")
            if not name:
                for k in MEDICINE_DB:
                    if re.search(r'\b'+re.escape(k)+r'\b',line,re.IGNORECASE):
                        name=k.title(); break
            if not name or name.lower() in seen: continue
            if name.lower() in {"tab","cap","syp","inj","the","and","for","with"}: continue
            seen.add(name.lower())
            dm=re.search(r"\d+\.?\d*\s*(?:mg|ml|gm|g|mcg|gram)",line,re.IGNORECASE)
            qm=re.search(r"[x×]\s*(\d+)",line,re.IGNORECASE)
            meds.append(Medicine(
                name=name,dosage=dm.group(0) if dm else "As prescribed",
                frequency=self._freq(line),duration=self._dur(line),
                instructions=self._instr(line),
                quantity=qm.group(1) if qm else "1",
                available=MEDICINE_DB.get(name.lower(),True),
            ))
        return meds

    def _pattern_diagnosis(self, text: str) -> List[str]:
        out=[]
        for m in re.finditer(r"[Cc][/\\][Oo]\.?\s+(.+?)(?:\n|$|\.)",text):
            c=m.group(1).strip().rstrip(".,")
            if c and len(c)>2: out.append(c)
        return out

    def _freq(self,t:str)->str:
        t=t.lower()
        for p,l in ((r"\b(once\s+daily|od|qd)\b","Once daily"),
                    (r"\b(twice\s+daily|bd|bid)\b","Twice daily"),
                    (r"\b(thrice\s+daily|tid|tds)\b","Three times daily"),
                    (r"\bqid\b","Four times daily"),
                    (r"\b(sos|prn)\b","Emergency use only (SOS)"),
                    (r"\bhs\b","At bedtime (HS)"),
                    (r"\bonce\b","Once daily"),
                    (r"\btwice\b","Twice daily")):
            if re.search(p,t): return l
        return "As directed"

    def _dur(self,t:str)->str:
        for c,n in (("②","2"),("③","3"),("④","4"),("⑤","5")):
            if c in t: return f"{n} weeks"
        m=re.search(r"[x×]\s*(\d+)",t)
        if m: return f"{m.group(1)} weeks"
        m=re.search(r"(\d+\s*(?:day|days|week|weeks|month|months))",t,re.IGNORECASE)
        return m.group(1) if m else "As prescribed"

    def _instr(self,t:str)->str:
        t=t.lower()
        if re.search(r"before\s+(?:food|meal)",t): return "Before meals"
        if re.search(r"after\s+(?:food|meal)",t):  return "After meals"
        if re.search(r"with\s+(?:food|meal)",t):   return "With meals"
        return ""

    @staticmethod
    def _make_error(pid:str,msg:str)->AnalysisResult:
        return AnalysisResult(prescription_id=pid,patient=Patient(),doctor=Doctor(),
                              medicines=[],diagnosis=[],confidence_score=0.0,
                              raw_text="",success=False,error=msg)

    def _split_prescription_regions(self, img: np.ndarray):
        h, w = img.shape[:2]
        header = img[0:int(h*0.35), :]
        body   = img[int(h*0.35):int(h*0.75), :]
        meds   = img[int(h*0.60):h, :]
        return header, body, meds

    def _merge_gemini_results(self, h, b, m):
        result = {
            "patient": {"name": "", "age": "", "gender": ""},
            "doctor": {"name": "", "specialization": "", "registration_number": ""},
            "medicines": [],
            "diagnosis": [],
            "vitals": {"bp": "", "weight": ""},
            "date": ""
        }
        for src in [h, b, m]:
            if not src:
                continue

            # patient fields
            for k, v in (src.get("patient") or {}).items():
                if v and not result["patient"].get(k):
                    result["patient"][k] = v

            # doctor fields
            for k, v in (src.get("doctor") or {}).items():
                if v and not result["doctor"].get(k):
                    result["doctor"][k] = v

            # medicines
            if src.get("medicines"):
                result["medicines"].extend(src["medicines"])

            # diagnosis
            if src.get("diagnosis"):
                result["diagnosis"].extend(src["diagnosis"])

            # vitals
            for k, v in (src.get("vitals") or {}).items():
                if v and not result["vitals"].get(k):
                    result["vitals"][k] = v

            if src.get("date") and not result["date"]:
                result["date"] = src["date"]

        return result


EnhancedPrescriptionAnalyzer = PrescriptionAnalyzer