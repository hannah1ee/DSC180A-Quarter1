# ============================================
# STEP 0: ENVIRONMENT CONFIGURATION
# ============================================
import os
os.environ["OPENAI_API_KEY"] = "your-sky-key"
os.environ["SEMLIB_DEFAULT_MODEL"] = "openai/gpt-4.1-mini"
os.environ["SEMLIB_MAX_CONCURRENCY"] = "3"

# Runtime & safety parameters
BATCH_SIZE   = 20      # requests per batch
PAUSE_SEC    = 5       # seconds between batches
MAX_RETRIES  = 5       # exponential backoff attempts
MAX_CHARS    = 8000    # truncate long dialogues
SAMPLE_N     = 50      # number of rows to process

# ============================================
# STEP 1: IMPORTS
# ============================================
import pandas as pd
import re, itertools, random, asyncio
import nest_asyncio; nest_asyncio.apply()

from typing import List, Optional, Literal
from pydantic import BaseModel, Field
from semlib import Session, OnDiskCache

# ============================================
# STEP 2: LOAD DATASET
# ============================================
CSV_URL = "https://raw.githubusercontent.com/abachaa/MTS-Dialog/main/Main-Dataset/MTS-Dialog-TrainingSet.csv"
df = pd.read_csv(CSV_URL)
TEXT_COL = "dialogue"

# ============================================
# STEP 3: DEFINE EXTRACTION SCHEMA
# ============================================
class VisitSummary(BaseModel):
    # Core fields
    chief_complaint: Optional[str] = Field(None)
    family_illnesses: List[str] = Field(default_factory=list)
    medications: List[str] = Field(default_factory=list)
    small_talk_topics: List[str] = Field(default_factory=list)
    has_small_talk: bool = Field(default=False)

    # Extended fields
    symptoms: List[str] = Field(default_factory=list)
    symptom_count: int = 0
    condition_duration: Optional[Literal["acute","subacute","chronic","unclear"]] = "unclear"
    investigations: List[str] = Field(default_factory=list)
    plan_actions: List[str] = Field(default_factory=list)
    followup_timeline: Optional[str] = None
    adherence_status: Optional[Literal["adherent","nonadherent","unclear"]] = "unclear"
    adherence_signals: List[str] = Field(default_factory=list)
    uncertainty_phrases: List[str] = Field(default_factory=list)
    sdh_flags: List[Literal["cost","insurance","transport","work","housing","caregiving","language","food"]] = Field(default_factory=list)
    patient_emotion: Optional[Literal["neutral","anxious","sad","frustrated","in_pain","other"]] = "neutral"
    doctor_emotion: Optional[Literal["neutral","warm","hurried","frustrated","other"]] = "neutral"
    patient_word_count: int = 0
    doctor_word_count: int = 0
    red_flag: bool = False

# ============================================
# STEP 4: PROMPT TEMPLATE
# ============================================
SYSTEM = """You extract structured fields from a doctor–patient conversation.
Return ONLY the JSON fields requested. Be conservative; if unsure, use empty/neutral/unclear.
Normalize medication names to generic forms when obvious (e.g., ibuprofen).
If speaker labels are present, use them to count patient vs doctor words.
"""

def mk_prompt(text: str) -> str:
    return f"""{SYSTEM}

CONVERSATION:
{text}

TASK — return JSON with these fields:
- chief_complaint: short phrase
- family_illnesses: list[str]
- medications: list[str]
- small_talk_topics: list[str]
- has_small_talk: boolean

- symptoms: list[str]
- symptom_count: integer
- condition_duration: one of ["acute","subacute","chronic","unclear"]
- investigations: list[str]
- plan_actions: list[str]
- followup_timeline: string or null
- adherence_status: one of ["adherent","nonadherent","unclear"]
- adherence_signals: list[str]
- uncertainty_phrases: list[str]
- sdh_flags: subset of ["cost","insurance","transport","work","housing","caregiving","language","food"]
- patient_emotion: one of ["neutral","anxious","sad","frustrated","in_pain","other"]
- doctor_emotion: one of ["neutral","warm","hurried","frustrated","other"]
- patient_word_count: integer
- doctor_word_count: integer
- red_flag: boolean

Return valid JSON ONLY.
"""

def truncate(s, max_chars=MAX_CHARS):
    """Safely truncate long conversations to avoid token overflows."""
    return s if len(s) <= max_chars else s[:max_chars]

# ============================================
# STEP 5: INITIALIZE SEMLIB SESSION
# ============================================
session = Session(cache=OnDiskCache(".semlib_cache"))
print("Model:", session.model)

# ============================================
# STEP 6: SAMPLE DIALOGUES
# ============================================
sample_df = df.sample(n=min(SAMPLE_N, len(df)), random_state=42)
dialogs = [truncate(str(x)) for x in sample_df[TEXT_COL].tolist()]
print(f"Loaded {len(dialogs)} sample dialogues.")

# ============================================
# STEP 7: EXTRACTION FUNCTIONS
# ============================================
async def extract_batch(batch_dialogs: List[str]):
    """Extract structured data for a batch with retry on rate limits."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return await session.map(
                batch_dialogs,
                template=lambda x: mk_prompt(x),
                return_type=VisitSummary,
            )
        except Exception as e:
            msg = str(e).lower()
            if any(k in msg for k in ["rate limit", "429", "rpm", "tpm"]):
                backoff = min(2 ** attempt, 20) + random.random()
                print(f"Rate limit. Retrying in {backoff:.1f}s...")
                await asyncio.sleep(backoff)
            else:
                raise
    raise RuntimeError("Gave up after repeated rate limits.")

async def run_paced(dialogs: List[str], batch_size=BATCH_SIZE, pause=PAUSE_SEC):
    """Run batched extraction with pacing between requests."""
    extracted: List[VisitSummary] = []
    total = len(dialogs)
    for i in range(0, total, batch_size):
        batch = dialogs[i:i + batch_size]
        results = await extract_batch(batch)
        extracted.extend(results)
        print(f"Processed {min(i+batch_size, total)}/{total} | Cost ${session.total_cost():.3f}")
        if i + batch_size < total:
            await asyncio.sleep(pause)
    return extracted

# ============================================
# STEP 8: RUN EXTRACTION
# ============================================
extracted: List[VisitSummary] = asyncio.run(run_paced(dialogs))
print(f"Done. Total extracted: {len(extracted)} | Cost: ${session.total_cost():.3f}")

# ============================================
# STEP 9: NORMALIZATION & OUTPUT
# ============================================
def norm_med(m: str) -> str:
    """Normalize medication text to lowercase alphanumeric."""
    return re.sub(r'[^a-z0-9\- ]','', m.strip().lower())

rows = []
for vs in extracted:
    rows.append({
        "chief_complaint": vs.chief_complaint,
        "family_illnesses": vs.family_illnesses,
        "medications": [norm_med(m) for m in vs.medications],
        "small_talk_topics": vs.small_talk_topics,
        "has_small_talk": vs.has_small_talk,
        "symptoms": vs.symptoms,
        "symptom_count": vs.symptom_count,
        "condition_duration": vs.condition_duration,
        "investigations": vs.investigations,
        "plan_actions": vs.plan_actions,
        "followup_timeline": vs.followup_timeline,
        "adherence_status": vs.adherence_status,
        "adherence_signals": vs.adherence_signals,
        "uncertainty_phrases": vs.uncertainty_phrases,
        "sdh_flags": vs.sdh_flags,
        "patient_emotion": vs.patient_emotion,
        "doctor_emotion": vs.doctor_emotion,
        "patient_word_count": vs.patient_word_count,
        "doctor_word_count": vs.doctor_word_count,
        "red_flag": vs.red_flag,
    })
xed = pd.DataFrame(rows)

# Derived metrics for analysis
xed["plan_count"] = xed["plan_actions"].apply(lambda v: len(v) if isinstance(v, list) else 0)
xed["investigation_count"] = xed["investigations"].apply(lambda v: len(v) if isinstance(v, list) else 0)
xed["has_sdh"] = xed["sdh_flags"].apply(lambda v: bool(v))
xed["uncertainty_count"] = xed["uncertainty_phrases"].apply(lambda v: len(v) if isinstance(v, list) else 0)
xed["patient_talk_ratio"] = xed.apply(
    lambda r: (r["patient_word_count"] / (r["patient_word_count"] + r["doctor_word_count"]))
    if (r["patient_word_count"] + r["doctor_word_count"])>0 else None, axis=1
)

# Save structured data
xed.to_csv("visit_summaries_subset.csv", index=False)
print("Saved -> visit_summaries_subset.csv")

# ============================================
# STEP 10: BASIC ANALYSIS SNAPSHOT
# ============================================
from collections import Counter

# Top-5 medications
med_counter = Counter(itertools.chain.from_iterable(xed["medications"]))
top5_meds = med_counter.most_common(5)

# Small talk summary
prop_small_talk = float(xed["has_small_talk"].mean() or 0)
num_with_small_talk = int(xed["has_small_talk"].sum())
num_total = len(xed)

print("\nSummary Statistics")
print(f"Top-5 medications: {top5_meds}")
print(f"Small talk appears in {num_with_small_talk}/{num_total} dialogs ({prop_small_talk:.1%})")

# ============================================
# STEP 11: SDOH AND ADHERENCE INSIGHT
# ============================================
sdh = xed["sdh_flags"].explode().dropna()
sdh_top = sdh.value_counts()

sdh_summary = xed.groupby("has_sdh").agg(
    mean_plan_items=("plan_actions", lambda lsts: pd.Series([len(x) for x in lsts]).mean()),
    mean_symptoms=("symptom_count","mean"),
    n=("has_sdh","size")
).reset_index()

print("\nSocial Determinants of Health Summary")
print(sdh_summary)
print("\nTop SDOH mentions:")
print(sdh_top)

