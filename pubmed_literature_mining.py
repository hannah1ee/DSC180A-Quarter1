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
# STEP 1: IMPORTS & ENVIRONMENT SETUP
# ============================================
import os, asyncio, re, nest_asyncio, pandas as pd
from Bio import Entrez
from semlib import Session
from semlib.cache import OnDiskCache

nest_asyncio.apply()

# ============================================
# STEP 2: API KEYS & SEMLIB CONFIG
# ============================================

os.environ["OPENAI_API_KEY"] = "your-sky-key"
os.environ["SEMLIB_DEFAULT_MODEL"] = "openai/gpt-4.1-mini"
os.environ["SEMLIB_MAX_CONCURRENCY"] = "3"

# Runtime Controls
BATCH_SIZE   = 20
PAUSE_SEC    = 5
MAX_RETRIES  = 5
MAX_CHARS    = 8000
SAMPLE_N     = 50



# ============================================
# STEP 3: RETRIEVE PUBMED ABSTRACTS (TYPE 2 DIABETES)
# ============================================
Entrez.email = "hal131@ucsd.edu"

query = (
    '("type 2 diabetes mellitus" OR "T2DM") '
    'AND ("drug repurposing" OR "drug repositioning" OR "therapeutic candidate" '
    'OR "antidiabetic mechanism" OR "AI drug discovery" OR "metabolic pathway")'
)

handle = Entrez.esearch(db="pubmed", term=query, retmax=100)
record = Entrez.read(handle)
ids = record["IdList"]
print(f"Retrieved {len(ids)} PubMed IDs.")

handle = Entrez.efetch(db="pubmed", id=",".join(ids), rettype="abstract", retmode="xml")
records = Entrez.read(handle)

articles = []
for article in records["PubmedArticle"]:
    try:
        pmid = article["MedlineCitation"]["PMID"]
        title = article["MedlineCitation"]["Article"]["ArticleTitle"]
        abstract = article["MedlineCitation"]["Article"]["Abstract"]["AbstractText"]
        if isinstance(abstract, list):
            abstract = " ".join(abstract)
        abstract = abstract.strip()[:MAX_CHARS]
        articles.append({"pmid": pmid, "title": title, "abstract": abstract})
    except KeyError:
        continue

df = pd.DataFrame(articles)

# ============================================
# STEP 4: INITIALIZE SEMLIB SESSION
# ============================================
session = Session(
    model=MODEL_NAME,
    cache=OnDiskCache("t2d_cache.db"),
    max_concurrency=int(os.getenv("SEMLIB_MAX_CONCURRENCY", "3"))
)

# ============================================
# STEP 5: PROMPT TEMPLATE
# ============================================
template = (
    "You are a biomedical literature analysis assistant.\n"
    "Below is an abstract from a PubMed paper related to **Type 2 Diabetes Mellitus (T2DM)**.\n\n"
    "{abstract}\n\n"
    "Your task is to identify **drugs, compounds, or therapeutic agents** mentioned as being "
    "repurposed, tested, or proposed as treatments for Type 2 Diabetes. For each, briefly summarize:\n"
    "- its proposed mechanism or biological rationale\n"
    "- the context (e.g., anti-inflammatory, insulin sensitization, β-cell protection)\n\n"
    "Format your response as plain text bullet points, for example:\n"
    "- DrugName: short description of mechanism or rationale"
)

sample_abstracts = df["abstract"].head(SAMPLE_N).tolist()

# ============================================
# STEP 6: RUN SEMLIB EXTRACTION (ASYNC-SAFE)
# ============================================

async def run_sem_extraction():
    results = []
    for i, abs_ in enumerate(sample_abstracts, 1):
        try:
            prompt = template.format(abstract=abs_)
            result = await session.prompt(prompt)
            results.append({"index": i, "output": str(result)})
        except Exception as e:
            results.append({"index": i, "output": f"Error: {e}"})
    return results


async def main():
    print("\nRunning SemLib extraction...")
    results = await run_sem_extraction()
    sem_df = pd.DataFrame(results)

    # ============================================
    # STEP 7: POST-PROCESSING TO EXTRACT DRUG NAMES
    # ============================================
    def extract_drug_terms(text):
        possible = re.findall(r'\b[A-Z][a-zA-Z\-]{2,}\b', text)
        stop = {
            "Diabetes", "Mellitus", "Type", "Insulin", "Study", "Treatment",
            "Patients", "Disease", "Metabolic", "Therapy", "Blood", "Sugar"
        }
        return [p for p in possible if p not in stop]

    sem_df["extracted_drugs"] = sem_df["output"].apply(extract_drug_terms)
    unique_drugs = sorted({d for lst in sem_df["extracted_drugs"] for d in lst})

    print("\nCandidate Drug Mentions (SemLib Semantic Extraction – Type 2 Diabetes):")
    for d in unique_drugs:
        print(f"- {d}")
    print(f"\nFound {len(unique_drugs)} unique candidate drugs or compounds.")

    # ============================================
    # STEP 8: FILTER TRUE DRUG CANDIDATES (POST-SEMLIB)
    # ============================================
    df_drugs = pd.DataFrame(unique_drugs, columns=["raw_term"])

    # ---- Step 1: Remove non-drug / biological terms ----
    non_drug_patterns = [
        r'EGFR', r'MAPK', r'JAK', r'ROS', r'RNA', r'DNA', r'Pathway', r'Receptor',
        r'Inflammation', r'Apoptosis', r'Oxidative', r'Reduction', r'Suppression',
        r'Induction', r'Expression', r'Protein', r'Enzyme', r'Factor'
    ]
    mask = df_drugs["raw_term"].apply(lambda x: not any(re.search(p, x, re.I) for p in non_drug_patterns))
    df_drugs = df_drugs[mask]

    # ---- Step 2: Require drug-like patterns ----
    drug_suffixes = (
        "formin","gliflozin","gliptin","glitazone","statin","pril","sartan",
        "olol","coxib","fibrate","tide","mine","fen","azole","afil"
    )
    known_drug_keywords = [
        "metformin","pioglitazone","rosiglitazone","phenformin",
        "sitagliptin","vildagliptin","linagliptin",
        "canagliflozin","dapagliflozin","empagliflozin",
        "liraglutide","semaglutide","exenatide","acarbose",
        "repaglinide","nateglinide","curcumin","resveratrol",
        "berberine","quercetin","tocopherol","umbelliferone",
        "atorvastatin","simvastatin","rosuvastatin","fenofibrate"
    ]
    def looks_like_drug(name):
        n = name.lower()
        return any(n.endswith(suf) for suf in drug_suffixes) or any(k in n for k in known_drug_keywords)

    df_drugs = df_drugs[df_drugs["raw_term"].apply(looks_like_drug)]

    # ---- Step 3: Clean & deduplicate ----
    df_drugs["raw_term"] = df_drugs["raw_term"].str.strip().str.replace(r'[^A-Za-z0-9-]', '', regex=True)
    df_drugs = df_drugs.drop_duplicates(subset=["raw_term"]).reset_index(drop=True)

    # ---- Step 4: Categorize drug classes ----
    def categorize(drug):
        dl = drug.lower()
        if any(k in dl for k in ["formin","glitazone","gliptin","gliflozin"]):
            return "Antidiabetic / Glucose-lowering"
        if any(k in dl for k in ["statin","fibrate","sartan","pril"]):
            return "Cardiometabolic comorbidity"
        if any(k in dl for k in ["curcumin","resveratrol","berberine","quercetin","tocopherol","umbelliferone","myricetin","caffeic"]):
            return "Natural compound / Antioxidant"
        if any(k in dl for k in ["coxib","ibuprofen","aspirin","celecoxib"]):
            return "Anti-inflammatory adjunct"
        return "Other / Investigational"

    df_drugs["category"] = df_drugs["raw_term"].apply(categorize)

    # ---- Step 5: Display results ----
    print("\nCleaned Drug Candidates (Type 2 Diabetes):")
    for cat in df_drugs["category"].unique():
        subset = df_drugs[df_drugs["category"] == cat]["raw_term"].tolist()
        print(f"\n{cat} ({len(subset)}):")
        print(", ".join(subset))

# ============================================
# ENTRY POINT
# ============================================
if __name__ == "__main__":
    asyncio.run(main())


# ============================================
# KEYWORD SEARCH VERSION
# ============================================

import importlib
import requests, re, pandas as pd
from pydantic import BaseModel

# --- 1. PubMed search for Type 2 Diabetes and gut microbiome ---
query = "Type 2 Diabetes AND gut microbiome AND (treatment OR metabolites)"
base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
r = requests.get(f"{base}esearch.fcgi?db=pubmed&term={query}&retmax=50&retmode=json")
ids = r.json()["esearchresult"]["idlist"]
print(f"\nNow Keyword Searching: \nRetrieved {len(ids)} PubMed IDs")

# --- 2. Fetch abstracts ---
fetch = requests.get(
    f"{base}efetch.fcgi?db=pubmed&id={','.join(ids)}&rettype=abstract&retmode=text"
).text
abstract_blocks = [a.strip() for a in re.split(r"PMID-\s*\d+", fetch) if a.strip()]
min_len = min(len(ids), len(abstract_blocks))
data = [{"pmid": ids[i], "abstract": abstract_blocks[i]} for i in range(min_len)]
df = pd.DataFrame(data)
print(f"\nCombined {len(df)} abstracts successfully\n")

# --- 3. Schema and keywords ---
class Finding(BaseModel):
    term: str
    sentence: str

keywords = [
    "metformin", "probiotic", "butyrate", "berberine",
    "resveratrol", "curcumin", "microbiota", "insulin",
    "inflammation", "short-chain fatty", "therapy", "compound"
]

# --- 4. Extraction ---
hits=[]
for abs_ in df["abstract"]:
    for sent in re.split(r'(?<=[.!?]) +', abs_):
        for k in keywords:
            if re.search(rf"\b{k}\w*\b", sent, re.I):
                hits.append(Finding(term=k.capitalize(), sentence=sent.strip()))

# --- 5. Output ---
if hits:
    out = pd.DataFrame([h.dict() for h in hits]).drop_duplicates()
    print(f"\nExtracted {len(out)} relevant mentions:\n")
    print(out.head(10))
    out.to_csv("t2d_microbiome_findings.csv", index=False)
else:
    print("No results found — try adjusting keywords or retmax.")