# MTS-Dialog & PubMed Literature Mining – Reproducible Pipeline

This repository contains two scripts designed to extract and analyze biomedical information using **SemLib** and **PubMed** data.

---

## Files Overview

| File | Description |
|------|--------------|
| `mts_dialog_extraction.py` | Extracts structured visit summaries (symptoms, medications, emotions, etc.) from the **MTS-Dialog** dataset using SemLib. |
| `pubmed_literature_mining.py` | Retrieves and processes **PubMed** abstracts related to Type 2 Diabetes and drug repurposing using SemLib and keyword extraction. |

---

## Dependencies

You will need **Python 3.9+** and the following packages:

```bash
pip install pandas biopython semlib nest_asyncio pydantic
```

> 💡 *Optional:* You may install in a virtual environment for reproducibility:
```bash
python3 -m venv venv
source venv/bin/activate
```

---

## Environment Variables

Both scripts use OpenAI models through **SemLib**, so you must set your API key:

```bash
export OPENAI_API_KEY="your-api-key"
```

The default model is set to `openai/gpt-4.1-mini`. You can adjust this in the environment configuration section of each script.

---

## Data Access

### **MTS-Dialog Extraction**
- The dataset is automatically loaded from:
  ```
  https://raw.githubusercontent.com/abachaa/MTS-Dialog/main/Main-Dataset/MTS-Dialog-TrainingSet.csv
  ```
- No manual download is required.

### **PubMed Literature Mining**
- The script retrieves abstracts via the **NCBI Entrez API**.  
  Set your email in the script (for Entrez compliance):
  ```python
  Entrez.email = "youremail@domain.com"
  ```

---

## ▶How to Run

### **1. MTS-Dialog Extraction**
This script extracts structured data from the doctor–patient dialogues.

```bash
python mts_dialog_extraction.py
```

**Output files:**
- `visit_summaries_subset.csv` – structured dialogue extractions
- Console summary statistics (top medications, small talk, SDOH summaries)

---

### **2. PubMed Literature Mining**
This script queries PubMed for Type 2 Diabetes studies and extracts candidate drug mentions.

```bash
python pubmed_literature_mining.py
```

**Output files:**
- `t2d_microbiome_findings.csv` – extracted keyword mentions
- Console display of candidate drugs categorized by therapeutic class

---

## Reproducibility Summary

To reproduce the full pipeline:
1. Clone this repository  
   ```bash
   git clone <your_repo_url>
   cd <repo_name>
   ```
2. Install dependencies (see above)
3. Export your API key
4. Run each script sequentially
5. Verify the generated CSV files

---