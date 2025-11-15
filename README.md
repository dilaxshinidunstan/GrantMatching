
# 🚀 Grant Matching Application: End-to-End Demonstration

This document demonstrates the full grant matching pipeline, taking a company profile, extracting structured features, performing a hybrid vector search (ChromaDB + Gemini Embeddings), and finally using a Gemini 2.5 Flash "Judge" to rank and justify the top grants

## 1\. Setup and Initialization (Required Imports)

This section includes all necessary imports, API setup, and the Chroma client configuration.

```python
import os
import json
import time
from typing import Union, Dict, Any, List
import pandas as pd
import chromadb
from google import genai
from google.genai.types import EmbedContentConfig
from google.genai import errors as genai_errors
from google.genai import types

# --- Gemini setup ---
# NOTE: Ensure GEMINI_API_KEY is set in your environment or Colab secrets.
client = genai.Client(api_key=GEMINI_API_KEY)
EMBED_MODEL = "gemini-embedding-001"
JUDGE_MODEL = "gemini-2.5-flash"

# --- Chroma setup ---
# Re-use the existing client and collection (assuming ingestion ran successfully)
chroma_client = chromadb.PersistentClient(path="./chroma_canada_grants")
collection = chroma_client.get_or_create_collection(name="canada_grants")

# --- DataFrame Loading (for final display) ---
# NOTE: This assumes the data path is correct and the initial 50 rows were indexed correctly.
DATA_PATH = "/content/drive/MyDrive/Senzmate/Week04/Extracted_Details.xlsx"
df_grants = pd.read_excel(DATA_PATH).head(50)
```

---

## 2\. Core Functions: Extraction, Search, and Judge

### 2.1 Feature Extraction Function

Uses Gemini's JSON mode to convert unstructured details into a strict company profile.

```python
def extract_company_features(company_detail: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Uses Gemini to extract structured features from company details (JSON or text).
    Returns features in the required format, e.g., amount in millions.
    """
    if isinstance(company_detail, dict):
        input_text = json.dumps(company_detail, indent=2)
    else:
        input_text = str(company_detail)

    prompt = f"""
You are an information extraction assistant for startup funding.

Given details about a company, extract the following fields and return a strict JSON object:
{{
  "sector": "string, primary industry/sector of company",
  "amount": 0.0,
  "country": "string, country where the company is headquartered",
  "region": "string, city/state/region where the company is located",
  "operation": "string, core operations of the company",
  "business_type": ["string", "..."],
  "used_for": "string, activities the requested funds will be used for",
  "years": 0,
  "employees": 0
}}

Important rules:
- "amount": funding amount sought, in millions (e.g. 0.01 for $10,000). Numeric only.
- "business_type": array of strings like ["for-profit", "private limited company"].
- "region": prefer the city if known, else state/region.
- Infer reasonable values only when strongly implied (e.g., "Ltd." implies private limited company). If you cannot infer a value, use null.
- Return ONLY the JSON object.

Company details:
{input_text}
"""

    response = client.models.generate_content(
        model=JUDGE_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.0,
            max_output_tokens=1024,
        ),
    )
    return json.loads(response.text)
```

### 2.2 Hybrid Search Functions

Define helper functions for building the query, embedding it, creating the metadata filter, and executing the search.

```python
def build_query_text(company: Dict[str, Any]) -> str:
    """Combines key company features into a single text query for embedding."""
    parts = []
    if company.get("sector"): parts.append(f"Sector: {company['sector']}")
    if company.get("operation"): parts.append(f"Core operations: {company['operation']}")
    if company.get("used_for"): parts.append(f"Funding used for: {company['used_for']}")
    if company.get("business_type"):
        bt = company["business_type"]
        if isinstance(bt, list): bt = ", ".join(bt)
        parts.append(f"Business type: {bt}")
    if company.get("country"): parts.append(f"Country: {company['country']}")
    if company.get("region"): parts.append(f"Region: {company['region']}")
    return " | ".join(parts)

def embed_query(text: str) -> List[float]:
    """Embeds the query text using Gemini Embedding model (RETRIEVAL_QUERY task type)."""
    resp = client.models.embed_content(
        model=EMBED_MODEL,
        contents=[text],
        config=EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    return resp.embeddings[0].values

def build_meta_filter(company: Dict[str, Any]) -> Dict[str, Any]:
    """
    Builds a Chroma 'where' filter based on company attributes
    (country, region, years, employees, amount, percentage) [cite_start][cite: 4].
    """
    conditions = []

    if company.get("country"):
        conditions.append({"country": company["country"]})

    if company.get("region"):
        conditions.append({"region": company["region"]})

    if company.get("amount") is not None:
        conditions.append({"amount": {"$gte": company["amount"]}})

    if company.get("years") is not None:
        years = company["years"]
        conditions.append({"min_year": {"$lte": years}})
        conditions.append({"max_year": {"$gte": years}})

    if company.get("employees") is not None:
        emp = company["employees"]
        conditions.append({"min_employees": {"$lte": emp}})
        conditions.append({"max_employees": {"$gte": emp}})

    return {"$and": conditions} if conditions else {}

def search_grants_for_company(company_features: Dict[str, Any], top_k: int = 15) -> Dict[str, Any]:
    [cite_start]"""Performs hybrid search (vector similarity + metadata filter) in Chroma[cite: 4, 9]."""
    query_text = build_query_text(company_features)
    query_emb = embed_query(query_text)
    where_filter = build_meta_filter(company_features)

    result = collection.query(
        query_embeddings=[query_emb],
        n_results=top_k,
        where=where_filter if where_filter else None,
        include=["documents", "metadatas", "distances"],
    )

    matches = []
    ids = result.get("ids", [[]])[0]
    docs = result.get("documents", [[]])[0]
    metas = result.get("metadatas", [[]])[0]
    dists = result.get("distances", [[]])[0]

    for gid, doc, meta, dist in zip(ids, docs, metas, dists):
        matches.append({
            "grant_id": gid,
            "score": float(dist),   # vector distance (smaller = better)
            "text": doc,            # Combined text (Sector, activities, etc.)
            "metadata": meta,
        })

    return {
        "company": company_features,
        "query_text": query_text,
        "where": where_filter,
        "matches": matches,
    }
```

### 2.3 The LLM Judge Function (Scoring and Justification)

Uses Gemini's structured output capability to apply complex scoring logic and generate justifications.

```python
def score_grants_with_gemini(
    company_features: Dict[str, Any],
    search_result: Dict[str, Any],
    max_candidates: int = 5,
    max_results: int = 3,
) -> Dict[str, Any]:
    """
    Uses Gemini 2.5 Flash as a judge to re-score and justify top grant matches.
    Returns the ranked list including grant_id, fit_score, eligibility, reasons, risks.
    """
    matches = search_result.get("matches", [])
    if not matches:
        return {"company": company_features, "ranked_grants": []}

    # Prepare candidates: only send necessary data (remove large 'text' field)
    candidates = []
    for match in matches[:max_candidates]:
        candidates.append({
            "grant_id": match["grant_id"],
            "grant_metadata": match["metadata"],
            "retrieval_score": match["score"]
        })

    # Define the strict output schema
    output_schema = {
        "type": "object",
        "properties": {
            "ranked_grants": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "grant_id": {"type": "string"},
                        "fit_score": {"type": "integer", "description": "0-100, higher is better"},
                        "eligibility": {"type": "string", "enum": ["eligible", "probably_eligible", "ineligible"]},
                        "reasons": {"type": "array", "items": {"type": "string"}},
                        "risks": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["grant_id", "fit_score", "eligibility", "reasons", "risks"]
                }
            }
        },
        "required": ["ranked_grants"]
    }

    # The prompt defines the rules for the judge
    prompt = f"""
You are an expert grant-matching assistant.
Your task is to analyze the company profile against the candidate grants' metadata and provide a ranked list.

Company Profile:
{json.dumps(company_features, indent=2)}

Candidate Grants (metadata only):
{json.dumps(candidates, indent=2)}

Analysis Rules:
1.  **Eligibility**: Use "eligible" if all criteria match, "ineligible" if a hard rule is broken (e.g., requested amount > max amount, years/employees outside min/max range), and "probably_eligible" for borderline cases (e.g., missing data, region mismatch).
2.  **Fit Score (0-100)**: Rank grants based on overall fit, rewarding strong sector/operation alignment, amount being well within limits, and location match.
3.  **Reasons/Risks**: 'reasons' must highlight strong matches. 'risks' must highlight mismatches or concerns.
4.  **Final Output**: Sort by 'fit_score' descending. Only include the top {max_results} grants. Return ONLY the JSON object.
"""

    response = client.models.generate_content(
        model=JUDGE_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=output_schema,
            temperature=0.1,
        ),
    )

    ranked = json.loads(response.text)
    ranked["company"] = company_features
    return ranked
```

---

## 3\. End-to-End Demonstration and Final Output

This section runs the entire pipeline for a sample company and prints the final, required ranked output.

### 3.1 Define Sample Company Input

```python
# Sample company input
sample_company_detail = {
    "description": "EcoFuels Ltd. is an ecological company from Toronto, Canada. We are seeking $10,000 in funding to develop hydrogen and biofuel technologies. We are a low-carbon company with 50 employees, operating for 3 years. Involve in production related to low-carbon technology",
    "industry/sector": "Sustainable Energy",
    "city": "Toronto",
    "country": "Canada",
}
```

### 3.2 Execute Pipeline (Extract → Search → Judge)

```python
# 1. Extract structured features
company_features = extract_company_features(sample_company_detail)

# 2. Perform Hybrid Search
# Use the company features to perform semantic search + initial metadata filtering.
res = search_grants_for_company(company_features, top_k=15)

# 3. Apply LLM Judge for final scoring and justification
ranked_results = score_grants_with_gemini(company_features, res, max_candidates=5, max_results=3)

# Filter the original DataFrame to show the full grant details alongside the score
final_grant_ids = [g['grant_id'] for g in ranked_results.get('ranked_grants', [])]
final_df = df_grants[df_grants.index.astype(str).isin(final_grant_ids)]

# Combine the ranking data with the full grant details
final_output = []
ranking_data = {g['grant_id']: g for g in ranked_results.get('ranked_grants', [])}

for index, row in final_df.iterrows():
    grant_id_str = str(index)
    if grant_id_str in ranking_data:
        ranking = ranking_data[grant_id_str]

        # [cite_start]Display the required fields: fit_score, eligibility, reasons, risks [cite: 10]
        final_output.append({
            "Grant ID": grant_id_str,
            "Fit Score": ranking['fit_score'],
            "Eligibility": ranking['eligibility'],
            "Reasons": "\n".join([f"* {r}" for r in ranking['reasons']]),
            "Risks": "\n".join([f"* {r}" for r in ranking['risks']]) if ranking['risks'] else "None",
            "Grant Name (for reference)": row['Funding program'],
            "Max Funding (M)": row['amount'],
            "Max Coverage (%)": row['percentage'],
        })

final_demonstration_df = pd.DataFrame(final_output).sort_values(by="Fit Score", ascending=False).reset_index(drop=True)
```

### 3.3 Final Ranked Grants Table (Required Output)

This table provides the final ranked list with all required fields: **`grant_id`**, **`fit_score`**, **`eligibility`**, **`reasons`**, and **`risks`**.

```python
# Print the final DataFrame
print("\n### 🏆 Final Ranked Grants (LLM Judge Output) 🏆\n")
print(final_demonstration_df.to_string())
```

|       | Grant ID | Fit Score | Eligibility           | Reasons                                                                                                                                                                                       | Risks                                                                           | Grant Name (for reference)                       | Max Funding (M) | Max Coverage (%) |
| ----- | -------- | --------- | --------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- | ------------------------------------------------ | --------------- | ---------------- |
| **0** | **36**   | **95**    | **eligible**          | Strong sector match to 'Cleantech' and 'Health Technology.'The requested $0.01M is far below the max $175,000. Company is in a valid operation year (3) and employee count (50). | None                                                                            | Canadian International Innovation Program (CIIP) | 175000.0        | 80.0             |
| **1** | **9**    | **90**    | **probably_eligible** | Very strong sector alignment: 'Clean Technologies' and 'Sustainability.' Requested amount ($0.01M) is well within the $5.0M maximum. Broad eligibility for years and employees.     |  The grant is primarily focused on agriculture-related clean technology.      | Agricultural Clean Technology (ACT) Program      | 5.0             | 75.0             |
| **2** | **3**    | **70**    | **eligible**          |  Good secondary match on 'Clean Technology' sector. Requested amount is minimal compared to the $10.0M maximum. No hard eligibility barriers (years, employees).                  |  Primary sector is 'Aerospace Innovation,' a weak fit for Sustainable Energy. | Aerospace Regional Recovery Initiative (ARRI)    | 10.0            | 10.0             |

---

**You can now copy the entire text above and paste it into a new document file.**
