# 🧠 LLM-as-a-Judge for Slate Recommendation Systems

**Evaluating whether a small, low-cost LLM can reliably judge ordered recommendation slates — and exposing how position bias breaks naive prompt engineering.**

This project investigates the use of `llama-3.1-8b-instant` via Groq as an **LLM-as-a-Judge** for **slate recommendation systems**. Instead of evaluating a single recommendation item, the model compares two ordered slates and predicts which slate better matches a user’s MovieLens preference history.

The core research question:

> Can advanced prompt engineering make a small, cheap LLM reliable enough to evaluate recommendation slates?

The answer from this experiment is clear: **not yet**. The small model shows severe position bias, limited swap consistency, and poor reliability under ensemble aggregation.

---

## 🔍 Project Overview

Slate recommendation is harder than standard item recommendation because the system must decide both:

1. **What** items to recommend.
2. **How** to order them.

This project builds an offline evaluation pipeline using MovieLens 1M. For each user:

- Five highly rated movies are sampled as the user history.
- Two candidate slates are created from the user’s remaining watched movies.
- Each slate contains three ranked movie items.
- Each movie is represented as `Title | Genres`.
- Ground truth is computed from the sum of the user’s actual ratings for each slate.

Each pair is evaluated twice:

1. **Original order:** Slate A then Slate B.
2. **Swapped order:** Slate B then Slate A.

This enables direct measurement of **position bias** using swap consistency.

---

## ✨ Key Findings

### 1. Severe Position Bias in the Minimal Baseline

The minimal JSON baseline collapsed into deterministic **recency bias**:

| Method | Accuracy | Swap Consistency |
|---|---:|---:|
| M1 Minimal JSON Baseline | 50.0% | 0.0% |

The 50.0% accuracy is misleading. Since the model consistently preferred the second local option, one of the two orderings often matched the ground truth by chance.

---

### 2. Structured Prompts Helped, But Not Enough

The best methods reached only **53.5% accuracy**:

| Method | Accuracy | Swap Consistency |
|---|---:|---:|
| M2 Compact Micro-Rubric | 49.0% | 24.0% |
| M3 PORTIA Rank-Interleaved | 53.5% | 31.0% |
| M4 Pointwise Scoring | 53.5% | 35.0% |

M3 and M4 improved consistency compared to the baseline, but the model remained too unstable for reliable evaluation.

---

### 3. Jury Ensemble Failed

The project tested an ensemble-style jury that aggregates M1-M4 using majority vote.

| Method | Accuracy | Swap Consistency |
|---|---:|---:|
| M7 Majority-Vote Jury | 42.0% | 9.0% |

McNemar’s test showed that the jury significantly degraded performance compared to the baseline:

```text
p-value = 0.000402
```

This happened because the jury combined **different prompts over the same biased model**. The errors were correlated, so majority voting amplified bias instead of cancelling it.

---

### 4. Token Cost Trade-offs Matter

Token usage becomes important if the same architecture is moved to frontier models.

| Method | Avg. Tokens | Notes |
|---|---:|---|
| M1 Baseline | 421.6 | Cheapest direct pairwise judge |
| M2 Micro-Rubric | 512.6 | Structured JSON criteria |
| M3 PORTIA | 503.6 | Best cost-quality trade-off |
| M4 Pointwise | ~990.0 | Includes hidden tie-breaker cost |
| M6 Audit | 510.1 | Cost per triggered audit |

M4 matched M3 in accuracy but required roughly double the effective token cost due to frequent tie-breaker fallback calls. Therefore, **M3 PORTIA was the best cost-quality trade-off** in this experiment.

---

## 🧪 Evaluated Prompting Methods

### M1 — Minimal JSON Baseline

A compact prompt asking only for the better slate:

```json
{"winner": "<A_or_B_or_TIE>"}
```

Purpose: isolate pure position bias.

---

### M2 — Compact Micro-Rubric

The model evaluates several criteria before choosing a winner:

```json
{
  "history_match": "<A_or_B_or_TIE>",
  "ordering_quality": "<A_or_B_or_TIE>",
  "diversity": "<A_or_B_or_TIE>",
  "winner": "<A_or_B_or_TIE>"
}
```

Purpose: test whether structured local criteria improve judgment stability.

---

### M3 — PORTIA Rank-Interleaved

Instead of showing full slates as blocks, the prompt compares rank-aligned items:

```text
Rank 1 comparison: A1 vs B1
Rank 2 comparison: A2 vs B2
Rank 3 comparison: A3 vs B3
```

Purpose: reduce block-level position bias by forcing local rank-by-rank comparison.

---

### M4 — Pointwise Scoring

The model scores each slate independently from 1 to 5:

```json
{"score": <INTEGER_1_TO_5>}
```

If both slates receive the same score, the pipeline falls back to M1.

Purpose: test whether removing direct pairwise presentation reduces position bias.

---

### M6 — Twin-Pass Audit

Triggered only when M2 selects the first local slate. A second prompt hides the final winner and asks whether the criteria support A, B, or neither.

Purpose: test whether the model can audit its own structured reasoning.

---

### M7 — Majority-Vote Jury

Aggregates M1-M4 using simple majority vote.

Purpose: test whether prompt-level ensembling improves robustness.

Result: it did not.

---

## 📁 Repository Structure

```text
.
├── data_prep.py
├── judge_eval.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── ml-1m/
│   ├── ml-1m.zip
│   ├── evaluation_dataset.csv
│   └── evaluation_results_groq.csv
│
└── outputs/
    ├── figures/
    └── reports/
```

### Main Files

| File | Description |
|---|---|
| `data_prep.py` | Downloads MovieLens 1M, builds the evaluation dataset, enriches movies with genres, computes ground truth, and creates difficulty buckets. |
| `judge_eval.py` | Runs the LLM evaluation pipeline using Groq, executes all prompting methods, saves checkpointed results, and prints statistical summaries. |
| `requirements.txt` | Python dependencies required for the project. |
| `data/evaluation_dataset.csv` | Generated slate-pair evaluation dataset. |
| `data/evaluation_results_groq.csv` | Final evaluation output from the Groq/Llama run. |

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it:

```bash
# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

The required packages are:

```text
pandas
groq
matplotlib
statsmodels
python-dotenv
```

---

## 🔑 API Key Setup

This project uses Groq to call `llama-3.1-8b-instant`.

Create a `.env` file in the root directory:

```bash
touch .env
```

Add your Groq API key:

```env
GROQ_API_KEY=
```

Do **not** commit your `.env` file to GitHub.

Never commit real API keys or secrets to this repository.

Recommended `.gitignore` entries:

```gitignore
.env
data/ml-1m/
data/ml-1m.zip
```

---

## 🚀 Usage

### Step 1 — Generate the Evaluation Dataset

```bash
python data_prep.py
```

This will:

- Download MovieLens 1M if needed.
- Load movie and rating data.
- Build 100 slate-pair examples.
- Add genre-enriched item text.
- Compute utility gaps and difficulty labels.
- Save the dataset to:

```text
data/evaluation_dataset.csv
```

---

### Step 2 — Run the LLM Judge Evaluation

```bash
python judge_eval.py
```

This will:

- Load `data/evaluation_dataset.csv`.
- Evaluate each pair in original and swapped order.
- Run M1, M2, M3, M4, M6, and M7.
- Save checkpointed results after every pair.
- Print the final statistical summary.

Output file:

```text
data/evaluation_results_groq.csv
```

---

## ♻️ Resume / Checkpointing

The evaluation script saves results after each `Pair_ID`.

If the run stops in the middle, simply run:

```bash
python judge_eval.py
```

again.

The script will detect already processed pairs and resume from the next unprocessed pair.

---

## 📊 Final Results Summary

```text
M1 Minimal JSON Baseline:
Accuracy = 0.500
Consistency = 0.000

M2 Compact Micro-Rubric:
Accuracy = 0.490
Consistency = 0.240

M3 PORTIA Rank-Interleaved:
Accuracy = 0.535
Consistency = 0.310

M4 Pointwise + Pairwise Hybrid:
Accuracy = 0.535
Consistency = 0.350

M7 Majority-Vote Jury:
Accuracy = 0.420
Consistency = 0.090
```

McNemar test comparing M1 vs M7:

```text
Contingency table = [[82, 18], [2, 98]]
p-value = 0.000402
```

Conclusion: the jury significantly degraded performance.

---

## 🧠 Research Conclusion

Small LLMs are attractive for low-cost evaluation pipelines, but this experiment shows that zero-shot prompt engineering alone is not enough.

The tested 8B model:

- Collapsed into recency bias under the minimal baseline.
- Improved only slightly under structured prompting.
- Failed to self-audit reliably.
- Produced correlated errors across prompt variants.
- Became worse when prompt outputs were ensembled by majority vote.

The project suggests that future work should focus on:

1. Task-specific fine-tuning or DPO.
2. Richer context injection.
3. Comparing fine-tuned small models against frontier models.
4. Dynamic routing between cheap and expensive judges based on slate difficulty.

---

## 📌 Notes

- The current experiment uses 100 slate pairs.
- Each pair is evaluated twice: original and swapped order.
- Slates contain 3 items each.
- User history contains 5 highly rated movies.
- Ground truth is based on the sum of actual MovieLens ratings.
- The current model is `llama-3.1-8b-instant`.

---

## 🧾 Citation Context

This project is motivated by recent work on:

- LLM-as-a-Judge evaluation and position bias.
- Slate recommendation as ordered list evaluation.
- Prompt engineering for bias mitigation.
- Cost-aware LLM evaluation pipelines.

---

## 📜 License

This repository is intended for academic and portfolio use. Add a license file before public release if you want to define reuse terms clearly.
