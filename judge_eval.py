import json
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any
import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from statsmodels.stats.contingency_tables import mcnemar
import random


# ============================================================
# Configuration
# ============================================================

DATA_DIR = Path("data")
INPUT_DATASET_PATH = DATA_DIR / "evaluation_dataset.csv"
OUTPUT_RESULTS_PATH = DATA_DIR / "evaluation_results_groq.csv"  

MODEL_NAME = "llama-3.1-8b-instant"
TEMPERATURE = 0.0
REQUEST_DELAY_SECONDS = 0.25    


ACTIVE_PROVIDER = "groq" 

GROQ_MODEL_NAME = "llama-3.1-8b-instant"


load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# ============================================================
# JSON helpers
# ============================================================

JSON_SYSTEM_MESSAGE = {
    "role": "system",
    "content": (
        "You are a strict JSON API. "
        "Return only valid JSON. "
        "Do not include markdown, explanations, comments, or extra text."
    ),
}


def extract_first_json_object(text: str) -> str:
    """
    Extract the first balanced JSON object from a model response.
    Handles markdown wrappers and extra text safely.
    """
    if not isinstance(text, str):
        return "{}"

    cleaned = text.strip()

    # Remove markdown wrappers such as ```json ... ```
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    start = cleaned.find("{")
    if start == -1:
        return "{}"

    depth = 0
    in_string = False
    escape = False

    for idx in range(start, len(cleaned)):
        ch = cleaned[idx]

        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return cleaned[start : idx + 1]

    return "{}"


def safe_json_loads(text: str) -> dict[str, Any]:
    """Parse JSON safely. Return empty dict on failure."""
    candidate = extract_first_json_object(text)

    try:
        obj = json.loads(candidate)
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        return {}


def normalize_choice(
    value: Any,
    valid_choices: tuple[str, ...] = ("A", "B", "TIE"),
    default: str = "TIE",
) -> str:
    """Normalize model output into a valid choice."""
    if value is None:
        return default

    normalized = str(value).strip().upper()
    normalized = normalized.replace("[[", "").replace("]]", "")

    if normalized in valid_choices:
        return normalized

    return default


def parse_winner(response_text: str) -> tuple[str, dict[str, Any]]:
    """Parse {'winner': 'A'|'B'|'TIE'} safely."""
    obj = safe_json_loads(response_text)
    winner = normalize_choice(obj.get("winner"), ("A", "B", "TIE"), default="TIE")
    return winner, obj


def parse_supports(response_text: str) -> tuple[str, dict[str, Any]]:
    """Parse {'supports': 'A'|'B'|'NEITHER'} safely."""
    obj = safe_json_loads(response_text)
    supports = normalize_choice(
        obj.get("supports"),
        ("A", "B", "NEITHER"),
        default="NEITHER",
    )
    return supports, obj


def parse_score(response_text: str) -> tuple[int | None, dict[str, Any]]:
    """Parse {'score': 1-5} safely."""
    obj = safe_json_loads(response_text)

    try:
        score = int(round(float(obj.get("score"))))
    except (TypeError, ValueError):
        return None, obj

    if 1 <= score <= 5:
        return score, obj

    return None, obj


# ============================================================
# API helper
# ============================================================

def call_llm(
    messages: list[dict[str, str]],
    max_tokens: int = 200,
    expect_json: bool = True,
    max_retries: int = 4
) -> tuple[str, int]:
    """
    Send request to Groq with Exponential Backoff for Rate Limits.
    Incorporates micro-throttling to prevent RPM quota exhaustion.
    """
    kwargs = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": TEMPERATURE,
        "max_tokens": max_tokens,
    }

    if expect_json:
        kwargs["response_format"] = {"type": "json_object"}

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content or "{}"
            tokens = int(response.usage.total_tokens)
            
            # Rate shaping: Add a consistent delay after every successful call.
            # This ensures we process at ~20 RPM, staying safely below Groq's 30 RPM limit.
            # It prevents the token bucket from emptying and avoids 60-second penalty blocks.
            time.sleep(2.5) 
            
            return content, tokens
            
        except Exception as exc:
            err_msg = str(exc).lower()
            
            # If we hit a Rate Limit (429), trigger exponential backoff.
            if "rate limit" in err_msg or "429" in err_msg:
                wait_time = 60 * (attempt + 1)
                print(f"\n[!] Rate limit hit. Waiting {wait_time} seconds before attempt {attempt + 1}/{max_retries}...")
                time.sleep(wait_time)
                continue
            
            # If the model explicitly complains about JSON formatting, fallback to raw text.
            elif expect_json and ("format" in err_msg or "type" in err_msg):
                kwargs.pop("response_format", None)
                expect_json = False
                continue
                
            # Unhandled errors (e.g., connection drops)
            else:
                print(f"API Error: {exc}")
                return "{}", 0

    print("Max retries reached. Returning empty JSON.")
    return "{}", 0



# ============================================================
# Prompt builders
# ============================================================

def build_minimal_baseline_prompt(history: str, slate_a: str, slate_b: str) -> str:
    return f"""
You must evaluate the actual movie data below.

<User_History>
{history}
</User_History>

<Slate_A>
{slate_a}
</Slate_A>

<Slate_B>
{slate_b}
</Slate_B>

Task:
Choose which recommendation slate better fits the user's history.
Use movie titles and genres. Do not choose based on the label A/B.
If both slates are similarly good or the evidence is unclear, choose TIE.

Output rules:
Return exactly one valid JSON object.
Use this schema, but replace the placeholder with your real decision:
{{"winner": "<A_or_B_or_TIE>"}}

Allowed values:
- "A" if Slate_A is better
- "B" if Slate_B is better
- "TIE" if neither slate is clearly better

Do not copy the placeholder.
Do not add explanations.
""".strip()


def build_micro_rubric_prompt(history: str, slate_a: str, slate_b: str) -> str:
    return f"""
You must evaluate the actual movie data below.

<User_History>
{history}
</User_History>

<Slate_A>
{slate_a}
</Slate_A>

<Slate_B>
{slate_b}
</Slate_B>

Task:
Compare Slate_A and Slate_B using the user's movie history.
Use titles and genres as evidence.

Criteria:
1. history_match: Which slate better matches the user's past liked genres/titles?
2. ordering_quality: Which slate places stronger matches earlier in the ordered list?
3. diversity: Which slate has better variety while still fitting the user?
4. winner: Overall better slate after considering the three criteria.

Output rules:
Return exactly one valid JSON object.
Use this schema, but replace every placeholder with a real decision:
{{
  "history_match": "<A_or_B_or_TIE>",
  "ordering_quality": "<A_or_B_or_TIE>",
  "diversity": "<A_or_B_or_TIE>",
  "winner": "<A_or_B_or_TIE>"
}}

Allowed values for every field:
- "A"
- "B"
- "TIE"

Do not copy the placeholders.
Do not add explanations.
""".strip()


def split_slate_items(slate_text: str) -> list[str]:
    """Split a ranked slate text into clean item strings."""
    items = []

    for line in slate_text.splitlines():
        line = line.strip()
        if not line:
            continue

        # Remove leading ranking prefixes like "1. "
        line = re.sub(r"^\d+\.\s*", "", line)
        items.append(line)

    return items


def build_rank_interleaved_text(slate_a: str, slate_b: str) -> str:
    """
    PORTIA-style rank interleaving:
    Rank 1 A vs Rank 1 B, Rank 2 A vs Rank 2 B, etc.
    """
    items_a = split_slate_items(slate_a)
    items_b = split_slate_items(slate_b)

    max_len = max(len(items_a), len(items_b))
    blocks = []

    for idx in range(max_len):
        a_item = items_a[idx] if idx < len(items_a) else "MISSING"
        b_item = items_b[idx] if idx < len(items_b) else "MISSING"

        blocks.append(
            f"Rank {idx + 1} comparison:\n"
            f"A{idx + 1}: {a_item}\n"
            f"B{idx + 1}: {b_item}"
        )

    return "\n\n".join(blocks)


def build_portia_prompt(history: str, slate_a: str, slate_b: str) -> str:
    interleaved = build_rank_interleaved_text(slate_a, slate_b)

    return f"""
You must evaluate the actual rank-by-rank movie data below.

<User_History>
{history}
</User_History>

<Rank_Interleaved_Slate_Comparison>
{interleaved}
</Rank_Interleaved_Slate_Comparison>

Task:
Compare the two slates rank-by-rank.
For each rank, decide whether the A item or B item better fits the user's history.
Then choose the overall better slate.
Use movie titles and genres. Do not choose based on label position.

Output rules:
Return exactly one valid JSON object.
Use this schema, but replace every placeholder with a real decision:
{{
  "rank1": "<A_or_B_or_TIE>",
  "rank2": "<A_or_B_or_TIE>",
  "rank3": "<A_or_B_or_TIE>",
  "winner": "<A_or_B_or_TIE>"
}}

Allowed values for every field:
- "A"
- "B"
- "TIE"

Do not copy the placeholders.
Do not add explanations.
""".strip()


def build_pointwise_score_prompt(history: str, slate: str) -> str:
    return f"""
You must score the actual recommendation slate below.

<User_History>
{history}
</User_History>

<Slate>
{slate}
</Slate>

Task:
Score how well this slate fits the user's history.
Use movie titles and genres.
Evaluate this slate independently. Do not compare it to any missing slate.

Scoring scale:
1 = very poor fit
2 = weak fit
3 = acceptable fit
4 = strong fit
5 = excellent fit

Output rules:
Return exactly one valid JSON object.
Use this schema, but replace the placeholder with a real integer:
{{"score": <INTEGER_1_TO_5>}}

The value of "score" must be an integer, not a string.
Do not copy the placeholder.
Do not add explanations.
""".strip()




def build_twin_pass_audit_prompt(
    history: str,
    slate_a: str,
    slate_b: str,
    criteria_json_without_winner: dict[str, Any],
) -> str:
    hidden_reasoning = json.dumps(
        criteria_json_without_winner,
        ensure_ascii=False,
        indent=2,
    )

    return f"""
You are an objective third-party auditor.

You are given:
1. The user history
2. Slate_A
3. Slate_B
4. An anonymous criterion-level evaluation

The original final winner has been hidden.

<User_History>
{history}
</User_History>

<Slate_A>
{slate_a}
</Slate_A>

<Slate_B>
{slate_b}
</Slate_B>

<Anonymous_Criteria_JSON>
{hidden_reasoning}
</Anonymous_Criteria_JSON>

Task:
Decide what the anonymous criterion-level evaluation supports.
Do not infer the original judge's intended answer.
Use the criteria only as audit evidence.
If the criteria are mixed, empty, invalid, or not strong enough, choose NEITHER.

Output rules:
Return exactly one valid JSON object.
Use this schema, but replace the placeholder with your real audit decision:
{{"supports": "<A_or_B_or_NEITHER>"}}

Allowed values:
- "A" if the criteria support Slate_A
- "B" if the criteria support Slate_B
- "NEITHER" if the criteria do not clearly support either slate

Do not copy the placeholder.
Do not apologize.
Do not add explanations.
""".strip()

# ============================================================
# Verdict mapping and aggregation
# ============================================================

def map_raw_to_original_label(raw_verdict: str, first_label: str, second_label: str) -> str:
    """
    Map local prompt labels back to original dataset labels.

    In Original order:
        local A -> original A
        local B -> original B

    In Swapped order:
        local A -> original B
        local B -> original A
    """
    if raw_verdict == "A":
        return first_label
    if raw_verdict == "B":
        return second_label
    return "TIE"


def majority_vote(verdicts: list[str]) -> str:
    """
    Simple Python-only jury over M1-M4.

    If the top vote is tied, return TIE.
    Otherwise return the most common label.
    """
    counts = Counter(verdicts)
    most_common = counts.most_common()

    if not most_common:
        return "TIE"

    top_label, top_count = most_common[0]

    # Tie for first place.
    if len(most_common) > 1 and most_common[1][1] == top_count:
        return "TIE"

    return top_label


# ============================================================
# Method runners
# ============================================================

def run_method_1_baseline(history: str, slate_a: str, slate_b: str) -> dict[str, Any]:
    prompt = build_minimal_baseline_prompt(history, slate_a, slate_b)
    response, tokens = call_llm([JSON_SYSTEM_MESSAGE, {"role": "user", "content": prompt}], max_tokens=80)
    winner, parsed = parse_winner(response)

    return {
        "raw_winner": winner,
        "tokens": tokens,
        "response": response,
        "parsed": parsed,
        "messages": [JSON_SYSTEM_MESSAGE, {"role": "user", "content": prompt}],
    }


def run_method_2_micro_rubric(history: str, slate_a: str, slate_b: str) -> dict[str, Any]:
    prompt = build_micro_rubric_prompt(history, slate_a, slate_b)
    response, tokens = call_llm([JSON_SYSTEM_MESSAGE, {"role": "user", "content": prompt}], max_tokens=160)
    parsed = safe_json_loads(response)

    winner = normalize_choice(parsed.get("winner"), ("A", "B", "TIE"), default="TIE")

    return {
        "raw_winner": winner,
        "tokens": tokens,
        "response": response,
        "parsed": parsed,
    }


def run_method_3_portia(history: str, slate_a: str, slate_b: str) -> dict[str, Any]:
    prompt = build_portia_prompt(history, slate_a, slate_b)
    response, tokens = call_llm([JSON_SYSTEM_MESSAGE, {"role": "user", "content": prompt}], max_tokens=180)
    parsed = safe_json_loads(response)

    winner = normalize_choice(parsed.get("winner"), ("A", "B", "TIE"), default="TIE")

    return {
        "raw_winner": winner,
        "tokens": tokens,
        "response": response,
        "parsed": parsed,
    }


def run_method_4_pointwise_hybrid(
    history: str,
    slate_a: str,
    slate_b: str,
    baseline_raw_winner: str,
) -> dict[str, Any]:
    prompt_a = build_pointwise_score_prompt(history, slate_a)
    response_a, tokens_a = call_llm(
        [JSON_SYSTEM_MESSAGE, {"role": "user", "content": prompt_a}],
        max_tokens=80,
    )
    score_a, parsed_a = parse_score(response_a)

    prompt_b = build_pointwise_score_prompt(history, slate_b)
    response_b, tokens_b = call_llm(
        [JSON_SYSTEM_MESSAGE, {"role": "user", "content": prompt_b}],
        max_tokens=80,
    )
    score_b, parsed_b = parse_score(response_b)

    if score_a is not None and score_b is not None and score_a > score_b:
        winner = "A"
    elif score_a is not None and score_b is not None and score_b > score_a:
        winner = "B"
    else:
        # Tie or parsing failure: use M1 as pairwise tie-breaker.
        winner = baseline_raw_winner

    return {
        "raw_winner": winner,
        "score_a": score_a,
        "score_b": score_b,
        "tokens": tokens_a + tokens_b,
        "response_a": response_a,
        "response_b": response_b,
        "parsed_a": parsed_a,
        "parsed_b": parsed_b,
    }



def run_method_6_twin_pass(
    history: str,
    slate_a: str,
    slate_b: str,
    method_2_raw_winner: str,
    method_2_parsed: dict[str, Any],
) -> dict[str, Any]:
    """
    Triggered only if Method 2 chooses the first option, local Slate A.
    """
    triggered = method_2_raw_winner == "A"

    if not triggered:
        return {
            "triggered": False,
            "raw_winner": method_2_raw_winner,
            "supports": "",
            "tokens": 0,
            "response": "",
            "parsed": {},
            "flipped": False,
        }

    # Hide final winner. Keep only criterion-level fields.
    criteria_only = {
        key: value
        for key, value in method_2_parsed.items()
        if key != "winner"
    }

    prompt = build_twin_pass_audit_prompt(
        history=history,
        slate_a=slate_a,
        slate_b=slate_b,
        criteria_json_without_winner=criteria_only,
    )

    response, tokens = call_llm(
        [JSON_SYSTEM_MESSAGE, {"role": "user", "content": prompt}],
        max_tokens=100,
    )

    supports, parsed = parse_supports(response)

    if supports == "NEITHER":
        winner = "TIE"
    else:
        winner = supports

    return {
        "triggered": True,
        "raw_winner": winner,
        "supports": supports,
        "tokens": tokens,
        "response": response,
        "parsed": parsed,
        "flipped": winner != method_2_raw_winner,
    }


# ============================================================
# Statistical summary
# ============================================================

def consistency_rate(results_df: pd.DataFrame, method_prefix: str) -> tuple[float, int]:
    """
    Consistency rate across Original vs Swapped order.

    A pair is consistent if the mapped original-label verdict is identical
    in both prompt orders.
    """
    verdict_col = f"{method_prefix}_Verdict"

    pivot = results_df.pivot_table(
        index="Pair_ID",
        columns="Order",
        values=verdict_col,
        aggfunc="first",
    )

    valid = pivot.dropna(subset=["Original", "Swapped"])

    if len(valid) == 0:
        return 0.0, 0

    rate = (valid["Original"] == valid["Swapped"]).mean()
    return float(rate), int(len(valid))


def accuracy(results_df: pd.DataFrame, method_prefix: str) -> float:
    """Order-level accuracy against original dataset ground truth."""
    verdict_col = f"{method_prefix}_Verdict"
    return float((results_df[verdict_col] == results_df["Ground_Truth"]).mean())


def sycophancy_flip_rate(results_df: pd.DataFrame, method_prefix: str) -> tuple[float, int, int]:
    """Flip rate among triggered rows only."""
    triggered_col = f"{method_prefix}_Triggered"
    flipped_col = f"{method_prefix}_Flipped"

    triggered_df = results_df[results_df[triggered_col] == True]

    if len(triggered_df) == 0:
        return 0.0, 0, 0

    flips = int(triggered_df[flipped_col].sum())
    total = int(len(triggered_df))
    return flips / total, flips, total


def run_mcnemar_baseline_vs_jury(results_df: pd.DataFrame) -> None:
    """Run McNemar test comparing M1 baseline accuracy vs M7 jury accuracy."""
    m1_correct = results_df["M1_Verdict"] == results_df["Ground_Truth"]
    m7_correct = results_df["M7_Verdict"] == results_df["Ground_Truth"]

    both_correct = int((m1_correct & m7_correct).sum())
    m1_only = int((m1_correct & ~m7_correct).sum())
    m7_only = int((~m1_correct & m7_correct).sum())
    both_wrong = int((~m1_correct & ~m7_correct).sum())

    table = [[both_correct, m1_only], [m7_only, both_wrong]]

    result = mcnemar(table, exact=True)

    print("\nMcNemar Test: M1 Baseline Accuracy vs M7 Jury Accuracy")
    print("Contingency table [[both_correct, M1_only], [M7_only, both_wrong]]:")
    print(table)
    print(f"P-value: {result.pvalue:.6f}")

    if result.pvalue < 0.05:
        print("Result: statistically significant difference at alpha=0.05.")
    else:
        print("Result: not statistically significant at alpha=0.05.")


def print_statistical_summary(results_df: pd.DataFrame) -> None:
    """Print final statistical summary."""
    print("\n" + "=" * 70)
    print("FINAL STATISTICAL SUMMARY")
    print("=" * 70)

    main_methods = {
        "M1": "Minimal JSON Baseline",
        "M2": "Compact Micro-Rubric",
        "M3": "PORTIA Rank-Interleaved",
        "M4": "Pointwise + Pairwise Hybrid",
        "M7": "Reliability-Weighted Jury",
    }

    print("\nConsistency and Accuracy")
    for prefix, name in main_methods.items():
        consistency, n_pairs = consistency_rate(results_df, prefix)
        acc = accuracy(results_df, prefix)

        print(
            f"{prefix} ({name}): "
            f"Consistency={consistency:.3f} over {n_pairs} pairs | "
            f"Accuracy={acc:.3f}"
        )

    print("\nSycophancy / Flip Rates")
    for prefix, name in {
        "M6": "Twin-Pass Audit",
    }.items():
        rate, flips, total = sycophancy_flip_rate(results_df, prefix)
        print(
            f"{prefix} ({name}): "
            f"Flip Rate={rate:.3f} ({flips}/{total} triggered cases)"
        )

    run_mcnemar_baseline_vs_jury(results_df)


# ============================================================
# Main evaluation loop
# ============================================================

def evaluate_dataset() -> pd.DataFrame:
    print(f"Loading dataset from: {INPUT_DATASET_PATH}")
    df = pd.read_csv(INPUT_DATASET_PATH)

    if len(df) == 0:
        raise RuntimeError("Input dataset is empty.")

    # --- Resume Mechanism: Check which pairs have already been processed ---
    processed_pairs = set()
    if OUTPUT_RESULTS_PATH.exists():
        try:
            existing_df = pd.read_csv(OUTPUT_RESULTS_PATH)
            if "Pair_ID" in existing_df.columns:
                processed_pairs = set(existing_df["Pair_ID"].unique())
                print(f"Found existing results! Resuming after {len(processed_pairs)} processed pairs...")
        except Exception as e:
            print(f"Could not read existing results: {e}. Starting from scratch.")

    for row_idx, row in df.iterrows():
        pair_id = int(row["Pair_ID"]) if "Pair_ID" in row else row_idx + 1

        # Skip this pair if it has already been processed and saved
        if pair_id in processed_pairs:
            continue

        print(
            f"\nEvaluating pair {row_idx + 1}/{len(df)} "
            f"(Pair_ID={pair_id}, Difficulty={row.get('Difficulty', 'NA')})"
        )

        history = row["User_History"]
        original_slate_a = row["Slate_A"]
        original_slate_b = row["Slate_B"]

        orders = [
            {
                "order": "Original",
                "slate_a": original_slate_a,
                "slate_b": original_slate_b,
                "first_label": "A",
                "second_label": "B",
            },
            {
                "order": "Swapped",
                "slate_a": original_slate_b,
                "slate_b": original_slate_a,
                "first_label": "B",
                "second_label": "A",
            },
        ]

        # Store the results for the current pair (both Original and Swapped)
        current_pair_results = []

        for order_cfg in orders:
            order_name = order_cfg["order"]
            slate_a = order_cfg["slate_a"]
            slate_b = order_cfg["slate_b"]
            first_label = order_cfg["first_label"]
            second_label = order_cfg["second_label"]

            print(f"  - Order: {order_name}")

            # M1: Minimal JSON Baseline
            m1 = run_method_1_baseline(history, slate_a, slate_b)
            m1_verdict = map_raw_to_original_label(
                m1["raw_winner"],
                first_label,
                second_label,
            )

            # M2: Compact Micro-Rubric
            m2 = run_method_2_micro_rubric(history, slate_a, slate_b)
            m2_verdict = map_raw_to_original_label(
                m2["raw_winner"],
                first_label,
                second_label,
            )

            # M3: PORTIA Rank-Interleaved
            m3 = run_method_3_portia(history, slate_a, slate_b)
            m3_verdict = map_raw_to_original_label(
                m3["raw_winner"],
                first_label,
                second_label,
            )

            # M4: Pointwise + Pairwise Hybrid
            m4 = run_method_4_pointwise_hybrid(
                history=history,
                slate_a=slate_a,
                slate_b=slate_b,
                baseline_raw_winner=m1["raw_winner"],
            )
            m4_verdict = map_raw_to_original_label(
                m4["raw_winner"],
                first_label,
                second_label,
            )

            # M6: Twin-Pass Audit
            m6 = run_method_6_twin_pass(
                history=history,
                slate_a=slate_a,
                slate_b=slate_b,
                method_2_raw_winner=m2["raw_winner"],
                method_2_parsed=m2["parsed"],
            )
            m6_verdict = map_raw_to_original_label(
                m6["raw_winner"],
                first_label,
                second_label,
            )

            # M7: Reliability-Weighted Jury, currently simple majority over M1-M4.
            m7_raw = majority_vote(
                [
                    m1["raw_winner"],
                    m2["raw_winner"],
                    m3["raw_winner"],
                    m4["raw_winner"],
                ]
            )
            m7_verdict = map_raw_to_original_label(
                m7_raw,
                first_label,
                second_label,
            )

            current_pair_results.append(
                {
                    "Pair_ID": pair_id,
                    "UserID": row["UserID"],
                    "Order": order_name,
                    "Difficulty": row.get("Difficulty", ""),
                    "Utility_A": row["Utility_A"],
                    "Utility_B": row["Utility_B"],
                    "Utility_Gap": row["Utility_Gap"],
                    "Ground_Truth": row["Ground_Truth"],

                    # M1
                    "M1_Raw": m1["raw_winner"],
                    "M1_Verdict": m1_verdict,
                    "M1_Tokens": m1["tokens"],
                    "M1_Response": m1["response"],

                    # M2
                    "M2_Raw": m2["raw_winner"],
                    "M2_Verdict": m2_verdict,
                    "M2_Tokens": m2["tokens"],
                    "M2_Response": m2["response"],
                    "M2_Parsed": json.dumps(m2["parsed"], ensure_ascii=False),

                    # M3
                    "M3_Raw": m3["raw_winner"],
                    "M3_Verdict": m3_verdict,
                    "M3_Tokens": m3["tokens"],
                    "M3_Response": m3["response"],
                    "M3_Parsed": json.dumps(m3["parsed"], ensure_ascii=False),

                    # M4
                    "M4_Raw": m4["raw_winner"],
                    "M4_Verdict": m4_verdict,
                    "M4_Score_A": m4["score_a"],
                    "M4_Score_B": m4["score_b"],
                    "M4_Tokens": m4["tokens"],
                    "M4_Response_A": m4["response_a"],
                    "M4_Response_B": m4["response_b"],  

                    # M6
                    "M6_Triggered": m6["triggered"],
                    "M6_Supports": m6["supports"],
                    "M6_Raw": m6["raw_winner"],
                    "M6_Verdict": m6_verdict,
                    "M6_Tokens": m6["tokens"],
                    "M6_Response": m6["response"],
                    "M6_Flipped": m6["flipped"],

                    # M7
                    "M7_Raw": m7_raw,
                    "M7_Verdict": m7_verdict,
                }
            )

            time.sleep(REQUEST_DELAY_SECONDS)

        # --- Checkpointing: Save to file immediately after finishing the pair ---
        row_df = pd.DataFrame(current_pair_results)
        # Write header only if the file doesn't exist yet
        write_header = not OUTPUT_RESULTS_PATH.exists()
        row_df.to_csv(OUTPUT_RESULTS_PATH, mode='a', header=write_header, index=False)
        print(f"*** Saved Pair_ID {pair_id} to CSV! ***")

    # Read the full dataset at the end to return it for statistical summary
    print(f"\nFinished processing all pairs. Loading full results from: {OUTPUT_RESULTS_PATH}")
    final_results_df = pd.read_csv(OUTPUT_RESULTS_PATH)
    return final_results_df


def main() -> None:
    results_df = evaluate_dataset()
    print_statistical_summary(results_df)


if __name__ == "__main__":
    main()