import os
import random
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd


# ============================================================
# Configuration
# ============================================================

DATA_DIR = Path("data")
ML_ZIP_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
ZIP_PATH = DATA_DIR / "ml-1m.zip"
EXTRACT_DIR = DATA_DIR / "ml-1m"
OUTPUT_PATH = DATA_DIR / "evaluation_dataset.csv"

RANDOM_SEED = 42
TARGET_NUM_PAIRS = 100
HISTORY_SIZE = 5
SLATE_SIZE = 3
MIN_USER_RATINGS = 30


# ============================================================
# Data loading
# ============================================================

def download_and_extract_movielens() -> None:
    """Download and extract MovieLens-1M if it does not already exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if EXTRACT_DIR.exists():
        print("MovieLens-1M already exists. Skipping download.")
        return

    print("Downloading MovieLens-1M dataset...")
    urllib.request.urlretrieve(ML_ZIP_URL, ZIP_PATH)

    print("Extracting MovieLens-1M...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(DATA_DIR)

    print("Extraction complete.")


def load_movielens() -> pd.DataFrame:
    """Load MovieLens movies and ratings, then merge them into one DataFrame."""
    print("Loading MovieLens data...")

    movies_cols = ["MovieID", "Title", "Genres"]
    ratings_cols = ["UserID", "MovieID", "Rating", "Timestamp"]

    movies = pd.read_csv(
        EXTRACT_DIR / "movies.dat",
        sep="::",
        engine="python",
        names=movies_cols,
        encoding="latin-1",
    )

    ratings = pd.read_csv(
        EXTRACT_DIR / "ratings.dat",
        sep="::",
        engine="python",
        names=ratings_cols,
    )

    df = ratings.merge(movies, on="MovieID")

    # Feature enrichment: Title | Genres
    df["Movie_Feature_Text"] = df["Title"] + " | " + df["Genres"]

    return df


# ============================================================
# Dataset generation
# ============================================================

def difficulty_from_gap(gap: int) -> str:
    """Convert absolute utility gap into difficulty bucket."""
    if gap >= 4:
        return "Easy"
    if gap in (2, 3):
        return "Medium"
    if gap == 1:
        return "Hard"

    raise ValueError(f"Invalid gap for difficulty: {gap}")


def format_history(items: pd.DataFrame) -> str:
    """Format user history as bullet list."""
    return "\n".join(f"- {x}" for x in items["Movie_Feature_Text"].tolist())


def format_slate(items: pd.DataFrame) -> str:
    """Format slate as ranked list."""
    return "\n".join(
        f"{i + 1}. {x}"
        for i, x in enumerate(items["Movie_Feature_Text"].tolist())
    )


def build_single_example(
    user_data: pd.DataFrame,
    rng: random.Random,
) -> dict | None:
    """
    Build one pairwise slate-comparison example for a single user.

    Returns None if the user does not have enough usable data or if the
    generated pair has equal utility.
    """
    positive_movies = user_data[user_data["Rating"] >= 4]

    if len(positive_movies) < HISTORY_SIZE:
        return None

    history = positive_movies.sample(
        HISTORY_SIZE,
        random_state=rng.randint(0, 10**9),
    )

    remaining_movies = user_data[~user_data["MovieID"].isin(history["MovieID"])]

    if len(remaining_movies) < SLATE_SIZE * 2:
        return None

    candidates = remaining_movies.sample(
        SLATE_SIZE * 2,
        random_state=rng.randint(0, 10**9),
    )

    slate_a = candidates.iloc[:SLATE_SIZE]
    slate_b = candidates.iloc[SLATE_SIZE:]

    utility_a = int(slate_a["Rating"].sum())
    utility_b = int(slate_b["Rating"].sum())
    gap = abs(utility_a - utility_b)

    # Strictly filter out ties.
    if gap == 0:
        return None

    ground_truth = "A" if utility_a > utility_b else "B"
    difficulty = difficulty_from_gap(gap)

    return {
        "UserID": int(user_data["UserID"].iloc[0]),
        "User_History": format_history(history),
        "Slate_A": format_slate(slate_a),
        "Utility_A": utility_a,
        "Slate_B": format_slate(slate_b),
        "Utility_B": utility_b,
        "Utility_Gap": gap,
        "Difficulty": difficulty,
        "Ground_Truth": ground_truth,
    }


def generate_evaluation_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Generate exactly TARGET_NUM_PAIRS examples if enough valid users exist."""
    print(f"Generating {TARGET_NUM_PAIRS} evaluation pairs...")

    rng = random.Random(RANDOM_SEED)

    user_counts = df["UserID"].value_counts()
    valid_users = user_counts[user_counts >= MIN_USER_RATINGS].index.tolist()

    rng.shuffle(valid_users)

    rows = []

    # First pass: one example per user.
    for uid in valid_users:
        if len(rows) >= TARGET_NUM_PAIRS:
            break

        user_data = df[df["UserID"] == uid]
        example = build_single_example(user_data, rng)

        if example is not None:
            rows.append(example)

    if len(rows) < TARGET_NUM_PAIRS:
        raise RuntimeError(
            f"Could only generate {len(rows)} valid pairs. "
            f"Try lowering MIN_USER_RATINGS or increasing candidate attempts."
        )

    final_df = pd.DataFrame(rows).head(TARGET_NUM_PAIRS)

    # Stable pair ID for downstream evaluation.
    final_df.insert(0, "Pair_ID", range(1, len(final_df) + 1))

    return final_df


def main() -> None:
    download_and_extract_movielens()
    df = load_movielens()

    final_df = generate_evaluation_dataset(df)
    final_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved dataset to: {OUTPUT_PATH}")
    print(f"Total pairs: {len(final_df)}")
    print("\nDifficulty distribution:")
    print(final_df["Difficulty"].value_counts().to_string())
    print("\nGround-truth distribution:")
    print(final_df["Ground_Truth"].value_counts().to_string())


if __name__ == "__main__":
    main()