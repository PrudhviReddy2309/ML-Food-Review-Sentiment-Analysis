
"""
Food Review Sentiment Analysis - 9,000 record educational project
Algorithms: Multinomial Naive Bayes, Logistic Regression, Random Forest,
Linear SVM, and KNN.

Run in Spyder. The CSV is created automatically if it is missing.
Metrics are calculated from a held-out test set; they are never hard-coded.
"""

from pathlib import Path
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, ConfusionMatrixDisplay
)

RANDOM_SEED = 42
DATA_FILE = Path(__file__).with_name("south_indian_restaurant_reviews_9000.csv")

def create_dataset(n_records=9000, seed=RANDOM_SEED):
    """Create the 9-column demo dataset if the CSV is not present."""
    import random
    from datetime import date, timedelta

    rng = random.Random(seed)
    positive = [
        "freshly prepared", "well seasoned", "served hot", "good portion size",
        "friendly staff", "quick service", "clean dining area", "worth the price",
        "soft idlis", "crispy dosa", "fluffy vada", "rich sambar",
        "aromatic biryani", "tender paneer", "balanced chutney", "pleasant experience"
    ]
    negative = [
        "served cold", "poorly seasoned", "long waiting time", "small portion size",
        "rude staff", "slow service", "unclean dining area", "overpriced meal",
        "soggy idlis", "greasy dosa", "hard vada", "watery sambar",
        "bland biryani", "rubbery paneer", "stale chutney", "disappointing experience"
    ]
    neutral = [
        "visited during the evening", "the restaurant was moderately busy",
        "the menu had several choices", "payment was available by card",
        "the table was near the window", "the order included a side dish",
        "the meal was packed in a paper container", "the restaurant is near the main road"
    ]
    openings = [
        "I visited this place recently.", "We ordered dinner here.",
        "My family tried this restaurant.", "I stopped by for lunch.",
        "I ordered takeaway from this outlet.", "This was my second visit.",
        "I tried this place with friends.", "I visited on a weekday."
    ]
    closings = [
        "I may visit again.", "The experience was average overall.",
        "I would recommend checking the menu first.", "The visit had both good and bad points.",
        "I expected a little more.", "It was a memorable meal.", "I will decide after another visit."
    ]
    cuisines = ["South Indian", "Andhra", "Tamil", "Kerala", "Udupi", "Hyderabadi"]
    price_ranges = ["Budget", "Moderate", "Premium"]
    places = [
        "the local cafe", "this family restaurant", "the food court outlet",
        "a small neighbourhood place", "the city branch", "the roadside restaurant"
    ]
    start = date(2024, 1, 1)
    rows = []

    for i in range(1, n_records + 1):
        liked = i % 2
        signal_pool = positive if liked else negative
        opposite_pool = negative if liked else positive
        if liked:
            rating = rng.choices([3, 4, 5], weights=[12, 48, 40])[0]
            service_rating, food_quality = rng.randint(3, 5), rng.randint(3, 5)
        else:
            rating = rng.choices([1, 2, 3], weights=[38, 47, 15])[0]
            service_rating, food_quality = rng.randint(1, 3), rng.randint(1, 3)

        selected = rng.sample(signal_pool, k=rng.randint(2, 4))
        fragments = [rng.choice(openings), f"We visited {rng.choice(places)}."]
        fragments += [
            f"The {rng.choice(['food', 'service', 'place', 'meal'])} was {phrase}."
            for phrase in selected
        ]
        fragments += [f"Also, {phrase}." for phrase in rng.sample(neutral, k=2)]
        if rng.random() < 0.28:
            fragments += [f"One issue was {rng.choice(opposite_pool)}."]
        rng.shuffle(fragments[1:])
        review = " ".join(fragments + [rng.choice(closings)])

        rows.append({
            "Review_ID": f"R{i:05d}",
            "Review": review,
            "Liked": int(liked),
            "Cuisine_Type": rng.choice(cuisines),
            "Review_Date": (start + timedelta(days=rng.randint(0, 1000))).isoformat(),
            "Rating": int(rating),
            "Price_Range": rng.choice(price_ranges),
            "Service_Rating": int(service_rating),
            "Food_Quality_Rating": int(food_quality)
        })

    df = pd.DataFrame(rows)
    # Simulate 12% ambiguous/mislabelled reviews, as can happen in human annotations.
    flip_rng = random.Random(seed + 101)
    flip_indices = flip_rng.sample(range(len(df)), k=int(0.12 * len(df)))
    df.loc[flip_indices, "Liked"] = 1 - df.loc[flip_indices, "Liked"]
    return df

def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    if not DATA_FILE.exists():
        create_dataset().to_csv(DATA_FILE, index=False)
        print("Dataset created:", DATA_FILE)

    df = pd.read_csv(DATA_FILE)
    required = ["Review", "Liked"]
    if not all(c in df.columns for c in required):
        raise ValueError("Dataset must contain Review and Liked columns.")

    print("\\nDATASET INFORMATION")
    print("Records:", len(df))
    print("Columns:", len(df.columns))
    print("Column names:", ", ".join(df.columns))
    print("\\nTarget distribution:")
    print(df["Liked"].value_counts().sort_index().rename(index={0: "Negative", 1: "Positive"}))

    X_train, X_test, y_train, y_test = train_test_split(
        df["Review"].astype(str), df["Liked"].astype(int),
        test_size=0.20, random_state=RANDOM_SEED, stratify=df["Liked"]
    )

    # Shared text representation keeps comparison consistent across algorithms.
    vectorizer = TfidfVectorizer(
        lowercase=True, stop_words="english", ngram_range=(1, 2),
        min_df=2, max_features=12000, sublinear_tf=True
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    models = {
        "Multinomial Naive Bayes": MultinomialNB(alpha=1.5),
        "Logistic Regression": LogisticRegression(C=0.7, max_iter=1500, random_state=RANDOM_SEED),
        "Random Forest": RandomForestClassifier(
            n_estimators=180, max_depth=18, min_samples_leaf=3,
            random_state=RANDOM_SEED, n_jobs=-1
        ),
        "Linear SVM": LinearSVC(C=0.5, random_state=RANDOM_SEED),
        "KNN": KNeighborsClassifier(n_neighbors=7)
    }
    # KNN uses a fixed, documented neighbour count.
    models["KNN"] = KNeighborsClassifier(n_neighbors=7)

    results = []
    fitted = {}
    for name, model in models.items():
        model.fit(X_train_vec, y_train)
        pred = model.predict(X_test_vec)
        row = {
            "Algorithm": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1_Score": f1_score(y_test, pred, zero_division=0)
        }
        results.append(row)
        fitted[name] = model
        print(f"\\n{name}")
        print(classification_report(y_test, pred, target_names=["Negative", "Positive"], zero_division=0))

    result_df = pd.DataFrame(results).sort_values("Accuracy", ascending=False)
    result_df.to_csv("algorithm_comparison_metrics.csv", index=False)
    print("\\nMODEL COMPARISON (decimal values from held-out test data)")
    print(result_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Confusion matrix for the selected model (Multinomial Naive Bayes).
    selected_name = "Multinomial Naive Bayes"
    selected_pred = fitted[selected_name].predict(X_test_vec)
    cm = confusion_matrix(y_test, selected_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    print("\\nMULTINOMIAL NAIVE BAYES CONFUSION MATRIX")
    print(cm)
    print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["Negative", "Positive"]
    ).plot(cmap="Blues", values_format="d")
    plt.title("Multinomial Naive Bayes - Confusion Matrix")
    plt.tight_layout()
    plt.savefig("multinomial_nb_confusion_matrix.png", dpi=160)
    plt.show()

    result_plot = result_df.set_index("Algorithm")[["Accuracy", "Precision", "Recall", "F1_Score"]]
    ax = result_plot.plot(kind="bar", figsize=(12, 6), ylim=(0, 1), rot=18)
    ax.set_ylabel("Score (0 to 1)")
    ax.set_title("Food Review Sentiment - Algorithm Comparison")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig("algorithm_comparison.png", dpi=160)
    plt.show()

    # Live interactive prediction using the selected Multinomial NB model.
    print("\\nLIVE REVIEW SENTIMENT PREDICTION")
    print("Type a review and press Enter. Type 'exit' to stop.")
    while True:
        review = input("\\nEnter review: ").strip()
        if review.lower() == "exit":
            print("Prediction session ended.")
            break
        if not review:
            print("Please enter a review.")
            continue
        vector = vectorizer.transform([review])
        prediction = fitted[selected_name].predict(vector)[0]
        probabilities = fitted[selected_name].predict_proba(vector)[0]
        label = "Positive" if prediction == 1 else "Negative"
        print("Sentiment:", label)
        print(f"Negative probability: {probabilities[0]:.4f}")
        print(f"Positive probability: {probabilities[1]:.4f}")

if __name__ == "__main__":
    main()
