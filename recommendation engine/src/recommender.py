import os
import joblib

from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

vectorizer = joblib.load(
    os.path.join(
        MODEL_DIR,
        "tfidf_vectorizer.pkl"
    )
)

startup_matrix = joblib.load(
    os.path.join(
        MODEL_DIR,
        "startup_tfidf_matrix.pkl"
    )
)

startup_data = joblib.load(
    os.path.join(
        MODEL_DIR,
        "startup_data.pkl"
    )
)


# ============================================================
# RECOMMENDATION FUNCTION
# ============================================================

def recommend_startups(
    skills="",
    interests="",
    industry="",
    startup_stage="",
    location="",
    top_n=5
):

    user_profile = " ".join([
        skills,
        interests,
        industry,
        startup_stage,
        location
    ])

    # Convert user profile into TF-IDF vector
    user_vector = vectorizer.transform(
        [user_profile]
    )

    # Calculate similarity against all startups
    scores = cosine_similarity(
        user_vector,
        startup_matrix
    ).flatten()

    # Highest score first
    ranked_indices = scores.argsort()[::-1]

    recommendations = []

    for index in ranked_indices[:top_n]:

        startup = startup_data.iloc[index]

        recommendations.append({
            "startup_id": startup["startup_id"],
            "startup_name": startup["startup_name"],
            "industry": startup["industry_category"],
            "sub_industry": startup["sub_industry"],
            "location": startup["location"],
            "funding_stage": startup["funding_stage"],
            "required_skills": startup["required_skills"],
            "recommended_roles": startup["recommended_roles"],
            "match_score": round(
                float(scores[index]) * 100,
                2
            )
        })

    return recommendations


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    results = recommend_startups(
        skills="Python Machine Learning SQL",
        interests="Artificial Intelligence Healthcare",
        industry="Healthcare AI",
        startup_stage="Seed",
        location="Mumbai",
        top_n=5
    )

    print("\n")
    print("=" * 60)
    print("STARTUPSYNC RECOMMENDATIONS")
    print("=" * 60)

    for position, result in enumerate(
        results,
        start=1
    ):

        print(
            f"\n{position}. "
            f"{result['startup_name']}"
        )

        print(
            f"   Match Score: "
            f"{result['match_score']}%"
        )

        print(
            f"   Industry: "
            f"{result['industry']}"
        )

        print(
            f"   Location: "
            f"{result['location']}"
        )

        print(
            f"   Funding Stage: "
            f"{result['funding_stage']}"
        )
