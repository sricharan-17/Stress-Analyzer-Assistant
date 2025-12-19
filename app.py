from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import joblib
import shap
import requests

# -----------------------------
# LLM helper
# -----------------------------
def get_llm_recommendations(stress_level, top_factors):
    prompt = f"""
You are a wellness assistant designed to support college students.

The system has identified:
- Predicted stress level: {stress_level}
- Top contributing factors: {', '.join(top_factors)}

Your task:
1. Briefly restate the contributing factors in one sentence.
2. For EACH factor, give 1–2 practical, non-medical suggestions.

Rules:
- Address each factor individually.
- Student-focused, realistic advice only.
- No medical or therapy language.
- No conclusions or meta commentary.
- Short, actionable points.

Format:

"Based on the analysis, the main factors contributing to your stress are: ...

For [Factor]:
- ...
- ...
"
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.1:8b",
            "prompt": prompt,
            "stream": False
        },
        timeout=60
    )

    return response.json()["response"].strip()


# -----------------------------
# App setup
# -----------------------------
app = Flask(__name__)

model = joblib.load("model/rf_stress_model_balanced.pkl")
label_encoders = joblib.load("model/label_encoders.pkl")

categorical_cols = [
    "Gender",
    "City",
    "Profession",
    "Sleep Duration",
    "Dietary Habits",
    "Degree",
    "Have you ever had suicidal thoughts ?",
    "Family History of Mental Illness"
]

# Store last result (single-user local app)
last_result = {
    "prediction": None,
    "contributors": [],
    "llm_text": None
}


# -----------------------------
# Routes
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        data = {
            "Gender": request.form["gender"],
            "Age": int(request.form["age"]),
            "City": request.form["city"],
            "Profession": "Student",
            "Academic Pressure": int(request.form["academic_pressure"]),
            "Work Pressure": 0,
            "CGPA": float(request.form["cgpa"]),
            "Study Satisfaction": int(request.form["study_satisfaction"]),
            "Job Satisfaction": 0,
            "Sleep Duration": request.form["sleep_duration"],
            "Dietary Habits": request.form["dietary_habits"],
            "Degree": request.form["degree"],
            "Have you ever had suicidal thoughts ?": request.form["suicidal"],
            "Work/Study Hours": int(request.form["work_hours"]),
            "Financial Stress": int(request.form["financial_stress"]),
            "Family History of Mental Illness": request.form["family_history"],
            "Depression": int(request.form["depression"])
        }

        df = pd.DataFrame([data])

        # Encode categoricals exactly like training
        for col in categorical_cols:
            df[col] = label_encoders[col].transform(df[col])

        # -----------------------------
        # Prediction
        # -----------------------------
        prediction = model.predict(df)[0]  # 'Low', 'Medium', 'High'

        # -----------------------------
        # SHAP explanation
        # -----------------------------
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(df)

        class_index = list(model.classes_).index(prediction)

        if isinstance(shap_values, list):
            shap_vals_for_class = shap_values[class_index][0]
        else:
            shap_vals_for_class = shap_values[0, :, class_index]

        raw_contribs = (
            pd.Series(shap_vals_for_class, index=df.columns)
            .abs()
            .sort_values(ascending=False)
        )

        # -----------------------------
        # Contributor filtering (IMPORTANT)
        # -----------------------------
        if prediction == "High":
            contributors = raw_contribs.head(5).index.tolist()

        elif prediction == "Medium":
            contributors = [
                c for c in raw_contribs.index
                if c not in [
                    "Have you ever had suicidal thoughts ?",
                    "Depression"
                ]
            ][:5]

        else:  # Low stress
            contributors = []

        # -----------------------------
        # LLM logic
        # -----------------------------
        try:
            if prediction in ["Medium", "High"] and contributors:
                llm_text = get_llm_recommendations(prediction, contributors)
            else:
                llm_text = (
                    "Your stress level appears to be low. "
                    "This suggests that you are managing your academic workload, sleep, "
                    "and daily routines well. Maintaining these habits can help sustain "
                    "your overall well-being."
                )
        except Exception:
            llm_text = last_result["llm_text"]

        # Store result
        last_result["prediction"] = prediction
        last_result["contributors"] = contributors
        last_result["llm_text"] = llm_text

        return redirect(url_for("index"))

    # GET request
    return render_template(
        "index.html",
        prediction=last_result["prediction"],
        contributors=last_result["contributors"],
        llm_text=last_result["llm_text"]
    )


# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
