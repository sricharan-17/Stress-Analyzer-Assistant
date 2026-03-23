from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import joblib
import shap
import os
import requests
from dotenv import load_dotenv
load_dotenv()
from groq import Groq

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# -----------------------------
# LLM helper
# -----------------------------

def get_llm_recommendations(stress_level, top_factors):
    print("INSIDE LLM FUNCTION")
    prompt = f"""
You are a practical stress assistant for college students.

Context:
- Stress level: {stress_level}
- Top contributing factors: {', '.join(top_factors)}

Task:
1. Start with ONE short sentence listing the main stress factors.
2. Then give 1–2 practical actions for EACH factor.

STRICT RULES:
- No medical advice
- No therapy suggestions
- No helplines or crisis language
- Do not assume extreme situations
- Keep tone simple, calm, and realistic
- Advice must be doable for a college student (limited time, limited money)
- No long paragraphs
- No motivational speeches
- No conclusion at the end

Focus on:
- managing study workload
- improving daily routine
- sleep habits
- basic budgeting (for financial stress)
- small behavior changes

Format:

Based on the analysis, your main stress factors are: ...

For [Factor]:
- ...
- ...
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content

# -----------------------------
# App setup
# -----------------------------
app = Flask(__name__)

MODEL_PATH = "model/rf_stress_model_balanced.pkl"

if not os.path.exists(MODEL_PATH):
    print("Downloading model from Hugging Face...")

    url = "https://huggingface.co/sricharan007/Stress-Analyzer-Assistant-model/resolve/main/rf_stress_model_balanced.pkl"

    os.makedirs("model", exist_ok=True)

    response = requests.get(url)
    with open(MODEL_PATH, "wb") as f:
        f.write(response.content)

    print("Model downloaded successfully.")

# Load model
model = joblib.load(MODEL_PATH)
ENCODER_PATH = "model/label_encoders.pkl"

if not os.path.exists(ENCODER_PATH):
    print("Downloading encoders...")

    encoder_url = "https://huggingface.co/sricharan007/Stress-Analyzer-Assistant-model/resolve/main/label_encoders.pkl"

    os.makedirs("model", exist_ok=True)

    r = requests.get(encoder_url)
    with open(ENCODER_PATH, "wb") as f:
        f.write(r.content)

    print("Encoders downloaded.")

label_encoders = joblib.load(ENCODER_PATH)

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
            if contributors:
                print("CALLING GROQ...")
                llm_text = get_llm_recommendations(prediction, contributors)
            else:
                llm_text = "No contributors found."

        except Exception as e:
            import traceback
            print("========== LLM ERROR ==========")
            traceback.print_exc()
            print("================================")
            llm_text = "LLM failed"

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
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
