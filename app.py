# =========================
# Required Packages
# =========================
# pip install scikit-learn==1.7.2
# pip install numpy
# pip install flask

from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# =========================
# Absolute Path for Models
# =========================
base_path = r"C:\Users\91970\PycharmProjects\Education recommendation system\Models"

# =========================
# Load Scaler and Model
# =========================
scaler = None
model = None

try:
    if os.path.exists(base_path):
        print("Looking for files in:", base_path)
        print("Files found:", os.listdir(base_path))

        # Load Scaler
        for file in os.listdir(base_path):
            if "scaler" in file.lower() and file.endswith(".pkl"):
                scaler_path = os.path.join(base_path, file)
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
                print(f"✅ Loaded Scaler: {file}")
                break

        # Load Model
        for file in os.listdir(base_path):
            if "model" in file.lower() and file.endswith(".pkl"):
                model_path = os.path.join(base_path, file)
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                print(f"✅ Loaded Model: {file}")
                break

    else:
        print("❌ Models folder does not exist:", base_path)

except Exception as e:
    print("❌ Error while loading files:", e)

# =========================
# Class Names
# =========================
class_names = [
    'Lawyer', 'Doctor', 'Government Officer', 'Artist', 'Unknown',
    'Software Engineer', 'Teacher', 'Business Owner', 'Scientist',
    'Banker', 'Writer', 'Accountant', 'Designer',
    'Construction Engineer', 'Game Developer', 'Stock Investor',
    'Real Estate Developer'
]

# =========================
# Recommendation Function
# =========================
def Recommendations(gender, part_time_job, absence_days, extracurricular_activities,
                    weekly_self_study_hours, math_score, history_score, physics_score,
                    chemistry_score, biology_score, english_score, geography_score,
                    total_score, average_score):
    if scaler is None or model is None:
        return [("Error: Model or Scaler not loaded", 0.0)]

    try:
        # Encode categorical variables
        gender_encoded = 1 if gender.lower() == 'female' else 0
        part_time_job_encoded = 1 if part_time_job else 0
        extracurricular_activities_encoded = 1 if extracurricular_activities else 0

        # Create feature array
        feature_array = np.array([[
            gender_encoded, part_time_job_encoded, absence_days,
            extracurricular_activities_encoded, weekly_self_study_hours,
            math_score, history_score, physics_score, chemistry_score,
            biology_score, english_score, geography_score,
            total_score, average_score
        ]], dtype=float)

        # Scale features
        scaled_features = scaler.transform(feature_array)

        # Predict probabilities
        probabilities = model.predict_proba(scaled_features)

        # Top 3 classes with probabilities
        top_classes_idx = np.argsort(-probabilities[0])[:3]
        top_classes_names_probs = [
            (class_names[idx], round(float(probabilities[0][idx]) * 100, 2))
            for idx in top_classes_idx
        ]

        return top_classes_names_probs

    except Exception as e:
        print("❌ Error during prediction:", e)
        return [("Error during prediction", 0.0)]

# =========================
# Routes
# =========================
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/recommend')
def recommend():
    return render_template('recommend.html')

@app.route('/pred', methods=['POST'])
def pred():
    try:
        # Get input values safely
        gender = request.form.get('gender', 'male')
        part_time_job = request.form.get('part_time_job', 'false') == 'true'
        absence_days = int(request.form.get('absence_days', 0))
        extracurricular_activities = request.form.get('extracurricular_activities', 'false') == 'true'
        weekly_self_study_hours = int(request.form.get('weekly_self_study_hours', 0))
        math_score = int(request.form.get('math_score', 0))
        history_score = int(request.form.get('history_score', 0))
        physics_score = int(request.form.get('physics_score', 0))
        chemistry_score = int(request.form.get('chemistry_score', 0))
        biology_score = int(request.form.get('biology_score', 0))
        english_score = int(request.form.get('english_score', 0))
        geography_score = int(request.form.get('geography_score', 0))
        total_score = float(request.form.get('total_score', 0))
        average_score = float(request.form.get('average_score', 0))

        # Get recommendations
        recommendations = Recommendations(
            gender, part_time_job, absence_days, extracurricular_activities,
            weekly_self_study_hours, math_score, history_score, physics_score,
            chemistry_score, biology_score, english_score, geography_score,
            total_score, average_score
        )

        # Render result page
        return render_template('results.html', recommendations=recommendations)

    except Exception as e:
        return f"❌ Error in processing input: {e}"

# =========================
# Run App
# =========================
if __name__ == '__main__':
    app.run(debug=True)
