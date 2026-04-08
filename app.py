from flask import Flask, request, jsonify, render_template_string
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# ==============================
# Load Model
# ==============================
try:
    with open("heart_model.pkl", "rb") as f:
        model = pickle.load(f)
    print("Model loaded successfully!")
except FileNotFoundError:
    print("WARNING: heart_model.pkl not found. Train and save the model first.")
    model = None

# ==============================
# Scaler (must match training scaler — ideally save/load it too)
# ==============================
# NOTE: For production, save and load the scaler with pickle just like the model.
# Here we instantiate a fresh one as a placeholder.
scaler = StandardScaler()

# ==============================
# HTML Front-End (single-file UI)
# ==============================
HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
    <title>Heart Disease Prediction</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 700px; margin: 40px auto; padding: 0 20px; background: #f5f5f5; }
        h1   { color: #c0392b; text-align: center; }
        form { background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,.1); }
        label { display: block; margin-top: 12px; font-weight: bold; color: #333; }
        input[type=number] { width: 100%; padding: 8px; margin-top: 4px; border: 1px solid #ccc; border-radius: 4px; box-sizing: border-box; }
        button { margin-top: 20px; width: 100%; padding: 12px; background: #c0392b; color: white; border: none; border-radius: 6px; font-size: 16px; cursor: pointer; }
        button:hover { background: #a93226; }
        #result { margin-top: 20px; padding: 15px; border-radius: 6px; text-align: center; font-size: 18px; display: none; }
        .positive { background: #fdecea; color: #c0392b; border: 1px solid #c0392b; }
        .negative { background: #eafaf1; color: #1e8449; border: 1px solid #1e8449; }
    </style>
</head>
<body>
    <h1>❤️ Heart Disease Prediction</h1>
    <form id="predForm">
        <label>Age</label>
        <input type="number" name="age" placeholder="e.g. 52" required />

        <label>Sex (1 = Male, 0 = Female)</label>
        <input type="number" name="sex" min="0" max="1" placeholder="0 or 1" required />

        <label>Chest Pain Type (0-3)</label>
        <input type="number" name="cp" min="0" max="3" placeholder="0–3" required />

        <label>Resting Blood Pressure</label>
        <input type="number" name="trestbps" placeholder="e.g. 125" required />

        <label>Serum Cholesterol (mg/dl)</label>
        <input type="number" name="chol" placeholder="e.g. 212" required />

        <label>Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)</label>
        <input type="number" name="fbs" min="0" max="1" placeholder="0 or 1" required />

        <label>Resting ECG Results (0-2)</label>
        <input type="number" name="restecg" min="0" max="2" placeholder="0–2" required />

        <label>Max Heart Rate Achieved</label>
        <input type="number" name="thalach" placeholder="e.g. 168" required />

        <label>Exercise Induced Angina (1 = Yes, 0 = No)</label>
        <input type="number" name="exang" min="0" max="1" placeholder="0 or 1" required />

        <label>ST Depression (oldpeak)</label>
        <input type="number" name="oldpeak" step="0.1" placeholder="e.g. 1.0" required />

        <label>Slope of Peak Exercise ST Segment (0-2)</label>
        <input type="number" name="slope" min="0" max="2" placeholder="0–2" required />

        <label>Number of Major Vessels (0-3)</label>
        <input type="number" name="ca" min="0" max="3" placeholder="0–3" required />

        <label>Thalassemia (0 = Normal, 1 = Fixed Defect, 2 = Reversible Defect)</label>
        <input type="number" name="thal" min="0" max="2" placeholder="0–2" required />

        <button type="submit">Predict</button>
    </form>

    <div id="result"></div>

    <script>
        const LABELS = {
            0: "No Heart Disease",
            1: "Heart Disease (Stage 1)",
            2: "Heart Disease (Stage 2)"
        };

        document.getElementById("predForm").addEventListener("submit", async function(e) {
            e.preventDefault();
            const formData = new FormData(e.target);
            const data = {};
            formData.forEach((v, k) => data[k] = parseFloat(v));

            const res = await fetch("/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(data)
            });
            const json = await res.json();

            const div = document.getElementById("result");
            div.style.display = "block";
            if (json.error) {
                div.className = "";
                div.innerHTML = "⚠️ " + json.error;
            } else {
                const label = LABELS[json.prediction] ?? `Class ${json.prediction}`;
                const isPositive = json.prediction !== 0;
                div.className = isPositive ? "positive" : "negative";
                div.innerHTML = `<strong>Result:</strong> ${label}<br/>
                                 <small>Confidence: ${(json.confidence * 100).toFixed(1)}%</small>`;
            }
        });
    </script>
</body>
</html>
"""

# ==============================
# Routes
# ==============================

@app.route("/")
def index():
    return render_template_string(HTML_PAGE)


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Please train and save heart_model.pkl first."}), 500

    data = request.get_json()
    if not data:
        return jsonify({"error": "No input data provided."}), 400

    # Expected feature order (must match training data column order)
    feature_order = [
        "age", "sex", "cp", "trestbps", "chol", "fbs",
        "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
    ]

    try:
        features = [float(data[feat]) for feat in feature_order]
    except KeyError as e:
        return jsonify({"error": f"Missing feature: {e}"}), 400
    except ValueError as e:
        return jsonify({"error": f"Invalid value: {e}"}), 400

    features_array = np.array(features).reshape(1, -1)

    # Scale input
    # NOTE: In production, load and use the saved scaler (not a freshly fitted one).
    features_scaled = scaler.fit_transform(features_array)  # placeholder

    prediction = int(model.predict(features_scaled)[0])

    # Confidence (probability of predicted class)
    try:
        proba = model.predict_proba(features_scaled)[0]
        confidence = float(proba[prediction])
    except AttributeError:
        confidence = 1.0  # SVM without probability=True

    return jsonify({
        "prediction": prediction,
        "confidence": confidence,
        "message": "Heart disease detected." if prediction != 0 else "No heart disease detected."
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})


# ==============================
# Run
# ==============================
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
