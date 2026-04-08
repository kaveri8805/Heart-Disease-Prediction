import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize

# =============================
# Page Config
# =============================
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================
# Custom CSS
# =============================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        color: #e63946;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6c757d;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #fff5f5, #ffe0e0);
        border-left: 4px solid #e63946;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
    }
    .prediction-box-high {
        background: #fff0f0;
        border: 2px solid #e63946;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    .prediction-box-low {
        background: #f0fff4;
        border: 2px solid #2a9d8f;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
    }
    .sidebar .sidebar-content {
        background: #1a1a2e;
    }
</style>
""", unsafe_allow_html=True)

# =============================
# Load or Train Model
# =============================
@st.cache_resource
def load_or_train_model():
    """Try to load saved model, otherwise train a new one."""
    # Try loading saved model
    try:
        with open("heart_model.pkl", "rb") as f:
            model = pickle.load(f)
        scaler = None  # Will retrain scaler from data
    except FileNotFoundError:
        model = None
        scaler = None

    # Load dataset and train scaler (always needed)
    try:
        df = pd.read_csv("heart.csv")
    except FileNotFoundError:
        # Generate synthetic data matching the notebook's structure
        np.random.seed(42)
        n = 303
        df = pd.DataFrame({
            'age': np.random.randint(29, 77, n),
            'sex': np.random.randint(0, 2, n),
            'cp': np.random.randint(0, 4, n),
            'trestbps': np.random.randint(94, 200, n),
            'chol': np.random.randint(126, 564, n),
            'fbs': np.random.randint(0, 2, n),
            'restecg': np.random.randint(0, 3, n),
            'thalach': np.random.randint(71, 202, n),
            'exang': np.random.randint(0, 2, n),
            'oldpeak': np.round(np.random.uniform(0, 6.2, n), 1),
            'slope': np.random.randint(0, 3, n),
            'ca': np.random.randint(0, 5, n),
            'thal': np.random.randint(0, 4, n),
            'prediction': np.random.randint(0, 3, n)
        })

    X = df.drop("prediction", axis=1)
    y = df["prediction"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if model is None:
        model = RandomForestClassifier(max_depth=4, min_samples_split=2, n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)

    return model, scaler, df, X_test_scaled, y_test, X.columns.tolist()

model, scaler, df, X_test_scaled, y_test, feature_names = load_or_train_model()

# =============================
# Sidebar — Input Form
# =============================
st.sidebar.markdown("## 🩺 Patient Input")
st.sidebar.markdown("Enter patient clinical measurements:")

age = st.sidebar.slider("Age", 20, 80, 55)
sex = st.sidebar.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female (0)" if x == 0 else "Male (1)")
cp = st.sidebar.selectbox("Chest Pain Type (cp)", options=[0, 1, 2, 3],
    format_func=lambda x: ["Typical Angina (0)", "Atypical Angina (1)", "Non-Anginal (2)", "Asymptomatic (3)"][x])
trestbps = st.sidebar.slider("Resting Blood Pressure (mmHg)", 80, 200, 120)
chol = st.sidebar.slider("Serum Cholesterol (mg/dl)", 100, 600, 240)
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1],
    format_func=lambda x: "No (0)" if x == 0 else "Yes (1)")
restecg = st.sidebar.selectbox("Resting ECG Results", options=[0, 1, 2],
    format_func=lambda x: ["Normal (0)", "ST-T Abnormality (1)", "LV Hypertrophy (2)"][x])
thalach = st.sidebar.slider("Max Heart Rate Achieved", 70, 210, 150)
exang = st.sidebar.selectbox("Exercise Induced Angina", options=[0, 1],
    format_func=lambda x: "No (0)" if x == 0 else "Yes (1)")
oldpeak = st.sidebar.slider("ST Depression (Oldpeak)", 0.0, 7.0, 1.0, step=0.1)
slope = st.sidebar.selectbox("Slope of Peak ST Segment", options=[0, 1, 2],
    format_func=lambda x: ["Upsloping (0)", "Flat (1)", "Downsloping (2)"][x])
ca = st.sidebar.slider("Number of Major Vessels (ca)", 0, 4, 0)
thal = st.sidebar.selectbox("Thal", options=[0, 1, 2, 3],
    format_func=lambda x: ["Normal (0)", "Fixed Defect (1)", "Reversable Defect (2)", "Other (3)"][x])

predict_btn = st.sidebar.button("🔍 Predict", use_container_width=True, type="primary")

# =============================
# Main Content
# =============================
st.markdown('<h1 class="main-header">❤️ Heart Disease Prediction</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">ML-powered clinical decision support tool using Random Forest</p>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Model Performance", "📈 Data Insights"])

# ---- TAB 1: Prediction ----
with tab1:
    if predict_btn:
        input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                                 thalach, exang, oldpeak, slope, ca, thal]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        proba = model.predict_proba(input_scaled)[0]

        label_map = {0: "No Disease", 1: "Mild Disease", 2: "Severe Disease"}
        color_map = {0: "prediction-box-low", 1: "prediction-box-high", 2: "prediction-box-high"}
        emoji_map = {0: "✅", 1: "⚠️", 2: "🚨"}

        pred_label = label_map[prediction]
        pred_emoji = emoji_map[prediction]
        pred_class = color_map[prediction]

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(f"""
            <div class="{pred_class}">
                <h2 style="margin:0">{pred_emoji} {pred_label}</h2>
                <p style="font-size:1.1rem; margin-top:0.5rem; color:#555">
                    Predicted Class: <strong>{prediction}</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### 📊 Prediction Confidence")
        conf_df = pd.DataFrame({
            "Class": [f"Class {i} — {label_map[i]}" for i in range(3)],
            "Probability": [round(p * 100, 1) for p in proba]
        })
        fig, ax = plt.subplots(figsize=(7, 3))
        colors = ["#2a9d8f" if i != prediction else "#e63946" for i in range(3)]
        bars = ax.barh(conf_df["Class"], conf_df["Probability"], color=colors)
        ax.set_xlabel("Probability (%)")
        ax.set_xlim(0, 100)
        ax.set_title("Class Probability Distribution")
        for bar, val in zip(bars, conf_df["Probability"]):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                    f"{val}%", va='center', fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig)
        plt.close()

        st.markdown("### 🧾 Input Summary")
        input_summary = pd.DataFrame({
            "Feature": feature_names,
            "Value": [age, sex, cp, trestbps, chol, fbs, restecg,
                      thalach, exang, oldpeak, slope, ca, thal]
        })
        st.dataframe(input_summary.set_index("Feature").T, use_container_width=True)

    else:
        st.info("👈 Fill in patient details in the sidebar and click **Predict** to see results.")
        st.markdown("### 📋 Feature Reference")
        st.markdown("""
        | Feature | Description |
        |---------|-------------|
        | **age** | Age in years |
        | **sex** | 0 = Female, 1 = Male |
        | **cp** | Chest pain type (0–3) |
        | **trestbps** | Resting blood pressure (mmHg) |
        | **chol** | Serum cholesterol (mg/dl) |
        | **fbs** | Fasting blood sugar > 120 mg/dl (1 = Yes) |
        | **restecg** | Resting ECG results (0–2) |
        | **thalach** | Maximum heart rate achieved |
        | **exang** | Exercise-induced angina (1 = Yes) |
        | **oldpeak** | ST depression induced by exercise |
        | **slope** | Slope of peak exercise ST segment |
        | **ca** | Number of major vessels (0–4) |
        | **thal** | Thal type (0–3) |
        | **prediction** | Target: 0 = No disease, 1 = Mild, 2 = Severe |
        """)

# ---- TAB 2: Model Performance ----
with tab2:
    col1, col2, col3 = st.columns(3)

    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)

    from sklearn.metrics import accuracy_score
    acc = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_prob, multi_class='ovr')

    col1.metric("✅ Accuracy", f"{acc*100:.1f}%")
    col2.metric("📐 ROC-AUC", f"{auc_score:.4f}")
    col3.metric("🌲 Model", "Random Forest")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', ax=ax,
                    xticklabels=["No Disease", "Mild", "Severe"],
                    yticklabels=["No Disease", "Mild", "Severe"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)
        plt.close()

    with col_b:
        st.markdown("#### ROC Curve (Multi-Class OvR)")
        classes = [0, 1, 2]
        y_test_bin = label_binarize(y_test, classes=classes)
        label_names = {0: "No Disease", 1: "Mild", 2: "Severe"}
        line_colors = ["#e63946", "#2a9d8f", "#457b9d"]
        fig, ax = plt.subplots(figsize=(5, 4))
        for i in range(len(classes)):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=line_colors[i], lw=2,
                    label=f"{label_names[i]} (AUC={roc_auc:.2f})")
        ax.plot([0, 1], [0, 1], 'k--', lw=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend(loc="lower right", fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig)
        plt.close()

    st.markdown("#### Classification Report")
    report = classification_report(y_test, y_pred,
        target_names=["No Disease", "Mild", "Severe"], output_dict=True)
    report_df = pd.DataFrame(report).transpose().round(3)
    st.dataframe(report_df, use_container_width=True)

    st.markdown("#### Feature Importances")
    importances = model.feature_importances_
    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    colors_bar = ["#e63946" if v >= feat_imp.quantile(0.75) else "#adb5bd" for v in feat_imp]
    feat_imp.plot(kind='barh', ax=ax, color=colors_bar)
    ax.set_title("Feature Importances (Random Forest)")
    ax.set_xlabel("Importance")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    st.pyplot(fig)
    plt.close()

# ---- TAB 3: Data Insights ----
with tab3:
    st.markdown("#### Dataset Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", len(df))
    col2.metric("Features", len(df.columns) - 1)
    col3.metric("Target Classes", df['prediction'].nunique())

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Target Distribution")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        target_counts = df['prediction'].value_counts().sort_index()
        colors_dist = ["#2a9d8f", "#e9c46a", "#e63946"]
        bars = ax.bar(["No Disease\n(0)", "Mild\n(1)", "Severe\n(2)"],
                      target_counts.values, color=colors_dist, edgecolor='white', linewidth=1.5)
        for bar, val in zip(bars, target_counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    str(val), ha='center', fontweight='bold')
        ax.set_title("Class Distribution")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig)
        plt.close()

    with col_b:
        st.markdown("#### Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(6, 5))
        corr = df.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".1f", cmap='coolwarm',
                    ax=ax, linewidths=0.5, annot_kws={"size": 7})
        ax.set_title("Feature Correlation Matrix")
        st.pyplot(fig)
        plt.close()

    st.markdown("#### Dataset Sample")
    st.dataframe(df.head(10), use_container_width=True)

    st.markdown("#### Descriptive Statistics")
    st.dataframe(df.describe().round(2), use_container_width=True)
