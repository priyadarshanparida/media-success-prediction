import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# Layout options for Streamlit are: "centered" (default) and "wide"
st.set_page_config(page_title="Hit Predictor Dashboard", layout="centered")
# Use markdown for custom font size (Streamlit doesn't support direct font size change for st.title)
st.markdown("<h2 style='font-size:2.5rem;'>🎵 Real-Time Hit Predictor Dashboard</h2>", unsafe_allow_html=True)

# Load model + reference info
@st.cache_resource
def load_resources():
    model = joblib.load("../data/extracts/log_reg_model.pkl")
    feature_columns = pd.read_csv("../data/extracts/feature_columns.csv", header=None)[0].tolist()
    feature_columns = [c for c in feature_columns if c != "0"]
    feature_summary = pd.read_csv("../data/extracts/feature_summary.csv", index_col=0)
    return model, feature_columns, feature_summary

log_reg, feature_columns, feature_summary = load_resources()

# Preset toggle
st.subheader("☰ Input Controls")
st.write("")  # Adds a blank line for spacing

profile_choice = st.radio(
    "**Select Preset Profile**",
    ["Default Profile 1", "Default Profile 2"],
    horizontal=True
)

# Numeric feature list (from your training data)
numeric_features = [
    'danceability', 'energy', 'loudness', 'acousticness', 'speechiness',
    'instrumentalness', 'liveness', 'tempo_norm', 'log_mentions',
    'sentiment_intensity', 'engagement_score'
]

# Genre list for dropdown
genre_columns = [
    'genre_country', 'genre_edm', 'genre_folk', 'genre_hiphop', 'genre_jazz_blues',
    'genre_latin', 'genre_metal', 'genre_other', 'genre_pop', 'genre_rnb_soul',
    'genre_rock', 'genre_unknown'
]

# Sentiment components (fixed as fractions summing to 1)
sentiment_cols = ['sent_negative', 'sent_neutral', 'sent_positive']

# Playlist indicator
playlist_col = 'has_playlist_genre'

# Map renamed labels to the intended presets
high_hit_labels = ["Default Profile 1", "High-Hit Profile (≈80–90%)"]

# Helper: Feature Importance Visualization (Coefficient-Based)
def show_shap_feature_drivers(model, input_df, prob_value):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import streamlit as st

    # --- compute absolute coefficient magnitude ---
    coefs = model.coef_.flatten()
    coef_df = pd.DataFrame({
        "Feature": model.feature_names_in_,
        "Importance": np.abs(coefs)
    })
    top5 = coef_df.sort_values("Importance", ascending=False).head(5)

    # --- two-column layout: probability | importance chart ---
    colA, colB = st.columns([1, 1])
    with colA:
        st.metric(label="🎯 Predicted Hit Probability",
                  value=f"{prob_value*100:.2f}%")
        st.progress(prob_value)
    with colB:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.barh(top5["Feature"], top5["Importance"],
                color="#4F81BD")  # uniform blue bars
        ax.set_xlabel("Coefficient Magnitude (Feature Importance)")
        ax.set_title("Top 5 Most Influential Features")
        ax.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig)


# Derive slider defaults from feature_summary quantiles
def get_default_values(profile):
    defaults = {}
    for f in numeric_features:
        stats = feature_summary[f]  # access column instead of row
        if profile in high_hit_labels:
            defaults[f] = min(float(stats["90%"]) * 1.3, float(stats["max"]))
        else:
            defaults[f] = float(stats["10%"]) if "10%" in stats else float(stats["min"])

    # Sentiment and playlist defaults
    if profile in high_hit_labels:
        defaults.update({'sent_negative': 0.05, 'sent_neutral': 0.15, 'sent_positive': 0.8})
        defaults[playlist_col] = 1
    else:
        defaults.update({'sent_negative': 0.3, 'sent_neutral': 0.6, 'sent_positive': 0.1})
        defaults[playlist_col] = 0
    return defaults

preset_defaults = get_default_values(profile_choice)

# Collapsible input section
with st.expander("🎚 Expand / Collapse Feature Inputs", expanded=False):

    # Genre selection
    genre_choice = st.selectbox("Select Genre", [g.replace("genre_", "").title() for g in genre_columns], index=8)

    # Playlist toggle
    has_playlist = st.checkbox("Has Playlist Genre Association", value=bool(preset_defaults[playlist_col]))

    # Two-column slider layout
    st.markdown("#### Audio & Sentiment Features")
    cols = st.columns(2)
    user_inputs = {}
    for i, f in enumerate(numeric_features):
        col = cols[i % 2]
        stats = feature_summary[f]  # access as column, not row
        min_val = float(stats["min"])
        max_val = float(stats["max"])
        default_val = preset_defaults[f]
        user_inputs[f] = col.slider(f, min_val, max_val, float(default_val))


    # Sentiment sliders
    st.markdown("#### Sentiment Composition")
    col1, col2, col3 = st.columns(3)
    user_inputs['sent_negative'] = col1.slider("Negative", 0.0, 1.0, preset_defaults['sent_negative'])
    user_inputs['sent_neutral'] = col2.slider("Neutral", 0.0, 1.0, preset_defaults['sent_neutral'])
    user_inputs['sent_positive'] = col3.slider("Positive", 0.0, 1.0, preset_defaults['sent_positive'])

# Build model input row and clean DataFrame
input_row = {c: 0 for c in feature_columns}
input_row.update(user_inputs)
input_row[playlist_col] = int(has_playlist)

selected_genre = st.session_state.get("genre_choice", genre_columns[0].replace("genre_", "").title())
for g in genre_columns:
    input_row[g] = 1 if g == f"genre_{selected_genre.lower()}" else 0

input_df = pd.DataFrame([input_row])
input_df = input_df.loc[:, log_reg.feature_names_in_].copy()

# Prediction button & output
if st.button("🎯 Predict Hit Probability"):
    try:
        probs = log_reg.predict_proba(input_df)[:, 1]
        prob_value = float(probs[0])
        # st.metric(label="Predicted Hit Probability", value=f"{prob_value*100:.2f}%")
        # st.progress(prob_value)
        show_shap_feature_drivers(log_reg, input_df, prob_value)
    except Exception as e:
        st.error(f"Prediction failed: {e}")