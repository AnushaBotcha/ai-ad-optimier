import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="AI Ad Performance Optimizer", layout="centered")
st.title("AI-Based Ad Performance Optimizer")

# Load and train the model on sample data
@st.cache_resource
def load_and_train():
    df = pd.read_csv("data/ads_sample.csv")
    X = df.drop("ctr", axis=1)
    X = pd.get_dummies(X, columns=["headline"], drop_first=True)
    y = df["ctr"]
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, list(X.columns)

model, features = load_and_train()

st.subheader("Upload Your Ad")
image = st.file_uploader("Upload Image", type=["jpg", "png"])
headline = st.text_input("Ad Headline")
text_length = st.slider("Text Length", 10, 100, 20)
num_colors = st.slider("Number of Dominant Colors", 1, 10, 4)
brightness = st.slider("Image Brightness (0 to 1)", 0.0, 1.0, 0.7)

if st.button("Predict Performance"):
    if not headline:
        st.warning("Please enter a headline")
    else:
        input_data = {
            'image_brightness': [brightness],
            'text_length': [text_length],
            'num_colors': [num_colors]
        }
        for col in features:
            if col.startswith("headline_"):
                input_data[col] = [1 if col == f"headline_{headline}" else 0]
        df_input = pd.DataFrame(input_data)
        for f in features:
            if f not in df_input.columns:
                df_input[f] = 0

        ctr_pred = model.predict(df_input)[0]
        st.success(f"Predicted CTR: {ctr_pred * 100:.2f}%")

        if ctr_pred < 0.09:
            st.info(" Suggestion: Try using a more catchy headline or brighter image.")

if image:
    st.image(Image.open(image), caption="Uploaded Ad Image", use_column_width=True)
