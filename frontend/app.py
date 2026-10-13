import os

import requests
import streamlit as st

INFERENCE_URL = os.environ.get("INFERENCE_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Food-11 classifier", page_icon="🍲")
st.title("Food-11 classifier")

with st.sidebar:
    st.caption("Inference service")
    st.code(INFERENCE_URL, language=None)
    try:
        info = requests.get(f"{INFERENCE_URL}/model", timeout=5).json()
        st.success("connected")
        st.caption(f"model: {info.get('model_uri')}")
        st.caption(f"tracking: {info.get('tracking_uri')}")
    except requests.RequestException as exc:
        st.error(f"unreachable: {exc.__class__.__name__}")

uploaded = st.file_uploader("Upload a food image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    st.image(uploaded, width=300)
    files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
    try:
        response = requests.post(f"{INFERENCE_URL}/predict", files=files, timeout=60)
    except requests.RequestException as exc:
        st.error(f"Could not reach the inference service at {INFERENCE_URL}: {exc}")
    else:
        if response.ok:
            result = response.json()
            st.write(f"**Prediction:** {result['category']} ({result['confidence']:.1%})")
            if result.get("top3"):
                st.caption("Top 3")
                for item in result["top3"]:
                    st.progress(item["confidence"], text=f"{item['category']} — {item['confidence']:.1%}")
        else:
            st.error(f"Inference service returned {response.status_code}: {response.text}")
