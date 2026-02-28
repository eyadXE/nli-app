import streamlit as st
import requests

st.title("Text Classification App 📝")

text = st.text_area("Enter text here")
label = st.text_input("Enter label")

if st.button("Predict"):
    if text.strip() == "" or label.strip() == "":
        st.warning("Please enter both text and label")
    else:
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json={"text": text, "label": label}
        )
        result = response.json()
        prob = result['probability']

        # Determine decision based on probability / uncertain
        if prob == "Uncertain":
            decision = "Uncertain"
        elif prob >= 0.5:
            decision = "Entailment (matches label)"
        else:
            decision = "Contradiction (does not match label)"

        st.markdown(f"**Input Label:** {label}")
        st.markdown(f"**Probability of being true:** {prob}")
        st.markdown(f"**Decision:** {decision}")