# NLI Text Classifier

**Live Demo:** [Hugging Face Space](https://huggingface.co/spaces/eyad222/nli-app1)

---

## 📌 Project Description

The **NLI Text Classifier** is an application that performs **Natural Language Inference (NLI)** to determine whether a given text **entails**, **contradicts**, or is **neutral** with respect to a specified label.

For example, given a text about Linux distributions and the label `linux distro comparison`, the app predicts if the text supports (entailment), contradicts (contradiction), or is neutral to the given label.

This project was built using:

- **PyTorch** and **Hugging Face Transformers** for NLP modeling
- **Streamlit** for an interactive GUI
- **Docker** for containerization and deployment

The project was developed **under the guidance of Prof. Ashraf Mohamed and Eng. Mostafa Touny**, focusing on bridging theoretical NLP assumptions to practical, real-world usage.

---

## ⚠️ Problems in the Original Project

1. **Static Inputs**: The original code required hardcoded text and label inputs.
2. **Inefficient Model Loading**: The model loaded every time a prediction was made, causing delays.
3. **Limited Output**: Only probabilities were shown, with no human-readable label.
4. **No GUI**: Users had to modify and run Python scripts manually.
5. **Not Real-World Ready**: The code assumed ideal theoretical conditions (i.i.d data, full batch access, etc.) and could not handle streaming inputs or dynamic use cases.

---

## 🏭 Real-World Assumptions and How They Were Solved

### Problem 1: i.i.d Data → Streaming Input

**Theoretical assumption:** The model assumes that all data points are independent and identically distributed (i.i.d.) and available in batches.

**Reality:** Users provide text input one at a time, dynamically. The model needs to handle streaming single examples rather than large batches.

**How we solved it:**
- Use Streamlit to take real-time input
- Model inference runs per input
- Added caching so the model loads only once

**Code Example:**
```python
import streamlit as st
from app import classify, load_model

tokenizer, model, device = load_model()  # cached model load

st.title("NLI Text Classifier")
user_text = st.text_area("Enter your text:")
label_input = st.text_input("Enter label:")

if st.button("Predict"):
    prob, label_name = classify(user_text, label_input)
    st.write(f"Predicted label: **{label_name}**")
    st.write(f"Probability: {prob}")
```

**Benefit:** The app now handles streaming user input, not just static hardcoded examples.

---

### Problem 2: Fixed Graph → Dynamic Text and Labels

**Theoretical assumption:** The original model was built for a fixed set of inputs and labels.

**Reality:** Users can enter any text and any label, dynamically.

**How we solved it:**
- Build hypothesis on-the-fly using user label
- Process input text dynamically

**Code Example:**
```python
def classify(sequence, label):
    premise = sequence
    hypothesis = f"This example is {label}."  # dynamic label
    inputs = tokenizer(
        premise, hypothesis, return_tensors="pt", truncation=True
    ).to(device)
    logits = model(**inputs).logits
    idx = torch.tensor([0, 2], device=logits.device)
    entail_contradiction_logits = logits.index_select(1, idx)
    probs = torch.softmax(entail_contradiction_logits, dim=1)
    prob_label_is_true = probs[0, 1].item()
    label_name = "Entailment" if prob_label_is_true > 0.5 else "Contradiction"
    return round(prob_label_is_true, 2), label_name
```

**Benefit:** Now supports any text and label combination, not a fixed pre-defined set.

---

### Problem 3: Full Batch Access → Real-Time Single Example Inference

**Theoretical assumption:** The model assumes full batch access (all data in memory) for inference.

**Reality:** The app must work per single user input in real-time without pre-loading batches.

**How we solved it:**
- Removed batch processing assumption
- Model predicts on one input at a time
- Softmax and probability computation done per example

**Code Example:**
```python
# Single example inference
inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True).to(device)
logits = model(**inputs).logits  # shape [1, 3]
# select contradiction and entailment logits only
idx = torch.tensor([0, 2], device=logits.device)
entail_contradiction_logits = logits.index_select(1, idx)
probs = torch.softmax(entail_contradiction_logits, dim=1)
prob_label_is_true = probs[0, 1].item()
```

**Benefit:** Users get real-time predictions without waiting for batch processing, making the app practical for industry usage.

---

## 🏗️ How to Download & Run Locally

```bash
git clone <repo-url>
cd NLI_App
python -m venv venv

# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

---

## 🐳 Docker Deployment

```bash
docker build -t nli-text-classifier .
docker run -p 8501:8501 nli-text-classifier
```

---

## 🎯 Live Version

Access the live application here: [https://huggingface.co/spaces/eyad222/nli-app1](https://huggingface.co/spaces/eyad222/nli-app1)

---

## 📚 References & Instructors

- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [PyTorch Documentation](https://pytorch.org/docs/)
- **Prof. Ashraf Mohamed** – Academic guidance
- **Eng. Mostafa Touny** – Project supervision and instructions
