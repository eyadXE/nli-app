import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from fastapi import FastAPI
from pydantic import BaseModel

# ----------------- Load model once -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "facebook/bart-large-mnli"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
model.eval()

temperature = 1.2

# ----------------- FastAPI app -----------------
app = FastAPI(title="Text Classification API")

class InputData(BaseModel):
    text: str
    label: str

def classify(sequence, label):
    hypothesis = f"This example is {label}."
    inputs = tokenizer(
        sequence,
        hypothesis,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits

    idx = torch.tensor([0, 2], device=logits.device)
    logits = logits.index_select(1, idx)

    probs = torch.softmax(logits / temperature, dim=1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-8))

    if entropy > 0.8:
        return "Uncertain"
    return round(probs[0, 1].item(), 3)

# ----------------- API Endpoint -----------------
@app.post("/predict")
def predict(data: InputData):
    result = classify(data.text, data.label)
    return {"probability": result}