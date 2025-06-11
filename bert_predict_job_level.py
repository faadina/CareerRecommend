from transformers import BertTokenizer, BertForSequenceClassification
import torch
import numpy as np

# === Load Best Model (adjust folder name if needed) ===
model_path = "bert_model_90/final_model"  # or 70 or 80 if better
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)
model.eval()

# === Label Map (make sure it matches training encoding) ===
label_map = {
    0: "Entry-Level",
    1: "Mid-Level",
    2: "Senior-Level"
}

# === Predict Function ===
def predict_job_level(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    pred = torch.argmax(outputs.logits, dim=1).item()
    return label_map.get(pred, "Unknown")

# === Example Use ===
resume_text = "I have experience in data science, leadership, and managing engineering teams."
predicted_level = predict_job_level(resume_text)
print("Predicted Job Level:", predicted_level)
