import torch
import pandas as pd
from bnlp import CleanText
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ==============================
# LOAD MODEL HASIL TRAINING
# ==============================
model_path = "./Bangla_Smishing_Model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

id_to_label = {
    0: "normal",
    1: "promo",
    2: "smish"
}

# ==============================
# CLEANER
# ==============================
cleaner = CleanText(
    fix_unicode=True,
    unicode_norm=True,
    unicode_norm_form="NFKC",
    remove_url=True,
    remove_email=True,
    remove_emoji=True
)

# ==============================
# FUNCTION PREDICT
# ==============================
def predict_sms(text):
    cleaned_text = cleaner(str(text))

    inputs = tokenizer(
        cleaned_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=64
    )

    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        predicted_id = torch.argmax(probabilities, dim=-1).item()

    predicted_label = id_to_label[predicted_id]
    confidence = probabilities[0][predicted_id].item()

    return predicted_label, confidence

# ==============================
# LOAD VALIDATION DATASET
# ==============================
validation_path = "./validation_sms_large.csv"

df = pd.read_csv(validation_path)
df.columns = df.columns.str.strip()

correct = 0
total = len(df)

print("\n==============================")
print("VALIDATION RESULT")
print("==============================")

for index, row in df.iterrows():
    text = row["text"]
    expected_label = row["label"]

    predicted_label, confidence = predict_sms(text)

    is_correct = predicted_label == expected_label

    if is_correct:
        correct += 1

    print(f"\nData ke-{index + 1}")
    print("Text       :", text)
    print("Expected   :", expected_label)
    print("Predicted  :", predicted_label)
    print("Confidence :", round(confidence * 100, 2), "%")
    print("Status     :", "BENAR" if is_correct else "SALAH")

accuracy = correct / total

print("\n==============================")
print("SUMMARY")
print("==============================")
print("Total data :", total)
print("Correct      :", correct)
print("Wrong    :", total - correct)
print("Accuracy   :", round(accuracy * 100, 2), "%")