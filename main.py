from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from huggingface_hub import login

# === REPLACE WITH YOUR HUGGING FACE TOKEN ===
login("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")  # ← CHANGE THIS

# Load AI model
predictor = pipeline("text-generation", model="meta-llama/Meta-Llama-3.1-8B-Instruct")

app = FastAPI()

class PredictionRequest(BaseModel):
    question: str

@app.post("/predict")
def predict(request: PredictionRequest):
    prompt = f"Answer in 1 short sentence with % chance: {request.question}"
    result = predictor(prompt, max_new_tokens=30, temperature=0.3)[0]['generated_text']
    answer = result.split("Answer:")[-1].strip()
    return {"prediction": answer}