from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

app = FastAPI()

class RequestBody(BaseModel):
    text: str

model_path = "Legeva1937/AITutor"
token = os.getenv("HF_TOKEN")  # Токен, если требуется

tokenizer = AutoTokenizer.from_pretrained(model_path, token=token)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    token=token
)
model.eval()

dialog_history = []

@app.get("/")
async def root():
    return {"message": "API is running"}

@app.post("/predict/")
async def predict(request: RequestBody):
    global dialog_history
    dialog_history.append(f"Пользователь: {request.text}")
    prompt = "\n".join(dialog_history + ["Ассистент:"])
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = response.split("Ассистент:")[-1].strip()
    dialog_history.append(f"Ассистент: {answer}")
    return {"response": answer}

@app.post("/reset/")
async def reset():
    global dialog_history
    dialog_history = []
    return {"message": "Диалог сброшен"}
