from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pydantic import BaseModel

app = FastAPI()

# Configure CORS to allow requests from the frontend
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained("OuteAI/Lite-Oute-1-300M-Instruct").to(device)
tokenizer = AutoTokenizer.from_pretrained("OuteAI/Lite-Oute-1-300M-Instruct")

class Message(BaseModel):
    message: str

@app.post("/chat")
async def chat(message: Message):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": message.message}
    ]
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(device)
    output = model.generate(
        input_ids,
        max_length=512,
        temperature=0.9,
        repetition_penalty=1.12,
        do_sample=True
    )
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return {"response": generated_text}

