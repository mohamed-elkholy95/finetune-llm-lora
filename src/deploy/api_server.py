"""FastAPI server for fine-tuned model."""
import logging
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import API_HOST, API_PORT, LORA_CONFIG, SUPPORTED_MODELS, DEFAULT_MODEL
from src.inference import generate_text, chat_completion, mock_generate

logger = logging.getLogger(__name__)

app = FastAPI(title="LoRA Fine-Tuned Model API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_model = None
_tokenizer = None
_model_name = DEFAULT_MODEL


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=10000)
    max_new_tokens: int = Field(default=128, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)


class GenerateResponse(BaseModel):
    prompt: str
    generated_text: str
    model: str


class ChatRequest(BaseModel):
    messages: List[dict] = Field(..., min_length=1)
    max_new_tokens: int = Field(default=256, ge=1, le=4096)


class ChatResponse(BaseModel):
    response: str
    model: str


class ModelInfo(BaseModel):
    model_name: str
    lora_config: dict
    supported_models: dict


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/model/info", response_model=ModelInfo)
async def model_info():
    return ModelInfo(
        model_name=_model_name,
        lora_config=LORA_CONFIG,
        supported_models=SUPPORTED_MODELS,
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    text = generate_text(req.prompt, _model, _tokenizer,
                         max_new_tokens=req.max_new_tokens,
                         temperature=req.temperature, top_p=req.top_p)
    return GenerateResponse(prompt=req.prompt, generated_text=text, model=_model_name)


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    response = chat_completion(req.messages, _model, _tokenizer,
                              max_new_tokens=req.max_new_tokens)
    return ChatResponse(response=response, model=_model_name)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT)
