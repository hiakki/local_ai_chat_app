#!/usr/bin/env python3
"""
FastAPI Backend Server for Llama 3.3 70B Chat

Provides REST API and Server-Sent Events (SSE) for streaming chat responses.
"""

import os
import json
import asyncio
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from llama_transformer import LlamaTransformer, get_transformer, PerfMetrics


# Configuration from environment
QUANT = os.getenv("QUANT", "Q4_K_M")
MODEL_PATH_ENV = os.getenv("MODEL_PATH", None)
CTX = int(os.getenv("CTX", "2048"))
GPU_LAYERS = int(os.getenv("GPU_LAYERS", "-1"))

# Resolve MODEL_PATH - handle directory vs file
def resolve_model_path(path_env: str | None, quant: str) -> str | None:
    """Resolve model path from environment variable."""
    if path_env is None:
        return None
    
    # Expand ~ to home directory
    path = os.path.expanduser(path_env)
    
    # If it's a directory, construct the full filename
    if os.path.isdir(path):
        filename = f"Llama-3.3-70B-Instruct-{quant}.gguf"
        full_path = os.path.join(path, filename)
        return full_path if os.path.exists(full_path) else None
    
    # If it's a file, return as-is
    if os.path.isfile(path):
        return path
    
    return None

MODEL_PATH = resolve_model_path(MODEL_PATH_ENV, QUANT)

# Global transformer instance
transformer: Optional[LlamaTransformer] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the transformer on startup."""
    global transformer
    print("\n🦙 Initializing Llama 3.3 70B Backend Server...")
    print(f"   Quantization: {QUANT}")
    print(f"   Context: {CTX} tokens")
    print(f"   GPU Layers: {GPU_LAYERS}")
    if MODEL_PATH:
        print(f"   Model Path: {MODEL_PATH}")
    
    transformer = get_transformer(
        quantization=QUANT,
        model_path=MODEL_PATH,
        n_ctx=CTX,
        n_gpu_layers=GPU_LAYERS,
    )
    
    print("\n✓ Server ready!")
    yield
    print("\n👋 Shutting down...")


app = FastAPI(
    title="Llama 3.3 70B Chat API",
    description="Chat API for Llama 3.3 70B running locally on Apple Silicon",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Message(BaseModel):
    role: str  # "user", "assistant", or "system"
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]
    max_tokens: int = 512
    temperature: float = 0.7
    stream: bool = True


class ChatResponse(BaseModel):
    response: str
    tokens_generated: int = 0


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model": "Llama-3.3-70B-Instruct",
        "quantization": QUANT,
        "context_window": CTX,
    }


@app.get("/api/models")
async def list_models():
    """List available quantization options."""
    return {
        "models": [
            {
                "id": name,
                "size_gb": info["size_gb"],
                "quality": info["quality"],
                "recommended_ram": info["recommended_ram"],
            }
            for name, info in LlamaTransformer.QUANT_OPTIONS.items()
        ],
        "current": QUANT,
    }


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    Chat completion endpoint.
    
    If stream=True, returns Server-Sent Events (SSE).
    If stream=False, returns complete response as JSON.
    """
    if transformer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    
    if request.stream:
        return StreamingResponse(
            stream_chat(messages, request.max_tokens, request.temperature),
            media_type="text/event-stream",
        )
    else:
        response = transformer.chat(
            messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            stream=False,
        )
        return ChatResponse(response=response)


async def stream_chat(
    messages: list[dict],
    max_tokens: int,
    temperature: float,
):
    """Generator for SSE streaming with performance metrics."""
    try:
        for token, metrics in transformer.chat_with_metrics(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
        ):
            if token is not None:
                # Streaming token
                data = json.dumps({"token": token, "done": False})
                yield f"data: {data}\n\n"
                await asyncio.sleep(0)  # Allow other tasks to run
            elif metrics is not None:
                # Final message with metrics
                data = json.dumps({
                    "token": "",
                    "done": True,
                    "metrics": metrics.to_dict(),
                })
                yield f"data: {data}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"


@app.post("/api/generate")
async def generate(prompt: str, max_tokens: int = 512, temperature: float = 0.7):
    """Raw text generation endpoint (non-chat format)."""
    if transformer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    response = transformer.generate(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=False,
    )
    return {"response": response}


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
