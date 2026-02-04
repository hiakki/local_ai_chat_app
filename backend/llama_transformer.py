#!/usr/bin/env python3
"""
Llama 3.3 70B Transformer for Mac M4

This implementation uses llama-cpp-python with GGUF quantized models
to run the massive 70B model on limited RAM through:
1. Aggressive quantization (Q2_K, Q3_K_S, or Q4_K_S)
2. Memory mapping (mmap) - loads from disk on demand
3. Metal GPU acceleration for Apple Silicon
4. Streaming layer-by-layer processing

Reference: https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct
"""

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional

# Check for llama-cpp-python
try:
    from llama_cpp import Llama
except ImportError:
    print("Error: llama-cpp-python not installed.")
    print("Install with: pip install llama-cpp-python")
    print("\nFor Metal GPU support on Mac, install with:")
    print('CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python --force-reinstall --no-cache-dir')
    sys.exit(1)

from huggingface_hub import hf_hub_download


@dataclass
class PerfMetrics:
    """Performance metrics for a generation (from llama.cpp internals)."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_time_ms: float = 0
    prompt_eval_time_ms: float = 0
    completion_time_ms: float = 0
    tokens_per_second: float = 0  # Generation speed
    prompt_per_second: float = 0  # Prompt processing speed
    
    def to_dict(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_time_ms": round(self.total_time_ms, 2),
            "prompt_eval_time_ms": round(self.prompt_eval_time_ms, 2),
            "completion_time_ms": round(self.completion_time_ms, 2),
            "tokens_per_second": round(self.tokens_per_second, 2),
            "prompt_per_second": round(self.prompt_per_second, 2),
        }


class LlamaTransformer:
    """
    Memory-efficient Llama 3.3 70B transformer for Mac M4.
    
    Uses GGUF quantized models with mmap for on-demand loading.
    Only the actively used portions of the model stay in RAM.
    """
    
    # Quantization options from smallest to largest
    # Full list from: https://huggingface.co/bartowski/Llama-3.3-70B-Instruct-GGUF
    # For 48GB RAM, Q4_K_M recommended (best quality/size balance)
    # For 16GB RAM, Q3_K_S or Q2_K recommended
    QUANT_OPTIONS = {
        # ============ 1-bit quantization ============
        "IQ1_M": {
            "size_gb": 17,
            "quality": "Extreme quantization, lowest quality",
            "recommended_ram": "10GB+",
            "repo": "bartowski/Llama-3.3-70B-Instruct-GGUF",
            "filename": "Llama-3.3-70B-Instruct-IQ1_M.gguf"
        },
        # ============ 2-bit quantization ============
        "IQ2_XXS": {
            "size_gb": 19,
            "quality": "Extreme quantization, smallest 2-bit",
            "recommended_ram": "10GB+",
            "repo": "bartowski/Llama-3.3-70B-Instruct-GGUF",
            "filename": "Llama-3.3-70B-Instruct-IQ2_XXS.gguf"
        },
        "IQ2_XS": {
            "size_gb": 21,
            "quality": "Very aggressive quantization",
            "recommended_ram": "11GB+",
            "repo": "bartowski/Llama-3.3-70B-Instruct-GGUF",
            "filename": "Llama-3.3-70B-Instruct-IQ2_XS.gguf"
        },
        "IQ2_S": {
            "size_gb": 22,
            "quality": "Aggressive quantization (I-quant)",
            "recommended_ram": "12GB+",
            "repo": "bartowski/Llama-3.3-70B-Instruct-GGUF",
            "filename": "Llama-3.3-70B-Instruct-IQ2_S.gguf"
        },
        "IQ2_M": {
            "size_gb": 24,
            "quality": "Aggressive quantization (I-quant)",
            "recommended_ram": "14GB+",
            "repo": "bartowski/Llama-3.3-70B-Instruct-GGUF",
            "filename": "Llama-3.3-70B-Instruct-IQ2_M.gguf"
        },
        "Q2_K": {
            "size_gb": 26,
            "quality": "Low quality, small size",
            "recommended_ram": "14GB+",
            "repo": "bartowski/Llama-3.3-70B-Instruct-GGUF",
            "filename": "Llama-3.3-70B-Instruct-Q2_K.gguf"
        },
        "Q2_K_L": {
            "size_gb": 27,
            "quality": "Low quality (Q8 embed/output)",
            "recommended_ram": "16GB+",
            "repo": "bartowski/Llama-3.3-70B-Instruct-GGUF",
            "filename": "Llama-3.3-70B-Instruct-Q2_K_L.gguf"
        },
        # ============ 3-bit quantization ============
        "IQ3_XXS": {
            "size_gb": 28,
            "quality": "Medium-low quality (I-quant)",
            "recommended_ram": "16GB+",
            "repo": "bartowski/Llama-3.3-70B-Instruct-GGUF",
            "filename": "Llama-3.3-70B-Instruct-IQ3_XXS.gguf"
        },
        "IQ3_XS": {
            "size_gb": 29,
            "quality": "Medium-low quality (I-quant)",
            "recommended_ram": "16GB+",
            "repo": "bartowski/Llama-3.3-70B-Instruct-GGUF",
            "filename": "Llama-3.3-70B-Instruct-IQ3_XS.gguf"
        },
        "Q3_K_S": {
            "size_gb": 31,
            "quality": "Medium quality",
            "recommended_ram": "18GB+",
            "repo": "bartowski/Llama-3.3-70B-Instruct-GGUF",
            "filename": "Llama-3.3-70B-Instruct-Q3_K_S.gguf"
        },
        "IQ3_M": {
            "size_gb": 32,
            "quality": "Medium quality (I-quant)",
            "recommended_ram": "18GB+",
            "repo": "bartowski/Llama-3.3-70B-Instruct-GGUF",
            "filename": "Llama-3.3-70B-Instruct-IQ3_M.gguf"
        },
        "Q3_K_M": {
            "size_gb": 34,
            "quality": "Medium-good quality",
            "recommended_ram": "20GB+",
            "repo": "bartowski/Llama-3.3-70B-Instruct-GGUF",
            "filename": "Llama-3.3-70B-Instruct-Q3_K_M.gguf"
        },
        "Q3_K_L": {
            "size_gb": 37,
            "quality": "Good quality (larger 3-bit)",
            "recommended_ram": "22GB+",
            "repo": "bartowski/Llama-3.3-70B-Instruct-GGUF",
            "filename": "Llama-3.3-70B-Instruct-Q3_K_L.gguf"
        },
        "Q3_K_XL": {
            "size_gb": 38,
            "quality": "Good quality (Q8 embed/output)",
            "recommended_ram": "24GB+",
            "repo": "bartowski/Llama-3.3-70B-Instruct-GGUF",
            "filename": "Llama-3.3-70B-Instruct-Q3_K_XL.gguf"
        },
        # ============ 4-bit quantization ============
        "IQ4_XS": {
            "size_gb": 38,
            "quality": "Good quality (I-quant 4-bit)",
            "recommended_ram": "24GB+",
            "repo": "bartowski/Llama-3.3-70B-Instruct-GGUF",
            "filename": "Llama-3.3-70B-Instruct-IQ4_XS.gguf"
        },
        "Q4_0": {
            "size_gb": 40,
            "quality": "Legacy format, ARM CPU optimization",
            "recommended_ram": "24GB+",
            "repo": "bartowski/Llama-3.3-70B-Instruct-GGUF",
            "filename": "Llama-3.3-70B-Instruct-Q4_0.gguf"
        },
        "IQ4_NL": {
            "size_gb": 40,
            "quality": "Similar to IQ4_XS, ARM CPU optimization",
            "recommended_ram": "24GB+",
            "repo": "bartowski/Llama-3.3-70B-Instruct-GGUF",
            "filename": "Llama-3.3-70B-Instruct-IQ4_NL.gguf"
        },
        "Q4_K_S": {
            "size_gb": 40,
            "quality": "Good quality, slightly lower than Q4_K_M",
            "recommended_ram": "24GB+",
            "repo": "bartowski/Llama-3.3-70B-Instruct-GGUF",
            "filename": "Llama-3.3-70B-Instruct-Q4_K_S.gguf"
        },
        "Q4_0_4_4": {
            "size_gb": 40,
            "quality": "ARM CPU optimized (NEON)",
            "recommended_ram": "24GB+",
            "repo": "bartowski/Llama-3.3-70B-Instruct-GGUF",
            "filename": "Llama-3.3-70B-Instruct-Q4_0_4_4.gguf"
        },
        "Q4_0_4_8": {
            "size_gb": 40,
            "quality": "ARM CPU optimized (SVE 256)",
            "recommended_ram": "24GB+",
            "repo": "bartowski/Llama-3.3-70B-Instruct-GGUF",
            "filename": "Llama-3.3-70B-Instruct-Q4_0_4_8.gguf"
        },
        "Q4_0_8_8": {
            "size_gb": 40,
            "quality": "AVX2/AVX512 CPU optimized",
            "recommended_ram": "24GB+",
            "repo": "bartowski/Llama-3.3-70B-Instruct-GGUF",
            "filename": "Llama-3.3-70B-Instruct-Q4_0_8_8.gguf"
        },
        "Q4_K_M": {
            "size_gb": 43,
            "quality": "Very good quality, recommended default",
            "recommended_ram": "32GB+",
            "repo": "bartowski/Llama-3.3-70B-Instruct-GGUF",
            "filename": "Llama-3.3-70B-Instruct-Q4_K_M.gguf"
        },
        "Q4_K_L": {
            "size_gb": 43,
            "quality": "Excellent quality (Q8 embed/output)",
            "recommended_ram": "32GB+",
            "repo": "bartowski/Llama-3.3-70B-Instruct-GGUF",
            "filename": "Llama-3.3-70B-Instruct-Q4_K_L.gguf"
        },
        # ============ 5-bit quantization ============
        "Q5_K_S": {
            "size_gb": 49,
            "quality": "High quality",
            "recommended_ram": "48GB+",
            "repo": "bartowski/Llama-3.3-70B-Instruct-GGUF",
            "filename": "Llama-3.3-70B-Instruct-Q5_K_S.gguf"
        },
        "Q5_K_M": {
            "size_gb": 50,
            "quality": "High quality, recommended",
            "recommended_ram": "48GB+",
            "repo": "bartowski/Llama-3.3-70B-Instruct-GGUF",
            "filename": "Llama-3.3-70B-Instruct-Q5_K_M.gguf"
        },
        "Q5_K_L": {
            "size_gb": 51,
            "quality": "High quality (Q8 embed/output)",
            "recommended_ram": "48GB+",
            "repo": "bartowski/Llama-3.3-70B-Instruct-GGUF",
            "filename": "Llama-3.3-70B-Instruct-Q5_K_L.gguf"
        },
        # ============ 6-bit quantization ============
        "Q6_K": {
            "size_gb": 58,
            "quality": "Very high quality, near perfect",
            "recommended_ram": "64GB+",
            "repo": "bartowski/Llama-3.3-70B-Instruct-GGUF",
            "filename": "Llama-3.3-70B-Instruct-Q6_K.gguf"
        },
        "Q6_K_L": {
            "size_gb": 58,
            "quality": "Very high quality (Q8 embed/output)",
            "recommended_ram": "64GB+",
            "repo": "bartowski/Llama-3.3-70B-Instruct-GGUF",
            "filename": "Llama-3.3-70B-Instruct-Q6_K_L.gguf"
        },
        # ============ 8-bit quantization ============
        "Q8_0": {
            "size_gb": 75,
            "quality": "Extremely high quality, max available",
            "recommended_ram": "80GB+",
            "repo": "bartowski/Llama-3.3-70B-Instruct-GGUF",
            "filename": "Llama-3.3-70B-Instruct-Q8_0.gguf"
        },
        # ============ 16-bit (full precision) ============
        "F16": {
            "size_gb": 141,
            "quality": "Full F16 weights, original precision",
            "recommended_ram": "160GB+",
            "repo": "bartowski/Llama-3.3-70B-Instruct-GGUF",
            "filename": "Llama-3.3-70B-Instruct-f16.gguf"
        },
    }
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        quantization: str = "Q4_K_M",
        n_ctx: int = 2048,
        n_batch: int = 512,  # Increased default for faster prompt processing
        n_gpu_layers: int = -1,  # -1 = use all layers on GPU (Metal)
        use_mmap: bool = True,
        use_mlock: bool = False,
        verbose: bool = True,
        cache_dir: Optional[str] = None,
        flash_attn: bool = True,  # Flash attention for speed
        offload_kqv: bool = True,  # Offload KV cache to GPU
    ):
        """
        Initialize the Llama transformer.
        
        Args:
            model_path: Path to GGUF model file. If None, downloads automatically.
            quantization: Quantization level (Q2_K, Q3_K_S, Q3_K_M, Q4_K_S, IQ2_XXS, IQ2_XS)
            n_ctx: Context window size. Smaller = less RAM. Default 2048.
            n_batch: Batch size for prompt processing. Default 512 (larger = faster).
            n_gpu_layers: Layers to offload to GPU. -1 = all (recommended for Metal).
            use_mmap: Memory map model file (loads on demand). Default True.
            use_mlock: Lock model in RAM (disable for low RAM). Default False.
            verbose: Print loading progress. Default True.
            cache_dir: Directory to cache downloaded models.
            flash_attn: Use flash attention for faster inference. Default True.
            offload_kqv: Offload KV cache to GPU. Default True.
        """
        self.flash_attn = flash_attn
        self.offload_kqv = offload_kqv
        self.quantization = quantization
        self.verbose = verbose
        
        if quantization not in self.QUANT_OPTIONS:
            raise ValueError(
                f"Unknown quantization: {quantization}. "
                f"Options: {list(self.QUANT_OPTIONS.keys())}"
            )
        
        quant_info = self.QUANT_OPTIONS[quantization]
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Llama 3.3 70B Instruct - {quantization} Quantization")
            print(f"{'='*60}")
            print(f"Model size: ~{quant_info['size_gb']}GB")
            print(f"Quality: {quant_info['quality']}")
            print(f"Recommended RAM: {quant_info['recommended_ram']}")
            print(f"Context window: {n_ctx} tokens")
            print(f"Memory mapping: {'Enabled' if use_mmap else 'Disabled'}")
            print(f"{'='*60}\n")
        
        # Find or download model
        if model_path is None:
            # First check for existing model in ~/llama-models (from brew_setup)
            local_model_dir = Path.home() / "llama-models"
            local_model_path = local_model_dir / quant_info["filename"]
            
            if local_model_path.exists():
                model_path = str(local_model_path)
                if verbose:
                    print(f"✓ Found existing model: {model_path}")
            else:
                # Download if not found locally
                model_path = self._download_model(quant_info, cache_dir)
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        if verbose:
            print(f"Loading model from: {model_path}")
            print("This may take a few minutes on first load...")
        
        # Initialize llama.cpp model with optimized settings for Apple Silicon
        # Note: verbose=False suppresses llama.cpp internal output (1000+ lines)
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_batch=n_batch,
            n_gpu_layers=n_gpu_layers,
            use_mmap=use_mmap,
            use_mlock=use_mlock,
            verbose=False,
            
            # === PERFORMANCE OPTIMIZATIONS ===
            
            # Thread settings for Apple Silicon M4
            # M4 Pro has 10 performance + 4 efficiency cores
            n_threads=10,          # Use performance cores for generation
            n_threads_batch=10,    # Use performance cores for batch processing
            
            # Flash Attention - faster attention with less memory
            flash_attn=self.flash_attn,
            
            # KV Cache Quantization - significant memory savings
            # F16 (type 1) is default, Q8_0 (type 4) saves more memory
            type_k=1,  # GGML_TYPE_F16
            type_v=1,  # GGML_TYPE_F16
            
            # Offload KV cache to GPU for faster inference
            offload_kqv=self.offload_kqv,
        )
        
        if verbose:
            print("\n✓ Model loaded successfully!")
            print(f"  Using Metal GPU acceleration: {'Yes' if n_gpu_layers != 0 else 'No'}")
    
    def _download_model(self, quant_info: dict, cache_dir: Optional[str]) -> str:
        """Download the GGUF model from Hugging Face."""
        if self.verbose:
            print("\n" + "=" * 60)
            print("📥 DOWNLOADING MODEL - THIS WILL TAKE A WHILE!")
            print("=" * 60)
            print(f"File: {quant_info['filename']}")
            print(f"From: {quant_info['repo']}")
            print(f"Size: ~{quant_info['size_gb']}GB")
            print("")
            print("⏳ Estimated time: 10-30 minutes (depends on internet)")
            print("   Progress bar will appear below...")
            print("   DO NOT close this window!")
            print("=" * 60 + "\n")
        
        # Use default HF cache or custom directory
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        
        model_path = hf_hub_download(
            repo_id=quant_info["repo"],
            filename=quant_info["filename"],
            cache_dir=cache_dir,
        )
        
        return model_path
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        stop: Optional[list] = None,
        stream: bool = False,
    ) -> str | Generator[str, None, None]:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input text prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 = deterministic)
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling
            repeat_penalty: Penalty for repetition
            stop: Stop sequences
            stream: If True, yields tokens as they're generated
            
        Returns:
            Generated text or generator if streaming
        """
        if stream:
            return self._generate_stream(
                prompt, max_tokens, temperature, top_p, top_k, repeat_penalty, stop
            )
        
        output = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            stop=stop or [],
            echo=False,
        )
        
        return output["choices"][0]["text"]
    
    def _generate_stream(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repeat_penalty: float,
        stop: Optional[list],
    ) -> Generator[str, None, None]:
        """Stream tokens as they're generated."""
        for output in self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            stop=stop or [],
            echo=False,
            stream=True,
        ):
            yield output["choices"][0]["text"]
    
    def chat(
        self,
        messages: list[dict],
        max_tokens: int = 512,
        temperature: float = 0.7,
        stream: bool = False,
    ) -> str | Generator[str, None, None]:
        """
        Chat completion with Llama 3.3 format.
        
        Args:
            messages: List of {"role": "user/assistant/system", "content": "..."}
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stream: If True, yields tokens as generated
            
        Returns:
            Assistant response or generator if streaming
        """
        # Format messages for Llama 3.3 chat template
        prompt = self._format_chat_prompt(messages)
        
        return self.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream,
            stop=["<|eot_id|>", "<|end_of_text|>"],
        )
    
    def chat_with_metrics(
        self,
        messages: list[dict],
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> Generator[tuple[str | None, PerfMetrics | None], None, None]:
        """
        Chat completion with performance metrics tracking.
        
        Yields (token, None) for each token, then (None, PerfMetrics) at the end.
        
        Args:
            messages: List of {"role": "user/assistant/system", "content": "..."}
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Yields:
            Tuples of (token, None) during generation, (None, metrics) at end
        """
        prompt = self._format_chat_prompt(messages)
        
        # Reset timings before generation (if available)
        try:
            self.llm.reset_timings()
        except AttributeError:
            pass  # Older versions may not have this
        
        # Fallback timing with manual measurement
        start_time = time.perf_counter()
        first_token_time = None
        token_count = 0
        
        for output in self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
            top_k=40,
            repeat_penalty=1.1,
            stop=["<|eot_id|>", "<|end_of_text|>"],
            echo=False,
            stream=True,
        ):
            token = output["choices"][0]["text"]
            if first_token_time is None:
                first_token_time = time.perf_counter()
            token_count += 1
            yield (token, None)
        
        end_time = time.perf_counter()
        
        # Try to get actual timings from llama.cpp
        metrics = None
        
        try:
            timings = None
            
            # Method 1: Use llama_perf_context from llama_cpp library (most reliable)
            try:
                import llama_cpp.llama_cpp as llama_cpp_lib
                if hasattr(llama_cpp_lib, 'llama_perf_context') and hasattr(self.llm, '_ctx'):
                    perf = llama_cpp_lib.llama_perf_context(self.llm._ctx.ctx)
                    if hasattr(perf, 'n_eval') and perf.n_eval > 0:
                        t_p_eval_ms = getattr(perf, 't_p_eval_ms', 0)
                        n_p_eval = getattr(perf, 'n_p_eval', 0)
                        t_eval_ms = getattr(perf, 't_eval_ms', 0)
                        n_eval = getattr(perf, 'n_eval', 0)
                        
                        timings = {
                            "prompt_n": n_p_eval,
                            "prompt_ms": t_p_eval_ms,
                            "prompt_per_second": (n_p_eval / t_p_eval_ms * 1000) if t_p_eval_ms > 0 else 0,
                            "predicted_n": n_eval,
                            "predicted_ms": t_eval_ms,
                            "predicted_per_second": (n_eval / t_eval_ms * 1000) if t_eval_ms > 0 else 0,
                        }
            except Exception:
                pass
            
            # Method 2: Try llm.timings property (some versions)
            if timings is None and hasattr(self.llm, 'timings'):
                raw_timings = self.llm.timings
                if isinstance(raw_timings, dict) and raw_timings.get("predicted_n", 0) > 0:
                    timings = raw_timings
                elif hasattr(raw_timings, 'predicted_n') and raw_timings.predicted_n > 0:
                    timings = {
                        "prompt_n": getattr(raw_timings, 'prompt_n', 0),
                        "prompt_ms": getattr(raw_timings, 'prompt_ms', 0),
                        "prompt_per_second": getattr(raw_timings, 'prompt_per_second', 0),
                        "predicted_n": getattr(raw_timings, 'predicted_n', 0),
                        "predicted_ms": getattr(raw_timings, 'predicted_ms', 0),
                        "predicted_per_second": getattr(raw_timings, 'predicted_per_second', 0),
                    }
            
            if timings:
                metrics = PerfMetrics(
                    prompt_tokens=timings.get("prompt_n", 0),
                    completion_tokens=timings.get("predicted_n", token_count),
                    total_time_ms=timings.get("prompt_ms", 0) + timings.get("predicted_ms", 0),
                    prompt_eval_time_ms=timings.get("prompt_ms", 0),
                    completion_time_ms=timings.get("predicted_ms", 0),
                    tokens_per_second=timings.get("predicted_per_second", 0),
                    prompt_per_second=timings.get("prompt_per_second", 0),
                )
                
        except Exception:
            pass
        
        # Fallback to manual timing if native timings not available
        if metrics is None:
            total_time_ms = (end_time - start_time) * 1000
            prompt_eval_time_ms = ((first_token_time or end_time) - start_time) * 1000
            completion_time_ms = (end_time - (first_token_time or start_time)) * 1000
            tokens_per_second = (token_count / completion_time_ms * 1000) if completion_time_ms > 0 else 0
            prompt_tokens_est = len(prompt) // 4  # Rough estimate
            prompt_per_second = (prompt_tokens_est / prompt_eval_time_ms * 1000) if prompt_eval_time_ms > 0 else 0
            
            metrics = PerfMetrics(
                prompt_tokens=prompt_tokens_est,
                completion_tokens=token_count,
                total_time_ms=total_time_ms,
                prompt_eval_time_ms=prompt_eval_time_ms,
                completion_time_ms=completion_time_ms,
                tokens_per_second=tokens_per_second,
                prompt_per_second=prompt_per_second,
            )
        
        yield (None, metrics)
    
    def _format_chat_prompt(self, messages: list[dict]) -> str:
        """Format messages using Llama 3.3 chat template."""
        # Note: llama.cpp adds <|begin_of_text|> automatically, don't duplicate
        prompt = ""
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                prompt += f"<|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>"
            elif role == "user":
                prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>"
            elif role == "assistant":
                prompt += f"<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>"
        
        # Add assistant header for generation
        prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        
        return prompt
    
    @classmethod
    def list_quantizations(cls):
        """Print available quantization options."""
        print("\nAvailable Quantization Options:")
        print("=" * 70)
        print(f"{'Quant':<10} {'Size':<10} {'RAM Needed':<12} {'Quality'}")
        print("-" * 70)
        
        for name, info in sorted(cls.QUANT_OPTIONS.items(), key=lambda x: x[1]["size_gb"]):
            print(f"{name:<10} {info['size_gb']}GB{'':<6} {info['recommended_ram']:<12} {info['quality']}")
        
        print("-" * 70)
        print("\n✓ Recommended for 48GB RAM: Q4_K_M (best balance)")
        print("✓ Recommended for 16GB RAM: Q3_K_S or Q2_K")
        print("  These use mmap to stream model layers from disk as needed.\n")


# Singleton instance for the API server
_transformer_instance: Optional[LlamaTransformer] = None


def get_transformer(
    quantization: str = "Q4_K_M",
    model_path: Optional[str] = None,
    n_ctx: int = 2048,
    n_gpu_layers: int = -1,
    n_batch: int = 512,
) -> LlamaTransformer:
    """Get or create the transformer singleton (for API server use)."""
    global _transformer_instance
    
    if _transformer_instance is None:
        _transformer_instance = LlamaTransformer(
            model_path=model_path,
            quantization=quantization,
            n_ctx=n_ctx,
            n_batch=n_batch,
            n_gpu_layers=n_gpu_layers,
            use_mmap=True,
            use_mlock=False,
            flash_attn=True,
            offload_kqv=True,
        )
    
    return _transformer_instance


def main():
    """Example usage of LlamaTransformer."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run Llama 3.3 70B on Mac M4"
    )
    parser.add_argument(
        "--quant", "-q",
        default="Q4_K_M",
        choices=list(LlamaTransformer.QUANT_OPTIONS.keys()),
        help="Quantization level (default: Q4_K_M)"
    )
    parser.add_argument(
        "--ctx", "-c",
        type=int,
        default=2048,
        help="Context window size (default: 2048, use smaller for less RAM)"
    )
    parser.add_argument(
        "--model-path", "-m",
        type=str,
        default=None,
        help="Path to GGUF model file (downloads if not provided)"
    )
    parser.add_argument(
        "--list-quants",
        action="store_true",
        help="List available quantization options and exit"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive chat mode"
    )
    parser.add_argument(
        "--gpu-layers", "-g",
        type=int,
        default=-1,
        help="Number of layers to offload to GPU (-1 = all, reduce if OOM)"
    )
    
    args = parser.parse_args()
    
    if args.list_quants:
        LlamaTransformer.list_quantizations()
        return
    
    # Memory-efficient settings
    print("\n🦙 Initializing Llama 3.3 70B for Mac M4...\n")
    
    transformer = LlamaTransformer(
        model_path=args.model_path,
        quantization=args.quant,
        n_ctx=args.ctx,
        n_batch=256,
        n_gpu_layers=args.gpu_layers,  # Use Metal GPU (-1 = all layers)
        use_mmap=True,                 # Critical: stream from disk
        use_mlock=False,               # Don't lock in RAM
    )
    
    if args.interactive:
        # Interactive chat mode
        print("\n" + "=" * 60)
        print("Interactive Chat Mode (type 'quit' to exit)")
        print("=" * 60 + "\n")
        
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."}
        ]
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                if user_input.lower() in ["quit", "exit", "q"]:
                    print("\nGoodbye! 👋")
                    break
                if not user_input:
                    continue
                
                messages.append({"role": "user", "content": user_input})
                
                print("\nAssistant: ", end="", flush=True)
                
                # Stream response
                full_response = ""
                for token in transformer.chat(messages, stream=True):
                    print(token, end="", flush=True)
                    full_response += token
                
                print()  # Newline after response
                
                messages.append({"role": "assistant", "content": full_response})
                
            except KeyboardInterrupt:
                print("\n\nInterrupted. Goodbye! 👋")
                break
    else:
        # Single prompt demo
        print("\n--- Demo Generation ---\n")
        
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "Explain quantum computing in simple terms."}
        ]
        
        print("Prompt: Explain quantum computing in simple terms.\n")
        print("Response: ", end="", flush=True)
        
        for token in transformer.chat(messages, stream=True):
            print(token, end="", flush=True)
        
        print("\n")


if __name__ == "__main__":
    main()
