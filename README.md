# Llama 3.3 70B Chat

A full-stack chat application running **bartowski's GGUF quantized versions** of Llama 3.3 70B Instruct locally.

**Supported Platforms:**
- ✅ **macOS** with Apple Silicon (M1/M2/M3/M4) - Metal GPU acceleration
- ✅ **Windows** with NVIDIA GPUs (RTX 3060, 3070, 3080, 4060, 4070, 4080, 4090, 5060, etc.) - CUDA acceleration
- ✅ **Linux** with NVIDIA GPUs - CUDA acceleration

**GGUF Models**: [bartowski/Llama-3.3-70B-Instruct-GGUF](https://huggingface.co/bartowski/Llama-3.3-70B-Instruct-GGUF)

**Original Model**: [meta-llama/Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) (by Meta)

## 🚀 Quick Start (Full-Stack App)

### macOS / Linux
```bash
# One-time setup
./setup_fullstack.sh

# Start the application
./start.sh
```

### Windows (PowerShell)
```powershell
# One-time setup
.\setup_windows.bat

# Start the application
.\start_windows.bat
```

Then open **http://localhost:5173** in your browser!

### Manual Setup - macOS (Apple Silicon)

If `setup_fullstack.sh` has issues, set up manually:

```bash
# 1. Backend setup
cd backend
rm -rf venv                    # Remove old venv if exists
python3 -m venv venv           # Create fresh venv
source venv/bin/activate       # Activate venv
pip install --upgrade pip      # Upgrade pip

# Install llama-cpp-python with Metal support (takes 5-10 min)
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python --force-reinstall --no-cache-dir

# Install other dependencies
pip install -r requirements.txt

deactivate
cd ..

# 2. Frontend setup
cd frontend
npm install
cd ..

# 3. Start the app
./start.sh
```

### Manual Setup - Windows (NVIDIA GPU)

#### Prerequisites (Install in Order)

| # | Software | Download Link | Notes |
|---|----------|---------------|-------|
| 1 | **Python 3.10+** | [python.org/downloads](https://www.python.org/downloads/) | ✅ Check "Add Python to PATH" during install |
| 2 | **Node.js 18+** | [nodejs.org](https://nodejs.org/) | LTS version recommended |
| 3 | **NVIDIA Driver** | [nvidia.com/drivers](https://www.nvidia.com/Download/index.aspx) | Latest Game Ready or Studio driver |
| 4 | **CUDA Toolkit 12.x** | [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads) | Required for GPU acceleration |
| 5 | **Visual Studio Build Tools** | [visualstudio.microsoft.com/visual-cpp-build-tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) | Select "Desktop development with C++" |
| 6 | **CMake** | [cmake.org/download](https://cmake.org/download/) | Or: `winget install Kitware.CMake` |

> **⚠️ Important**: After installing Visual Studio Build Tools, run the setup from **"x64 Native Tools Command Prompt for VS 2022"** (search in Start Menu), NOT regular PowerShell!

#### Quick Setup (Automated)

```powershell
# Run from "x64 Native Tools Command Prompt for VS 2022"
.\setup_windows.bat
```

The setup script will:
1. Check all prerequisites
2. Try pre-built CUDA wheels first (no compilation needed!)
3. Fall back to building from source if needed

#### Manual Setup (Step by Step)

```powershell
# Open "x64 Native Tools Command Prompt for VS 2022" from Start Menu
cd path\to\local_ai_chat_app

# 1. Backend setup
cd backend
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install --upgrade pip

# Install llama-cpp-python with CUDA support
# Option A: Pre-built wheel (faster, try this first!)
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124

# Option B: If Option A fails, build from source
$env:CMAKE_ARGS="-DGGML_CUDA=on"
$env:FORCE_CMAKE="1"
pip install llama-cpp-python --force-reinstall --no-cache-dir

# Install other dependencies
pip install -r requirements.txt

deactivate
cd ..

# 2. Frontend setup
cd frontend
npm install
cd ..

# 3. Start the app
.\start_windows.bat
```

#### Troubleshooting Windows Installation

**Error: `pip install llama-cpp-python` fails**

1. **"cl.exe not found"** → Run from "x64 Native Tools Command Prompt", not regular PowerShell
2. **"CUDA not found"** → Add CUDA to PATH:
   ```powershell
   $env:PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin;" + $env:PATH
   ```
3. **"CMake not found"** → Install CMake and restart terminal

**Error: `nvcc not found`**
```powershell
# Add CUDA to system PATH (run as Administrator)
[Environment]::SetEnvironmentVariable("Path", $env:Path + ";C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin", "Machine")
# Restart your terminal
```

**Still not working?** Try CPU-only mode (slower but works):
```powershell
pip install llama-cpp-python  # No CUDA, uses CPU
# Set GPU_LAYERS=0 when running
```

### Manual Setup - Linux (NVIDIA GPU)

```bash
# 1. Backend setup
cd backend
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip

# Install llama-cpp-python with CUDA support
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir

# Install other dependencies
pip install -r requirements.txt

deactivate
cd ..

# 2. Frontend setup
cd frontend
npm install
cd ..

# 3. Start the app
./start.sh
```

## ✅ Verified Working Configuration

**Tested on Mac M4 Pro with 48GB RAM** - running perfectly!

```bash
MODEL_PATH=~/llama-models QUANT=IQ3_M CTX=2048 GPU_LAYERS=40 ./start.sh
```

| Parameter | Value | Description |
|-----------|-------|-------------|
| `MODEL_PATH` | `~/llama-models` | Directory containing GGUF models |
| `QUANT` | `IQ3_M` | 32GB model, medium quality I-quant |
| `CTX` | `1080` | Context window (tokens) |
| `GPU_LAYERS` | `40` | Layers offloaded to Metal GPU |

## 📁 Project Structure

```
local_ai_chat_app/
├── frontend/              # React chat UI
│   ├── src/
│   │   ├── App.jsx       # Main chat component
│   │   └── App.css       # Styles
│   └── package.json
│
├── backend/              # FastAPI server
│   ├── server.py         # REST API + SSE streaming
│   ├── llama_transformer.py  # LLM wrapper (internal module)
│   └── requirements.txt
│
├── brew_setup/           # Homebrew CLI alternative (macOS)
│
├── setup_fullstack.sh    # Setup script (macOS/Linux)
├── setup_windows.bat     # Setup script (Windows)
├── start.sh              # Start app (macOS/Linux)
├── start_windows.bat     # Start app (Windows)
└── README.md
```

---

## 🎯 How It Works

Running a 70B parameter model (normally ~140GB in FP16) efficiently requires:

1. **Quantization**: Compress to 1-8 bits per weight (from 16-bit)
2. **Memory Mapping (mmap)**: Stream model layers from disk on-demand
3. **GPU Acceleration**: Metal (Apple Silicon) or CUDA (NVIDIA GPUs)

### All Available Quantizations

Full list from [bartowski/Llama-3.3-70B-Instruct-GGUF](https://huggingface.co/bartowski/Llama-3.3-70B-Instruct-GGUF):

| Quantization | Size | RAM | Quality | Notes |
|--------------|------|-----|---------|-------|
| **1-bit** |||||
| IQ1_M | ~17GB | 10GB+ | Extreme | Lowest quality, smallest |
| **2-bit** |||||
| IQ2_XXS | ~19GB | 10GB+ | Extreme | Smallest 2-bit |
| IQ2_XS | ~21GB | 11GB+ | Extreme | Very aggressive |
| IQ2_S | ~22GB | 12GB+ | Low | I-quant |
| IQ2_M | ~24GB | 14GB+ | Low | I-quant |
| Q2_K | ~26GB | 14GB+ | Low | K-quant |
| Q2_K_L | ~27GB | 16GB+ | Low | Q8 embed/output |
| **3-bit** |||||
| IQ3_XXS | ~28GB | 16GB+ | Medium-low | I-quant |
| IQ3_XS | ~29GB | 16GB+ | Medium-low | I-quant |
| Q3_K_S | ~31GB | 18GB+ | Medium | K-quant small |
| IQ3_M | ~32GB | 18GB+ | Medium | I-quant |
| Q3_K_M | ~34GB | 20GB+ | Medium-good | K-quant medium |
| Q3_K_L | ~37GB | 22GB+ | Good | K-quant large |
| Q3_K_XL | ~38GB | 24GB+ | Good | Q8 embed/output |
| **4-bit** |||||
| IQ4_XS | ~38GB | 24GB+ | Good | I-quant 4-bit |
| Q4_0 | ~40GB | 24GB+ | Good | Legacy, ARM optimized |
| IQ4_NL | ~40GB | 24GB+ | Good | ARM optimized |
| Q4_K_S | ~40GB | 24GB+ | Good | K-quant small |
| Q4_0_4_4 | ~40GB | 24GB+ | Good | ARM NEON optimized |
| Q4_0_4_8 | ~40GB | 24GB+ | Good | ARM SVE 256 optimized |
| Q4_0_8_8 | ~40GB | 24GB+ | Good | AVX2/AVX512 optimized |
| **Q4_K_M** | **~43GB** | **32GB+** | **Very good** | **Recommended default** |
| Q4_K_L | ~43GB | 32GB+ | Excellent | Q8 embed/output |
| **5-bit** |||||
| Q5_K_S | ~49GB | 48GB+ | High | K-quant small |
| Q5_K_M | ~50GB | 48GB+ | High | K-quant medium, recommended |
| Q5_K_L | ~51GB | 48GB+ | High | Q8 embed/output |
| **6-bit** |||||
| Q6_K | ~58GB | 64GB+ | Very high | Near perfect |
| Q6_K_L | ~58GB | 64GB+ | Very high | Q8 embed/output |
| **8-bit** |||||
| Q8_0 | ~75GB | 80GB+ | Excellent | Max available quant |
| **16-bit** |||||
| F16 | ~141GB | 160GB+ | Perfect | Full precision |

> **💡 Tip**: For Apple Silicon Macs, use K-quants (Q4_K_M, Q5_K_M, etc.) for best performance. I-quants (IQ3_M, IQ4_XS) offer better quality at the same size but may be slower on Metal.

### 🎮 NVIDIA GPU Recommendations

For Windows/Linux with NVIDIA GPUs, the model is split between **VRAM** (fast) and **System RAM** (slower). More layers on GPU = faster inference.

| GPU | VRAM | Recommended Quant | GPU Layers | System RAM Needed |
|-----|------|-------------------|------------|-------------------|
| **RTX 3060** | 12GB | IQ2_XXS, IQ2_XS | 20-25 | 16GB+ |
| **RTX 3070** | 8GB | IQ1_M, IQ2_XXS | 15-20 | 24GB+ |
| **RTX 3070 Ti** | 8GB | IQ1_M, IQ2_XXS | 15-20 | 24GB+ |
| **RTX 3080** | 10GB | IQ2_XXS, IQ2_XS | 18-22 | 20GB+ |
| **RTX 3090** | 24GB | Q3_K_S, IQ3_M | 35-45 | 16GB+ |
| **RTX 4060** | 8GB | IQ1_M, IQ2_XXS | 15-20 | 24GB+ |
| **RTX 4060 Ti** | 8/16GB | IQ2_XXS / Q2_K | 15-25 | 20GB+ |
| **RTX 4070** | 12GB | IQ2_XS, Q2_K | 22-28 | 20GB+ |
| **RTX 4070 Ti** | 12GB | IQ2_XS, Q2_K | 22-28 | 20GB+ |
| **RTX 4080** | 16GB | Q2_K, Q3_K_S | 28-35 | 16GB+ |
| **RTX 4090** | 24GB | Q3_K_M, IQ3_M | 40-50 | 16GB+ |
| **RTX 5060** | 8GB* | IQ1_M, IQ2_XXS | 15-20 | 24GB+ |
| **RTX 5070** | 12GB* | IQ2_XS, Q2_K | 22-28 | 20GB+ |
| **RTX 5080** | 16GB* | Q2_K, Q3_K_S | 28-35 | 16GB+ |
| **RTX 5090** | 32GB* | Q4_K_S, Q4_K_M | 50-60 | 16GB+ |

\* RTX 50-series specs are estimated based on expected configurations.

#### Example Commands for NVIDIA GPUs

```powershell
# RTX 4060 (8GB VRAM) - Windows
$env:MODEL_PATH="C:\llama-models"
$env:QUANT="IQ2_XXS"
$env:CTX="1024"
$env:GPU_LAYERS="18"
.\start_windows.bat

# RTX 4090 (24GB VRAM) - Linux
MODEL_PATH=~/llama-models QUANT=Q3_K_M CTX=2048 GPU_LAYERS=45 ./start.sh

# RTX 3070 Ti (8GB VRAM) - Lower context for stability
MODEL_PATH=~/llama-models QUANT=IQ2_XXS CTX=512 GPU_LAYERS=15 ./start.sh
```

> **⚠️ VRAM vs RAM**: Unlike Apple Silicon's unified memory, NVIDIA GPUs have separate VRAM. If the model doesn't fit in VRAM, layers are offloaded to CPU (slower). Use `GPU_LAYERS` to control how many layers go to GPU.

> **💡 Performance Tip**: Start with fewer GPU layers and increase until you hit VRAM limits. Watch for "CUDA out of memory" errors.

## 🚀 Quick Start

### Installation

```bash
# Install llama-cpp-python with Metal support
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python --force-reinstall --no-cache-dir

# Install other dependencies
pip install huggingface_hub tqdm
```

### Basic Usage

```python
# From within backend/ directory
from llama_transformer import LlamaTransformer

# Initialize (defaults optimized for 48GB RAM)
transformer = LlamaTransformer(
    quantization="Q4_K_M",  # Best balance for 48GB RAM
    n_ctx=4096,             # Good context window
    use_mmap=True,          # Stream from disk
)

# Chat completion
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain quantum computing simply."}
]

# Stream response
for token in transformer.chat(messages, stream=True):
    print(token, end="", flush=True)
```

### Command Line (Backend)

```bash
cd backend

# List available quantizations
python llama_transformer.py --list-quants

# Interactive chat mode (recommended for 48GB RAM)
python llama_transformer.py -q Q4_K_M -i

# For 16GB RAM, use smaller quantization
python llama_transformer.py -q Q3_K_S -c 2048 -i

# Use specific model file
python llama_transformer.py -m /path/to/model.gguf -i
```

## 📊 Memory Optimization Tips

### For 48GB RAM (Recommended)

```python
transformer = LlamaTransformer(
    quantization="Q4_K_M",  # Best quality/size balance
    n_ctx=4096,             # Good context window
    n_batch=512,            # Fast processing
    use_mmap=True,          # Enable for safety
)
```

### For 16GB RAM

```python
transformer = LlamaTransformer(
    quantization="Q3_K_S",  # Fits comfortably
    n_ctx=2048,             # Smaller context
    n_batch=256,            # Moderate batches
    use_mmap=True,          # Critical: stream from disk
    use_mlock=False,        # Don't lock in RAM
)
```

### System Preparation

1. **First run downloads model**: ~43GB for Q4_K_M, takes 10-30 minutes
2. **Subsequent runs are fast**: Model loads from cache

```bash
# Check current memory
vm_stat | head -5

# Monitor during inference
# Open Activity Monitor → Memory tab
```

## 🔧 Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    User Prompt                          │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│              Llama 3.3 Chat Template                    │
│  <|begin_of_text|><|start_header_id|>system...         │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                 llama.cpp Engine                        │
│  ┌─────────────────────────────────────────────────┐   │
│  │  GGUF Model (Q3_K_S quantized)                  │   │
│  │  - 80 transformer layers                        │   │
│  │  - Memory mapped from SSD                       │   │
│  │  - Only active layers in RAM                    │   │
│  └─────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Metal GPU Backend                              │   │
│  │  - Matrix operations on GPU                     │   │
│  │  - Unified memory architecture                  │   │
│  └─────────────────────────────────────────────────┘   │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│              Streamed Token Output                      │
└─────────────────────────────────────────────────────────┘
```

## 📁 Backend Files

| File | Description |
|------|-------------|
| `backend/llama_transformer.py` | Main transformer using llama.cpp (recommended) |
| `backend/mlx_transformer.py` | Alternative using Apple MLX (needs 32GB+) |
| `backend/server.py` | FastAPI server with SSE streaming |
| `backend/requirements.txt` | Python dependencies |

## ⚡ Performance Expectations

On Mac M4 with 48GB RAM:

| Metric | Q4_K_M | Q5_K_S |
|--------|--------|--------|
| Load time | 60-90s | 90-120s |
| Tokens/sec | 8-15 | 6-12 |
| Quality | Very Good | High |
| RAM usage | ~43GB | ~49GB |

*First run includes download time (~43-50GB)*

## 📈 Understanding Performance Metrics

The UI displays real-time metrics after each response. Here's what they mean:

```
⚡ 2.47 tok/s | 📝 9 tokens | ⏱️ 10.6s total
prompt: 39 tok @ 5.61 tok/s (7.0s) | gen: 9 tok @ 2.47 tok/s (3.6s)
```

### Prompt Processing (prompt eval)

```
prompt: 39 tok @ 5.61 tok/s (7.0s)
```

| Metric | Meaning |
|--------|---------|
| **39 tokens** | Your input (system prompt + user message) converted to tokens |
| **5.61 tok/s** | Speed at which the model *reads and understands* your input |
| **7.0s** | Total time to process the prompt |

This is the **"thinking" phase** where the model:
1. Tokenizes your text ("Hello" → `[15496]`)
2. Runs each token through all 80 transformer layers
3. Builds internal context/attention for generating a response

### Generation (eval)

```
gen: 9 tok @ 2.47 tok/s (3.6s)
```

| Metric | Meaning |
|--------|---------|
| **9 tokens** | Number of tokens the model *generated* in its response |
| **2.47 tok/s** | Speed at which the model *writes* new tokens |
| **3.6s** | Total time to generate the response |

This is the **"writing" phase** where the model:
1. Predicts the next token based on context
2. Appends it to the output
3. Repeats until done (stop token or max_tokens)

### Why is generation slower than prompt processing?

| Phase | Speed | Reason |
|-------|-------|--------|
| **Prompt** | ~5-6 tok/s | Can process tokens in **parallel** (batch processing) |
| **Generation** | ~2-3 tok/s | Must generate tokens **one at a time** (sequential) |

Generation is inherently sequential—each new token depends on all previous tokens, so the model can't parallelize this phase.

### Visual Timeline

```
[0s]────────────────────[7.0s]──────────────[10.6s]
        Prompt (39 tok)         Gen (9 tok)
        "Understanding"         "Writing"
```

## 🔍 Troubleshooting

### "Out of memory" errors

```python
# Use more aggressive settings
transformer = LlamaTransformer(
    quantization="IQ2_XS",  # Smallest model
    n_ctx=512,              # Minimal context
)
```

### Slow generation

- Close background apps
- Ensure Metal is being used (check verbose output)
- Use smaller batch size: `n_batch=128`

### Model download fails

```bash
# Manual download with huggingface-cli
pip install huggingface_hub
huggingface-cli download bartowski/Llama-3.3-70B-Instruct-GGUF \
    Llama-3.3-70B-Instruct-Q3_K_S.gguf

# Then specify path when running
cd backend
python llama_transformer.py -m ~/.cache/huggingface/hub/.../Llama-3.3-70B-Instruct-Q3_K_S.gguf
```

## 📝 Credits & License

### GGUF Quantizations
The quantized GGUF models used in this project are provided by **[bartowski](https://huggingface.co/bartowski)**. These quantizations make it possible to run the 70B model on consumer hardware.

### Original Model
The base Llama 3.3 70B Instruct model is created by **Meta** and subject to the [Llama 3.3 Community License](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct).

Key requirements:
- Accept the license on Hugging Face before use
- Display "Built with Llama" for public applications
- Monthly active users > 700M require separate license from Meta

### This Code
This application code is provided for educational purposes.
