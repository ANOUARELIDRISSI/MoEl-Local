# 🚀 MoEl - High-Performance Local LLM Platform

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green.svg)
![Flask](https://img.shields.io/badge/Flask-Frontend-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**Run powerful AI models locally with a beautiful web interface**

[Quick Start](#-quick-start) • [Models](#-recommended-models) • [Configuration](#%EF%B8%8F-configuration) • [API](#-api-reference)

</div>

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎯 **Single File Analysis** | Process individual files with custom prompts |
| 📦 **Batch Processing** | Upload and process multiple files simultaneously |
| 🤖 **Task Types** | Code Generation, Code Review, Translation, Summarization |
| ⚡ **CPU Optimized** | Fast inference even without GPU |
| 🎮 **GPU Acceleration** | NVIDIA CUDA, AMD ROCm, Apple MPS support |
| 🧠 **NPU Support** | Intel/Qualcomm NPU acceleration |
| 🌐 **Web Interface** | Modern, responsive UI at `localhost:5000` |
| 🔌 **REST API** | Programmatic access at `localhost:8000` |

---

## 📋 System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **OS** | Windows 10, Ubuntu 20.04, macOS 10.15 | Latest versions |
| **Python** | 3.8 | 3.10+ |
| **RAM** | 4GB | 8GB+ |
| **Storage** | 2GB | 5GB+ |
| **CPU** | 2 cores | 4+ cores |

---

## 🎯 Recommended Models

### 💻 CPU (No GPU Required)

| Model | Size | Speed | Quality | Best For |
|-------|------|-------|---------|----------|
| **`Qwen/Qwen2.5-0.5B-Instruct`** ⭐ | 1GB | ⚡⚡⚡ Fast | ★★★☆☆ | Default, general use |
| `Qwen/Qwen2.5-1.5B-Instruct` | 3GB | ⚡⚡ Medium | ★★★★☆ | Better quality |
| `microsoft/phi-2` | 5GB | ⚡ Slow | ★★★★★ | Best quality on CPU |
| `gpt2` | 500MB | ⚡⚡⚡⚡ Very Fast | ★★☆☆☆ | Testing only |

### 🎮 GPU (NVIDIA CUDA / AMD ROCm)

| Model | VRAM | Speed | Quality | Best For |
|-------|------|-------|---------|----------|
| **`Qwen/Qwen2.5-1.5B-Instruct`** ⭐ | 4GB | ⚡⚡⚡⚡ | ★★★★☆ | Default GPU choice |
| `Qwen/Qwen2.5-3B-Instruct` | 6GB | ⚡⚡⚡ | ★★★★★ | High quality |
| `Qwen/Qwen2.5-7B-Instruct` | 8GB | ⚡⚡ | ★★★★★ | Professional use |
| `microsoft/phi-2` | 4GB | ⚡⚡⚡⚡ | ★★★★★ | Excellent balance |
| `mistralai/Mistral-7B-Instruct-v0.2` | 8GB | ⚡⚡ | ★★★★★ | Top quality |

### 🍎 Apple Silicon (MPS)

| Model | Memory | Speed | Quality | Best For |
|-------|--------|-------|---------|----------|
| **`Qwen/Qwen2.5-1.5B-Instruct`** ⭐ | 4GB | ⚡⚡⚡⚡ | ★★★★☆ | M1/M2/M3 default |
| `Qwen/Qwen2.5-3B-Instruct` | 6GB | ⚡⚡⚡ | ★★★★★ | M1 Pro/Max |
| `microsoft/phi-2` | 4GB | ⚡⚡⚡⚡ | ★★★★★ | All Apple Silicon |

### 🧠 NPU (Intel/Qualcomm Neural Processors)

| Model | Memory | Speed | Quality | Best For |
|-------|--------|-------|---------|----------|
| **`Qwen/Qwen2.5-0.5B-Instruct`** ⭐ | 1GB | ⚡⚡⚡⚡⚡ | ★★★☆☆ | Intel Core Ultra |
| `Qwen/Qwen2.5-1.5B-Instruct` | 3GB | ⚡⚡⚡⚡ | ★★★★☆ | Snapdragon X Elite |
| `microsoft/phi-2` | 5GB | ⚡⚡⚡ | ★★★★★ | NPU with 8GB+ RAM |

> 💡 **Tip**: Start with `Qwen/Qwen2.5-0.5B-Instruct` - it's fast, accurate, and works on any hardware!

---

## 🚀 Quick Start

### Windows (PowerShell)

```powershell
# 1. Clone the repository
git clone https://github.com/ANOUARELIDRISSI/MoEl-Local.git
cd MoEl-Local

# 2. Run setup (installs dependencies + downloads model)
.\run.ps1 setup

# 3. Start the application
.\run.ps1 start

# 4. Open http://localhost:5000 in your browser
```

### Linux / macOS (Bash)

```bash
# 1. Clone the repository
git clone https://github.com/ANOUARELIDRISSI/MoEl-Local.git
cd MoEl-Local

# 2. Make script executable
chmod +x run.sh

# 3. Run setup
./run.sh setup

# 4. Start the application
./run.sh start

# 5. Open http://localhost:5000 in your browser
```

### Cross-Platform (Python)

```bash
# Works on any system with Python 3.8+
python setup.py setup
python setup.py start
```

---

## 🎮 Commands Reference

| Action | Windows | Linux/Mac | Python |
|--------|---------|-----------|--------|
| **Setup** | `.\run.ps1 setup` | `./run.sh setup` | `python setup.py setup` |
| **Start** | `.\run.ps1 start` | `./run.sh start` | `python setup.py start` |
| **Stop** | `.\run.ps1 stop` | `./run.sh stop` | `python setup.py stop` |
| **Restart** | `.\run.ps1 restart` | `./run.sh restart` | `python setup.py restart` |
| **Status** | `.\run.ps1 status` | `./run.sh status` | `python setup.py status` |
| **Logs** | `.\run.ps1 logs` | `./run.sh logs` | - |

---

## ⚙️ Configuration

### Hardware Configuration (`config/hardware.json`)

```json
{
  "device": "cpu",
  "gpu_type": "none",
  "gpu_layers": 0,
  "threads": 4,
  "context_length": 2048,
  "model_name": "Qwen/Qwen2.5-0.5B-Instruct"
}
```

### Configuration Options

| Option | Values | Description |
|--------|--------|-------------|
| `device` | `cpu`, `cuda`, `mps`, `npu` | Compute device |
| `gpu_type` | `none`, `nvidia`, `amd`, `mps`, `intel_npu` | Hardware type |
| `gpu_layers` | `0` - `35` | Layers offloaded to accelerator |
| `threads` | `1` - `16` | CPU threads for inference |
| `context_length` | `512` - `8192` | Max context window |
| `model_name` | HuggingFace ID | Model identifier |

### Device Configuration Examples

<details>
<summary><b>💻 CPU Only</b></summary>

```json
{
  "device": "cpu",
  "gpu_type": "none",
  "gpu_layers": 0,
  "threads": 8,
  "context_length": 2048,
  "model_name": "Qwen/Qwen2.5-0.5B-Instruct"
}
```
</details>

<details>
<summary><b>🎮 NVIDIA GPU</b></summary>

```json
{
  "device": "cuda",
  "gpu_type": "nvidia",
  "gpu_layers": 35,
  "threads": 4,
  "context_length": 4096,
  "model_name": "Qwen/Qwen2.5-1.5B-Instruct"
}
```
</details>

<details>
<summary><b>🍎 Apple Silicon</b></summary>

```json
{
  "device": "mps",
  "gpu_type": "mps",
  "gpu_layers": 35,
  "threads": 8,
  "context_length": 4096,
  "model_name": "Qwen/Qwen2.5-1.5B-Instruct"
}
```
</details>

<details>
<summary><b>🧠 Intel NPU (Core Ultra)</b></summary>

```json
{
  "device": "npu",
  "gpu_type": "intel_npu",
  "gpu_layers": 0,
  "threads": 8,
  "context_length": 2048,
  "model_name": "Qwen/Qwen2.5-0.5B-Instruct"
}
```
</details>

---

## 🔌 API Reference

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check & system info |
| `/inference` | POST | Single prompt inference |
| `/batch-inference` | POST | Batch processing |
| `/upload-and-process` | POST | File upload & processing |

### Health Check

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cpu",
  "model_name": "Qwen/Qwen2.5-0.5B-Instruct"
}
```

### Single Inference

```bash
curl -X POST http://localhost:8000/inference \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Write a Python function to calculate fibonacci",
    "max_tokens": 512,
    "temperature": 0.7,
    "task_type": "code_gen"
  }'
```

### Batch Inference

```bash
curl -X POST http://localhost:8000/batch-inference \
  -H "Content-Type: application/json" \
  -d '{
    "prompts": ["Explain Python lists", "What is recursion?"],
    "max_tokens": 256,
    "temperature": 0.7
  }'
```

---

## 📁 Project Structure

```
moel/
├── backend/
│   └── app.py              # FastAPI backend server
├── frontend/
│   ├── app.py              # Flask frontend server
│   └── templates/
│       └── index.html      # Web interface
├── config/
│   ├── hardware.json       # Hardware configuration (local)
│   └── hardware.json.example
├── logs/                   # Application logs
├── uploads/                # Uploaded files
├── outputs/                # Generated outputs
├── run.ps1                 # Windows launcher
├── run.sh                  # Linux/Mac launcher
├── setup.py                # Cross-platform launcher
└── requirements.txt        # Python dependencies
```

---

## 🔧 Troubleshooting

<details>
<summary><b>Port Already in Use</b></summary>

```powershell
# Windows
Get-NetTCPConnection -LocalPort 8000 | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }
```

```bash
# Linux/Mac
lsof -ti:8000 | xargs kill -9
```
</details>

<details>
<summary><b>Model Loading Slow</b></summary>

First startup downloads the model (~1GB). Subsequent starts use the cached model from `~/.cache/huggingface/`.
</details>

<details>
<summary><b>Out of Memory</b></summary>

- Use a smaller model (e.g., `gpt2` or `Qwen2.5-0.5B`)
- Reduce `context_length` in config
- Close other applications
</details>

<details>
<summary><b>Check Logs</b></summary>

```bash
# Windows
Get-Content .\logs\backend_*.log -Tail 50

# Linux/Mac
tail -50 logs/backend.log
```
</details>

---

## 📄 License

MIT License - feel free to use, modify, and distribute.

---

<div align="center">

**Made with ❤️ for local AI inference**

[⬆ Back to Top](#-moel---high-performance-local-llm-platform)

</div>
