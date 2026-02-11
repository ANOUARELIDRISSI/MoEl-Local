# 🚀 MoEl - High-Performance Local LLM Platform

**MoEl** (Model Engine Local) is a production-ready, high-performance application for interacting with local Large Language Models (LLMs). It features intelligent hardware optimization, batch processing capabilities, and a modern web interface.

## ✨ Key Features

### Core Capabilities
- **🎯 Single File Analysis**: Process individual text files with custom prompts
- **📦 Batch Processing**: Upload and process multiple files simultaneously
- **🤖 Task-Specific Execution**: 
  - Code Generation
  - Code Review
  - Text Translation
  - Summarization
  - General Analysis

### Performance & Optimization
- **⚡ CPU-Optimized by Default**: Efficient inference on any hardware
- **🎮 GPU Acceleration**: Automatic NVIDIA/AMD/Apple Silicon detection
- **🔧 Dynamic Configuration**: Choose hardware at startup
- **🚄 Parallel Processing**: Multi-threaded batch operations
- **💾 Memory Efficient**: Optimized for models up to 20B parameters

### Architecture
- **FastAPI Backend**: High-performance async API server
- **Flask Frontend**: Modern, responsive web interface
- **Bash Automation**: Zero-dependency setup and management
- **No Docker Required**: Pure bash-based deployment
- **Production Ready**: Comprehensive logging and error handling

---

## 📋 System Requirements

### Minimum Requirements
- **OS**: Linux (Ubuntu 20.04+), macOS (10.15+), or Windows (WSL2)
- **Python**: 3.8 or higher
- **RAM**: 8GB (16GB recommended for larger models)
- **Storage**: 10GB free space
- **CPU**: 4+ cores recommended

### Recommended for GPU Acceleration
- **NVIDIA GPU**: 6GB+ VRAM, CUDA 11.8+
- **AMD GPU**: ROCm compatible
- **Apple Silicon**: M1/M2/M3 chips (MPS acceleration)

---

## 🚀 Quick Start

### 1. Clone or Download MoEl
```bash
# If using git
git clone <repository-url>
cd moel

# Or extract the downloaded archive
cd moel
```

### 2. Run Setup
```bash
./run.sh setup
```

**What happens during setup:**
1. Detects your OS (Linux/macOS/Windows)
2. Detects available hardware (CPU/GPU)
3. Prompts for GPU acceleration preference
4. Asks you to select an LLM model
5. Creates Python virtual environment
6. Installs all dependencies automatically
7. Generates `requirements.txt`
8. Configures hardware settings

**Model Selection Options:**
- **GPT-2** (small): ~500MB - Fast, good for testing
- **GPT-2 Medium**: ~1.5GB - Better quality
- **GPT-2 Large**: ~3GB - High quality
- **GPT-2 XL**: ~6GB - Best GPT-2 quality
- **Custom Model**: Enter any HuggingFace model name

### 3. Start MoEl
```bash
./run.sh start
```

The application will:
- Start the backend API server on port 8000
- Start the frontend web interface on port 5000
- Display access URLs and status

### 4. Access the Web Interface
Open your browser and navigate to:
```
http://localhost:5000
```

---

## 📖 Usage Guide

### Web Interface

#### Single File Inference
1. Select a **Task Type** (General, Code Gen, Code Review, Translation, Summarization)
2. Enter your **Prompt**
3. Adjust **Max Tokens** and **Temperature** if needed
4. Click **Generate**
5. View results in the output panel

#### Batch Processing
1. Select a **Task Type**
2. (Optional) Enter a **Default Prompt Template** using `{content}` placeholder
   - Example: `"Please review this code: {content}"`
3. Click **Upload Files** and select multiple files
4. Adjust generation parameters
5. Click **Process Files**
6. View individual results for each file

#### Supported File Types
`.txt`, `.md`, `.py`, `.js`, `.java`, `.cpp`, `.c`, `.go`, `.rs`, `.json`, `.xml`, `.html`, `.css`

---

### File Format for Single Analysis

If you want to include a specific prompt within your file, use this format:

```text
### PROMPT ###
Your analysis question or task here
### END PROMPT ###

[Rest of your file content]
```

If no prompt markers are found, the entire file content is used as the prompt.

---

### API Endpoints

MoEl provides a RESTful API for programmatic access:

#### Health Check
```bash
GET http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda",
  "cuda_available": true,
  "gpu_name": "NVIDIA GeForce RTX 3090"
}
```

#### Single Inference
```bash
POST http://localhost:8000/inference
Content-Type: application/json

{
  "prompt": "Write a Python function to calculate fibonacci numbers",
  "max_tokens": 512,
  "temperature": 0.7,
  "task_type": "code_gen"
}
```

#### Batch Inference
```bash
POST http://localhost:8000/batch-inference
Content-Type: application/json

{
  "prompts": ["Prompt 1", "Prompt 2", "Prompt 3"],
  "default_prompt_template": "Analyze: {content}",
  "max_tokens": 512,
  "temperature": 0.7,
  "task_type": "general"
}
```

#### Upload and Process Files
```bash
POST http://localhost:8000/upload-and-process
Content-Type: multipart/form-data

files: [file1.txt, file2.py, ...]
task_type: code_review
max_tokens: 512
temperature: 0.7
default_prompt: "Review this code: {content}"
```

---

## 🛠️ Management Commands

### Start Application
```bash
./run.sh start
```
Starts both backend and frontend services.

### Stop Application
```bash
./run.sh stop
```
Gracefully stops all services.

### Restart Application
```bash
./run.sh restart
```
Stops and starts services.

### Check Status
```bash
./run.sh status
```
Shows whether services are running.

### View Logs
```bash
./run.sh logs
```
Live tail of backend and frontend logs.

### Manual Log Viewing
```bash
# Backend logs
tail -f logs/backend_YYYYMMDD.log

# Frontend logs
tail -f logs/frontend_YYYYMMDD.log
```

---

## ⚙️ Configuration

### Hardware Configuration

Configuration is stored in `config/hardware.json`:

```json
{
  "device": "cuda",
  "gpu_type": "nvidia",
  "gpu_layers": 35,
  "threads": 8,
  "context_length": 2048,
  "model_name": "gpt2"
}
```

**Parameters:**
- `device`: "cpu", "cuda", or "mps"
- `gpu_type`: "nvidia", "amd", "mps", or "none"
- `gpu_layers`: Number of layers to offload to GPU (0-35)
- `threads`: CPU threads for inference
- `context_length`: Maximum context window (tokens)
- `model_name`: HuggingFace model identifier

### Reconfiguring Hardware

To change hardware settings:
1. Stop the application: `./run.sh stop`
2. Run setup again: `./run.sh setup`
3. Or manually edit `config/hardware.json`
4. Restart: `./run.sh start`

---

## 📁 Project Structure

```
moel/
├── run.sh                 # Main launcher script
├── requirements.txt       # Python dependencies (auto-generated)
├── README.md             # This file
│
├── backend/              # FastAPI backend
│   └── app.py           # Main API server
│
├── frontend/            # Flask frontend
│   ├── app.py          # Flask application
│   ├── templates/
│   │   └── index.html  # Web interface
│   └── static/         # Static assets (if any)
│
├── config/              # Configuration files
│   └── hardware.json   # Hardware settings
│
├── logs/                # Application logs
│   ├── backend_*.log
│   └── frontend_*.log
│
├── uploads/             # Uploaded files storage
├── outputs/             # Processing results
├── models/              # Downloaded models cache
└── venv/                # Python virtual environment
```

---

## 🔧 Advanced Usage

### Using Custom Models

MoEl supports any HuggingFace Transformers-compatible model:

1. During setup, choose option 5 (Custom model)
2. Enter the model identifier, for example:
   - `facebook/opt-1.3b`
   - `EleutherAI/gpt-neo-2.7B`
   - `microsoft/phi-2`
   - `mistralai/Mistral-7B-v0.1`

**Note:** Larger models require more RAM/VRAM:
- 7B models: ~14GB RAM or ~8GB VRAM
- 13B models: ~26GB RAM or ~16GB VRAM
- 20B models: ~40GB RAM or ~24GB VRAM

### Task-Specific Prompting

MoEl automatically enhances prompts based on task type:

**Code Generation:**
```
Enhanced format:
### Task: Generate clean, efficient code
### Request: {your_prompt}
### Code:
```

**Code Review:**
```
Enhanced format:
### Task: Review the following code and provide feedback
### Code: {your_code}
### Review:
```

**Translation:**
```
Enhanced format:
### Task: Translate the following text
### Text: {your_text}
### Translation:
```

### Batch Processing with Templates

Use `{content}` as a placeholder in your default prompt:

```
Template: "Translate the following to Spanish: {content}"

File 1: "Hello, how are you?"
File 2: "Good morning"
File 3: "Thank you very much"

Results:
1. "Hola, ¿cómo estás?"
2. "Buenos días"
3. "Muchas gracias"
```

### Performance Tuning

**For Faster Inference:**
- Use smaller models (GPT-2, GPT-2 Medium)
- Reduce `max_tokens` to 256-512
- Lower `temperature` to 0.3-0.5
- Enable GPU acceleration

**For Better Quality:**
- Use larger models (GPT-2 Large/XL or custom models)
- Increase `max_tokens` to 1024-2048
- Adjust `temperature` to 0.7-0.9
- Provide detailed, specific prompts

### Python API Integration

Use MoEl programmatically:

```python
import requests

# Single inference
response = requests.post('http://localhost:8000/inference', json={
    'prompt': 'Explain quantum computing',
    'max_tokens': 512,
    'temperature': 0.7,
    'task_type': 'general'
})

result = response.json()
print(result['result'])
```

```python
# Batch processing
with open('file1.txt', 'rb') as f1, open('file2.txt', 'rb') as f2:
    files = [
        ('files', ('file1.txt', f1, 'text/plain')),
        ('files', ('file2.txt', f2, 'text/plain'))
    ]
    
    data = {
        'task_type': 'summarize',
        'max_tokens': 256,
        'temperature': 0.5
    }
    
    response = requests.post(
        'http://localhost:8000/upload-and-process',
        files=files,
        data=data
    )
    
    results = response.json()
    for item in results['results']:
        print(f"{item['filename']}: {item['result']}")
```

---

## 🐛 Troubleshooting

### Backend Won't Start

**Check logs:**
```bash
cat logs/backend_*.log
```

**Common issues:**
- Model download failed: Check internet connection
- CUDA errors: Verify GPU drivers are installed
- Port 8000 in use: Stop other services or change port in `backend/app.py`

### Frontend Won't Connect to Backend

**Verify backend is running:**
```bash
curl http://localhost:8000/health
```

**Check BACKEND_URL:**
Frontend expects backend at `http://localhost:8000`. If changed, update `frontend/app.py`.

### Out of Memory Errors

**Solutions:**
- Use a smaller model
- Switch to CPU mode
- Reduce `context_length` in `config/hardware.json`
- Lower batch size
- Close other applications

### GPU Not Detected

**NVIDIA:**
```bash
nvidia-smi
```
If this fails, install/update NVIDIA drivers and CUDA toolkit.

**Apple Silicon:**
```bash
uname -m  # Should show arm64
```

### Slow Inference

**Optimizations:**
- Enable GPU acceleration
- Reduce `max_tokens`
- Use smaller model
- Increase `threads` in config for CPU inference

### Permission Denied

```bash
chmod +x run.sh
sudo chown -R $USER:$USER moel/
```

---

## 🔒 Security Considerations

### Production Deployment

MoEl is designed for **local use**. For production deployment:

1. **Add Authentication**: Implement API keys or OAuth
2. **Enable HTTPS**: Use reverse proxy (nginx/Apache) with SSL
3. **Rate Limiting**: Prevent abuse with request throttling
4. **Input Validation**: Sanitize all user inputs
5. **Firewall**: Restrict access to ports 5000 and 8000
6. **File Upload Limits**: Already set to 50MB, adjust as needed
7. **Logging**: Monitor logs for suspicious activity

### File Upload Safety

- Only allowed extensions can be uploaded
- Files are sanitized with `secure_filename()`
- Temporary files are stored in isolated `uploads/` directory
- Results saved to `outputs/` directory

---

## 📊 Performance Benchmarks

Approximate inference speeds (depends on hardware):

| Model | Hardware | Tokens/sec | Time for 512 tokens |
|-------|----------|------------|---------------------|
| GPT-2 | CPU (8 cores) | 15-25 | ~25 seconds |
| GPT-2 | RTX 3090 | 100-150 | ~4 seconds |
| GPT-2 Medium | CPU | 8-15 | ~40 seconds |
| GPT-2 Medium | RTX 3090 | 60-90 | ~7 seconds |
| GPT-2 Large | CPU | 4-8 | ~80 seconds |
| GPT-2 Large | RTX 3090 | 35-50 | ~12 seconds |

*Note: Actual performance varies based on CPU/GPU model, temperature, and prompt complexity.*

---

## 🔄 Updating Dependencies

To update all Python packages:

```bash
source venv/bin/activate
pip install --upgrade pip
pip install --upgrade transformers torch accelerate fastapi uvicorn flask
pip freeze > requirements.txt
deactivate
```

---

## 📝 Example Use Cases

### 1. Code Review Automation
Upload multiple Python files for automated code review with suggestions.

### 2. Documentation Translation
Batch translate documentation files to multiple languages.

### 3. Code Generation Pipeline
Generate boilerplate code, tests, and documentation from specifications.

### 4. Text Summarization
Summarize multiple research papers or articles simultaneously.

### 5. Data Analysis Reports
Process CSV data files and generate analysis reports.

---

## 🤝 Contributing

MoEl is designed to be extensible. To add new features:

1. **Backend** (`backend/app.py`): Add new FastAPI endpoints
2. **Frontend** (`frontend/app.py` & `templates/index.html`): Add UI components
3. **Configuration**: Extend `config/hardware.json` schema
4. **Tasks**: Add new task types in `LLMEngine._enhance_prompt()`

---

## 📄 License

This project is open-source. Check the LICENSE file for details.

---

## 🙏 Acknowledgments

Built with:
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [Transformers](https://huggingface.co/transformers/) - LLM library
- [FastAPI](https://fastapi.tiangolo.com/) - Modern async API framework
- [Flask](https://flask.palletsprojects.com/) - Web framework
- [HuggingFace](https://huggingface.co/) - Model hub

---

## 📞 Support

For issues, questions, or feature requests:

1. Check the **Troubleshooting** section above
2. Review logs in `logs/` directory
3. Verify configuration in `config/hardware.json`
4. Test API endpoints with `curl` or Postman
5. Check system resources (RAM, disk space)

---

## 🎯 Roadmap

Future enhancements:
- [ ] Multi-modal support (image + text)
- [ ] Streaming responses
- [ ] Model quantization (4-bit, 8-bit)
- [ ] Distributed inference
- [ ] Web-based configuration UI
- [ ] Docker support (optional)
- [ ] RESTful API documentation (Swagger)
- [ ] WebSocket support for real-time inference
- [ ] Plugin system for custom tasks

---

**Made with ❤️ for the AI community**

*Last updated: 2026-02-10*
#   M o E l - L o c a l  
 