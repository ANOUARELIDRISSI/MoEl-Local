# MoEl - Setup & Running Guide

A high-performance local LLM platform with web interface.

## Quick Start

### Windows (PowerShell)

```powershell
# 1. Setup (install dependencies + download model)
.\run.ps1 setup

# 2. Start the application
.\run.ps1 start

# 3. Open browser at http://localhost:5000
```

### Linux / macOS (Bash)

```bash
# 1. Make script executable
chmod +x run.sh

# 2. Setup (install dependencies + download model)
./run.sh setup

# 3. Start the application
./run.sh start

# 4. Open browser at http://localhost:5000
```

### Cross-Platform (Python)

Works on any system with Python 3.8+:

```bash
# 1. Setup
python setup.py setup

# 2. Start
python setup.py start

# 3. Open browser at http://localhost:5000
```

## Available Commands

| Command | Windows | Linux/Mac | Python |
|---------|---------|-----------|--------|
| **Setup** | `.\run.ps1 setup` | `./run.sh setup` | `python setup.py setup` |
| **Start** | `.\run.ps1 start` | `./run.sh start` | `python setup.py start` |
| **Stop** | `.\run.ps1 stop` | `./run.sh stop` | `python setup.py stop` |
| **Restart** | `.\run.ps1 restart` | `./run.sh restart` | `python setup.py restart` |
| **Status** | `.\run.ps1 status` | `./run.sh status` | `python setup.py status` |
| **View Logs** | `.\run.ps1 logs` | `./run.sh logs` | - |

## System Requirements

- **Python**: 3.8 or higher
- **RAM**: 4GB minimum (8GB recommended)
- **Storage**: 2GB free space (for model + dependencies)
- **OS**: Windows 10+, Linux (Ubuntu 20.04+), macOS 10.15+

## Default Model

**Qwen/Qwen2.5-0.5B-Instruct**
- Size: ~1GB download
- Parameters: 500M
- Features: Instruction-tuned, fast CPU inference
- Speed: 2-4 tokens/sec on CPU

## Ports Used

| Service | Port | URL |
|---------|------|-----|
| Backend API | 8000 | http://localhost:8000 |
| Web Interface | 5000 | http://localhost:5000 |

## Project Structure

```
moel/
├── backend/           # FastAPI backend
│   └── app.py
├── frontend/          # Flask web interface
│   ├── app.py
│   └── templates/
├── config/            # Configuration files
├── logs/              # Log files (auto-created)
├── uploads/           # File uploads
├── outputs/           # Generated outputs
├── run.ps1            # Windows launcher
├── run.sh             # Linux/Mac launcher
├── setup.py           # Cross-platform launcher
└── requirements.txt   # Python dependencies
```

## Troubleshooting

### Port already in use
```powershell
# Windows - Kill process on port 8000
Get-NetTCPConnection -LocalPort 8000 | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }
```

```bash
# Linux/Mac - Kill process on port 8000
lsof -ti:8000 | xargs kill -9
```

### Model loading slowly
The first startup downloads the model (~1GB). Subsequent starts are faster as the model is cached.

### Backend not responding
Check logs:
```bash
# Windows
Get-Content .\logs\backend_*.log -Tail 50

# Linux/Mac
tail -50 logs/backend.log
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/inference` | POST | Single prompt inference |
| `/batch` | POST | Batch file processing |
| `/status` | GET | System status |

## License

MIT License
