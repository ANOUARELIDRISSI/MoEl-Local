#!/usr/bin/env python3
"""
MoEl - Cross-Platform Setup & Launcher
Works on Windows, Linux, and macOS
"""

import os
import sys
import subprocess
import platform
import shutil
import time
import json
from pathlib import Path

# Configuration
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
BACKEND_PORT = 8000
FRONTEND_PORT = 5000

# Colors (ANSI codes work on most terminals)
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    
    @staticmethod
    def disable():
        Colors.GREEN = Colors.RED = Colors.YELLOW = Colors.BLUE = Colors.RESET = ''

# Disable colors on Windows cmd (not PowerShell/Terminal)
if platform.system() == 'Windows' and 'WT_SESSION' not in os.environ:
    try:
        os.system('color')  # Enable ANSI on Windows
    except:
        Colors.disable()

def log(msg): print(f"{Colors.GREEN}[MoEl]{Colors.RESET} {msg}")
def log_error(msg): print(f"{Colors.RED}[ERROR]{Colors.RESET} {msg}")
def log_info(msg): print(f"{Colors.BLUE}[INFO]{Colors.RESET} {msg}")
def log_warn(msg): print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} {msg}")

def banner():
    print(f"""
{Colors.BLUE}═══════════════════════════════════════════════════════════
         🚀 MoEl - Local LLM Platform
         Cross-Platform Setup & Launcher
═══════════════════════════════════════════════════════════{Colors.RESET}
""")

# Paths
PROJECT_ROOT = Path(__file__).parent.absolute()
VENV_NAME = "venv_win" if platform.system() == "Windows" else "venv"
VENV_DIR = PROJECT_ROOT / VENV_NAME
BACKEND_DIR = PROJECT_ROOT / "backend"
FRONTEND_DIR = PROJECT_ROOT / "frontend"
LOGS_DIR = PROJECT_ROOT / "logs"
CONFIG_FILE = PROJECT_ROOT / "config" / "hardware.json"

def get_python_executable():
    """Get the venv Python executable path"""
    if platform.system() == "Windows":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"

def get_pip_executable():
    """Get the venv pip executable path"""
    if platform.system() == "Windows":
        return VENV_DIR / "Scripts" / "pip.exe"
    return VENV_DIR / "bin" / "pip"

def run_command(cmd, cwd=None, check=True, capture=False):
    """Run a shell command"""
    try:
        result = subprocess.run(
            cmd, shell=True, cwd=cwd, check=check,
            capture_output=capture, text=True
        )
        return result
    except subprocess.CalledProcessError as e:
        if capture:
            log_error(f"Command failed: {e.stderr}")
        raise

def check_python():
    """Check Python installation"""
    try:
        version = platform.python_version()
        log_info(f"Python {version} detected")
        if tuple(map(int, version.split('.')[:2])) < (3, 8):
            log_error("Python 3.8+ required")
            return False
        return True
    except:
        log_error("Python not found")
        return False

def create_directories():
    """Create required directories"""
    dirs = [LOGS_DIR, PROJECT_ROOT / "uploads", PROJECT_ROOT / "outputs", PROJECT_ROOT / "models"]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    log_info("Directories created")

def create_venv():
    """Create virtual environment"""
    if VENV_DIR.exists():
        log_info("Virtual environment already exists")
        return
    
    log_info("Creating virtual environment...")
    run_command(f'"{sys.executable}" -m venv "{VENV_DIR}"')
    log("Virtual environment created")

def install_dependencies():
    """Install Python dependencies"""
    log_info("Installing dependencies...")
    pip = get_pip_executable()
    
    # Upgrade pip
    run_command(f'"{pip}" install --quiet --upgrade pip', check=False)
    
    # Core dependencies
    deps = [
        "fastapi", "uvicorn[standard]", "flask", "flask-cors",
        "python-multipart", "pydantic", "pydantic-settings",
        "requests", "aiofiles", "httpx",
        "torch", "transformers", "accelerate",
        "tqdm", "colorlog", "numpy"
    ]
    
    run_command(f'"{pip}" install --quiet {" ".join(deps)}')
    log("Dependencies installed")

def download_model():
    """Pre-download the LLM model"""
    log_info(f"Downloading model: {MODEL_NAME}...")
    python = get_python_executable()
    
    code = f'''
from transformers import AutoModelForCausalLM, AutoTokenizer
print("Downloading tokenizer...")
AutoTokenizer.from_pretrained("{MODEL_NAME}", trust_remote_code=True)
print("Downloading model...")
AutoModelForCausalLM.from_pretrained("{MODEL_NAME}", trust_remote_code=True)
print("Done!")
'''
    run_command(f'"{python}" -c "{code}"')
    log("Model downloaded")

def update_config():
    """Update hardware config with selected model"""
    config = {
        "device": "cpu",
        "gpu_type": "none",
        "gpu_layers": 0,
        "threads": 4,
        "context_length": 2048,
        "model_name": MODEL_NAME
    }
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)
    log_info(f"Config updated: {MODEL_NAME}")

def setup():
    """Full setup process"""
    banner()
    log("Starting setup...")
    
    if not check_python():
        sys.exit(1)
    
    create_directories()
    create_venv()
    install_dependencies()
    update_config()
    download_model()
    
    print(f"""
{Colors.GREEN}═══════════════════════════════════════════════════════════
  ✅ Setup completed successfully!
═══════════════════════════════════════════════════════════{Colors.RESET}

To start MoEl, run:
  {Colors.BLUE}python setup.py start{Colors.RESET}

Or on Windows PowerShell:
  {Colors.BLUE}.\\run.ps1 start{Colors.RESET}

Or on Linux/Mac:
  {Colors.BLUE}./run.sh start{Colors.RESET}
""")

def kill_port(port):
    """Kill process on a specific port"""
    try:
        if platform.system() == "Windows":
            result = run_command(f'netstat -ano | findstr :{port}', capture=True, check=False)
            if result.stdout:
                for line in result.stdout.strip().split('\n'):
                    parts = line.split()
                    if len(parts) >= 5:
                        pid = parts[-1]
                        run_command(f'taskkill /F /PID {pid}', check=False)
        else:
            run_command(f'lsof -ti:{port} | xargs kill -9 2>/dev/null', check=False)
    except:
        pass

def start():
    """Start MoEl services"""
    banner()
    
    if not VENV_DIR.exists():
        log_error("Setup not completed. Run: python setup.py setup")
        sys.exit(1)
    
    python = get_python_executable()
    
    # Stop existing
    log_info("Stopping existing processes...")
    kill_port(BACKEND_PORT)
    kill_port(FRONTEND_PORT)
    time.sleep(1)
    
    # Start backend
    log_info(f"Starting Backend API (port {BACKEND_PORT})...")
    if platform.system() == "Windows":
        subprocess.Popen(
            [str(python), "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", str(BACKEND_PORT)],
            cwd=BACKEND_DIR,
            creationflags=subprocess.CREATE_NEW_CONSOLE
        )
    else:
        subprocess.Popen(
            [str(python), "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", str(BACKEND_PORT)],
            cwd=BACKEND_DIR,
            stdout=open(LOGS_DIR / "backend.log", "w"),
            stderr=subprocess.STDOUT,
            start_new_session=True
        )
    
    # Wait for backend
    log_info("Waiting for backend to initialize...")
    import urllib.request
    for i in range(60):
        try:
            urllib.request.urlopen(f"http://localhost:{BACKEND_PORT}/health", timeout=2)
            log("Backend is ready!")
            break
        except:
            print(".", end="", flush=True)
            time.sleep(2)
    print()
    
    # Start frontend
    log_info(f"Starting Frontend (port {FRONTEND_PORT})...")
    if platform.system() == "Windows":
        subprocess.Popen(
            [str(python), "app.py"],
            cwd=FRONTEND_DIR,
            creationflags=subprocess.CREATE_NEW_CONSOLE
        )
    else:
        subprocess.Popen(
            [str(python), "app.py"],
            cwd=FRONTEND_DIR,
            stdout=open(LOGS_DIR / "frontend.log", "w"),
            stderr=subprocess.STDOUT,
            start_new_session=True
        )
    
    time.sleep(3)
    
    print(f"""
{Colors.GREEN}═══════════════════════════════════════════════════════════
  ✅ MoEl is running!
═══════════════════════════════════════════════════════════{Colors.RESET}

  Backend API:   {Colors.BLUE}http://localhost:{BACKEND_PORT}{Colors.RESET}
  Web Interface: {Colors.BLUE}http://localhost:{FRONTEND_PORT}{Colors.RESET}
  Model:         {Colors.YELLOW}{MODEL_NAME}{Colors.RESET}

  Open your browser at: http://localhost:{FRONTEND_PORT}
""")
    
    # Open browser
    try:
        import webbrowser
        webbrowser.open(f"http://localhost:{FRONTEND_PORT}")
    except:
        pass

def stop():
    """Stop MoEl services"""
    banner()
    log("Stopping MoEl...")
    kill_port(BACKEND_PORT)
    kill_port(FRONTEND_PORT)
    log("MoEl stopped")

def status():
    """Check MoEl status"""
    banner()
    import urllib.request
    
    # Check backend
    try:
        urllib.request.urlopen(f"http://localhost:{BACKEND_PORT}/health", timeout=2)
        log(f"Backend:  RUNNING (port {BACKEND_PORT})")
    except:
        log_error(f"Backend:  STOPPED")
    
    # Check frontend
    try:
        urllib.request.urlopen(f"http://localhost:{FRONTEND_PORT}/", timeout=2)
        log(f"Frontend: RUNNING (port {FRONTEND_PORT})")
    except:
        log_error(f"Frontend: STOPPED")

def main():
    commands = {
        'setup': setup,
        'start': start,
        'stop': stop,
        'restart': lambda: (stop(), time.sleep(2), start()),
        'status': status,
    }
    
    if len(sys.argv) < 2:
        print(f"""
Usage: python setup.py <command>

Commands:
  setup    - Install dependencies and download model
  start    - Start MoEl (backend + frontend)
  stop     - Stop all MoEl services
  restart  - Restart MoEl
  status   - Check if services are running
""")
        sys.exit(0)
    
    cmd = sys.argv[1].lower()
    if cmd in commands:
        commands[cmd]()
    else:
        log_error(f"Unknown command: {cmd}")
        sys.exit(1)

if __name__ == "__main__":
    main()
