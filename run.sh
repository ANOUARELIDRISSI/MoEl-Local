#!/bin/bash

###############################################################################
# MoEl - High-Performance Local LLM Application
# Main Launcher Script with Auto-Configuration
###############################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project paths
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_ROOT/venv"
CONFIG_DIR="$PROJECT_ROOT/config"
BACKEND_DIR="$PROJECT_ROOT/backend"
FRONTEND_DIR="$PROJECT_ROOT/frontend"
LOGS_DIR="$PROJECT_ROOT/logs"

# Log function
log() {
    echo -e "${GREEN}[MoEl]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

# Banner
print_banner() {
    echo -e "${BLUE}"
    echo "╔═══════════════════════════════════════════════════════════╗"
    echo "║                                                           ║"
    echo "║              🚀  MoEl - Local LLM Platform  🚀            ║"
    echo "║                                                           ║"
    echo "║          High-Performance AI Inference System             ║"
    echo "║                                                           ║"
    echo "╚═══════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# Detect OS
detect_os() {
    log "Detecting operating system..."
    
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
        log_info "OS: Linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
        log_info "OS: macOS"
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
        OS="windows"
        log_info "OS: Windows"
    else
        OS="unknown"
        log_warning "Unknown OS: $OSTYPE"
    fi
}

# Detect hardware (CPU/GPU)
detect_hardware() {
    log "Detecting hardware capabilities..."
    
    GPU_AVAILABLE=false
    GPU_TYPE="none"
    
    # Check for NVIDIA GPU
    if command -v nvidia-smi &> /dev/null; then
        if nvidia-smi &> /dev/null; then
            GPU_AVAILABLE=true
            GPU_TYPE="nvidia"
            GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
            log_info "NVIDIA GPU detected: $GPU_NAME"
        fi
    fi
    
    # Check for AMD GPU (ROCm)
    if command -v rocm-smi &> /dev/null; then
        GPU_AVAILABLE=true
        GPU_TYPE="amd"
        log_info "AMD GPU detected"
    fi
    
    # Check for Apple Silicon
    if [[ "$OS" == "macos" ]]; then
        if [[ $(uname -m) == "arm64" ]]; then
            GPU_AVAILABLE=true
            GPU_TYPE="mps"
            log_info "Apple Silicon (M1/M2/M3) detected - MPS acceleration available"
        fi
    fi
    
    if [[ "$GPU_AVAILABLE" == false ]]; then
        log_info "No GPU detected - will use CPU inference"
    fi
    
    # Detect CPU info
    if [[ "$OS" == "linux" ]]; then
        CPU_CORES=$(nproc)
    elif [[ "$OS" == "macos" ]]; then
        CPU_CORES=$(sysctl -n hw.ncpu)
    else
        CPU_CORES=4
    fi
    log_info "CPU cores: $CPU_CORES"
}

# Ask user about GPU usage
configure_hardware() {
    log "Configuring hardware settings..."
    
    mkdir -p "$CONFIG_DIR"
    
    DEVICE="cpu"
    GPU_LAYERS=0
    
    if [[ "$GPU_AVAILABLE" == true ]]; then
        echo ""
        echo -e "${YELLOW}GPU detected! Would you like to enable GPU acceleration?${NC}"
        echo "1) Yes - Use GPU acceleration (faster, requires more VRAM)"
        echo "2) No - Use CPU only (slower, more compatible)"
        read -p "Enter choice [1-2]: " gpu_choice
        
        if [[ "$gpu_choice" == "1" ]]; then
            if [[ "$GPU_TYPE" == "nvidia" ]]; then
                DEVICE="cuda"
            elif [[ "$GPU_TYPE" == "mps" ]]; then
                DEVICE="mps"
            elif [[ "$GPU_TYPE" == "amd" ]]; then
                DEVICE="cuda"  # ROCm uses cuda API
            fi
            GPU_LAYERS=35
            log "GPU acceleration enabled!"
        else
            log "CPU-only mode selected"
        fi
    fi
    
    # Ask for model selection
    echo ""
    echo -e "${YELLOW}Select the LLM model to use:${NC}"
    echo "1) Qwen2.5-0.5B-Instruct (~1GB) - RECOMMENDED: Fast & accurate on CPU"
    echo "2) Qwen2.5-1.5B-Instruct (~3GB) - Better quality, still fast"
    echo "3) GPT-2 (small, ~500MB) - Basic, for testing"
    echo "4) GPT-2 Medium (~1.5GB) - Better GPT-2"
    echo "5) Custom model (enter HuggingFace model name)"
    read -p "Enter choice [1-5] (default: 1): " model_choice
    
    case $model_choice in
        1|"") MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct" ;;
        2) MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct" ;;
        3) MODEL_NAME="gpt2" ;;
        4) MODEL_NAME="gpt2-medium" ;;
        5) 
            read -p "Enter HuggingFace model name: " MODEL_NAME
            ;;
        *) 
            MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct"
            log_warning "Invalid choice, using default: Qwen2.5-0.5B-Instruct"
            ;;
    esac
    
    log_info "Selected model: $MODEL_NAME"
    
    # Save configuration
    cat > "$CONFIG_DIR/hardware.json" <<EOF
{
    "device": "$DEVICE",
    "gpu_type": "$GPU_TYPE",
    "gpu_layers": $GPU_LAYERS,
    "threads": $CPU_CORES,
    "context_length": 2048,
    "model_name": "$MODEL_NAME"
}
EOF
    
    log "Hardware configuration saved to $CONFIG_DIR/hardware.json"
}

# Install system dependencies
install_system_deps() {
    log "Checking system dependencies..."
    
    if [[ "$OS" == "linux" ]]; then
        # Check if running with sudo
        if command -v apt-get &> /dev/null; then
            log_info "Debian/Ubuntu detected"
            
            # Check for required packages
            REQUIRED_PACKAGES="python3 python3-pip python3-venv build-essential"
            
            for pkg in $REQUIRED_PACKAGES; do
                if ! dpkg -l | grep -q "^ii  $pkg"; then
                    log_warning "Package $pkg not found. Please install with:"
                    log_warning "sudo apt-get update && sudo apt-get install -y $REQUIRED_PACKAGES"
                fi
            done
        elif command -v yum &> /dev/null; then
            log_info "RHEL/CentOS detected"
            log_warning "Ensure python3, pip3, and development tools are installed"
        fi
    elif [[ "$OS" == "macos" ]]; then
        log_info "macOS detected"
        if ! command -v brew &> /dev/null; then
            log_warning "Homebrew not found. Consider installing: https://brew.sh"
        fi
    fi
}

# Setup Python virtual environment
setup_venv() {
    log "Setting up Python virtual environment..."
    
    if [[ ! -d "$VENV_DIR" ]]; then
        log "Creating virtual environment..."
        python3 -m venv "$VENV_DIR"
        log "Virtual environment created"
    else
        log_info "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    source "$VENV_DIR/bin/activate"
    log "Virtual environment activated"
    
    # Upgrade pip
    log "Upgrading pip..."
    pip install --upgrade pip setuptools wheel --quiet
}

# Install Python dependencies
install_dependencies() {
    log "Installing Python dependencies..."
    
    # Core dependencies
    log_info "Installing core packages..."
    
    # Install PyTorch based on hardware
    if [[ "$DEVICE" == "cuda" ]]; then
        log_info "Installing PyTorch with CUDA support..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --quiet
    elif [[ "$DEVICE" == "mps" ]]; then
        log_info "Installing PyTorch with MPS support..."
        pip install torch torchvision torchaudio --quiet
    else
        log_info "Installing PyTorch (CPU only)..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --quiet
    fi
    
    # Install transformers and other dependencies
    log_info "Installing Transformers and dependencies..."
    pip install transformers accelerate --quiet
    
    # Install FastAPI and dependencies
    log_info "Installing FastAPI..."
    pip install fastapi uvicorn[standard] python-multipart --quiet
    
    # Install Flask
    log_info "Installing Flask..."
    pip install flask flask-cors --quiet
    
    # Install utilities
    log_info "Installing utilities..."
    pip install requests aiofiles pydantic --quiet
    
    log "All dependencies installed successfully"
}

# Generate requirements.txt
generate_requirements() {
    log "Generating requirements.txt..."
    
    pip freeze > "$PROJECT_ROOT/requirements.txt"
    log "requirements.txt generated"
    log_info "$(wc -l < $PROJECT_ROOT/requirements.txt) packages installed"
}

# Start backend service
start_backend() {
    log "Starting backend service..."
    
    cd "$BACKEND_DIR"
    
    # Start backend in background
    nohup python app.py > "$LOGS_DIR/backend.log" 2>&1 &
    BACKEND_PID=$!
    echo $BACKEND_PID > "$LOGS_DIR/backend.pid"
    
    log "Backend started (PID: $BACKEND_PID)"
    log_info "Backend logs: $LOGS_DIR/backend.log"
    
    # Wait for backend to start
    log "Waiting for backend to be ready..."
    for i in {1..30}; do
        if curl -s http://localhost:8000/ > /dev/null 2>&1; then
            log "Backend is ready!"
            return 0
        fi
        sleep 1
    done
    
    log_error "Backend failed to start within 30 seconds"
    return 1
}

# Start frontend service
start_frontend() {
    log "Starting frontend service..."
    
    cd "$FRONTEND_DIR"
    
    # Set backend URL
    export BACKEND_URL="http://localhost:8000"
    
    # Start frontend in background
    nohup python app.py > "$LOGS_DIR/frontend.log" 2>&1 &
    FRONTEND_PID=$!
    echo $FRONTEND_PID > "$LOGS_DIR/frontend.pid"
    
    log "Frontend started (PID: $FRONTEND_PID)"
    log_info "Frontend logs: $LOGS_DIR/frontend.log"
    
    # Wait for frontend to start
    log "Waiting for frontend to be ready..."
    for i in {1..20}; do
        if curl -s http://localhost:5000/ > /dev/null 2>&1; then
            log "Frontend is ready!"
            return 0
        fi
        sleep 1
    done
    
    log_error "Frontend failed to start within 20 seconds"
    return 1
}

# Stop services
stop_services() {
    log "Stopping services..."
    
    if [[ -f "$LOGS_DIR/backend.pid" ]]; then
        BACKEND_PID=$(cat "$LOGS_DIR/backend.pid")
        if kill -0 $BACKEND_PID 2>/dev/null; then
            kill $BACKEND_PID
            log "Backend stopped"
        fi
        rm "$LOGS_DIR/backend.pid"
    fi
    
    if [[ -f "$LOGS_DIR/frontend.pid" ]]; then
        FRONTEND_PID=$(cat "$LOGS_DIR/frontend.pid")
        if kill -0 $FRONTEND_PID 2>/dev/null; then
            kill $FRONTEND_PID
            log "Frontend stopped"
        fi
        rm "$LOGS_DIR/frontend.pid"
    fi
}

# Main installation and setup
setup() {
    print_banner
    
    log "Starting MoEl setup..."
    
    # Create necessary directories
    mkdir -p "$CONFIG_DIR" "$LOGS_DIR"
    
    # Detect system
    detect_os
    detect_hardware
    
    # Configure hardware
    configure_hardware
    
    # Install system dependencies
    install_system_deps
    
    # Setup Python environment
    setup_venv
    
    # Install dependencies
    install_dependencies
    
    # Generate requirements.txt
    generate_requirements
    
    log ""
    log "✨ Setup completed successfully!"
    log ""
}

# Start application
start() {
    print_banner
    
    log "Starting MoEl application..."
    
    # Ensure venv exists
    if [[ ! -d "$VENV_DIR" ]]; then
        log_error "Virtual environment not found. Please run: ./run.sh setup"
        exit 1
    fi
    
    # Activate venv
    source "$VENV_DIR/bin/activate"
    
    # Load configuration
    if [[ -f "$CONFIG_DIR/hardware.json" ]]; then
        log_info "Configuration loaded"
    else
        log_error "Configuration not found. Please run: ./run.sh setup"
        exit 1
    fi
    
    # Stop any existing services
    stop_services
    
    # Start services
    start_backend
    if [[ $? -ne 0 ]]; then
        log_error "Failed to start backend"
        exit 1
    fi
    
    start_frontend
    if [[ $? -ne 0 ]]; then
        log_error "Failed to start frontend"
        stop_services
        exit 1
    fi
    
    # Success message
    echo ""
    echo -e "${GREEN}╔═══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                                                           ║${NC}"
    echo -e "${GREEN}║     ✅  MoEl is running successfully!                     ║${NC}"
    echo -e "${GREEN}║                                                           ║${NC}"
    echo -e "${GREEN}║     🌐  Web Interface: http://localhost:5000              ║${NC}"
    echo -e "${GREEN}║     📡  Backend API:   http://localhost:8000              ║${NC}"
    echo -e "${GREEN}║                                                           ║${NC}"
    echo -e "${GREEN}║     To stop: ./run.sh stop                                ║${NC}"
    echo -e "${GREEN}║     View logs: tail -f logs/*.log                         ║${NC}"
    echo -e "${GREEN}║                                                           ║${NC}"
    echo -e "${GREEN}╚═══════════════════════════════════════════════════════════╝${NC}"
    echo ""
    
    # Keep script running
    log "Press Ctrl+C to stop the application"
    trap stop_services EXIT
    
    # Tail logs
    tail -f "$LOGS_DIR/backend.log" "$LOGS_DIR/frontend.log"
}

# Show usage
usage() {
    echo "MoEl - High-Performance Local LLM Application"
    echo ""
    echo "Usage: ./run.sh [command]"
    echo ""
    echo "Commands:"
    echo "  setup     - Initial setup (install dependencies, configure hardware)"
    echo "  start     - Start the application"
    echo "  stop      - Stop all services"
    echo "  restart   - Restart all services"
    echo "  status    - Check service status"
    echo "  logs      - View logs"
    echo ""
}

# Check service status
status() {
    log "Checking service status..."
    
    echo ""
    if [[ -f "$LOGS_DIR/backend.pid" ]]; then
        BACKEND_PID=$(cat "$LOGS_DIR/backend.pid")
        if kill -0 $BACKEND_PID 2>/dev/null; then
            echo -e "${GREEN}✓${NC} Backend is running (PID: $BACKEND_PID)"
        else
            echo -e "${RED}✗${NC} Backend is not running"
        fi
    else
        echo -e "${RED}✗${NC} Backend is not running"
    fi
    
    if [[ -f "$LOGS_DIR/frontend.pid" ]]; then
        FRONTEND_PID=$(cat "$LOGS_DIR/frontend.pid")
        if kill -0 $FRONTEND_PID 2>/dev/null; then
            echo -e "${GREEN}✓${NC} Frontend is running (PID: $FRONTEND_PID)"
        else
            echo -e "${RED}✗${NC} Frontend is not running"
        fi
    else
        echo -e "${RED}✗${NC} Frontend is not running"
    fi
    echo ""
}

# View logs
view_logs() {
    log "Viewing logs..."
    tail -f "$LOGS_DIR/backend.log" "$LOGS_DIR/frontend.log"
}

# Main command handler
case "${1:-}" in
    setup)
        setup
        ;;
    start)
        start
        ;;
    stop)
        stop_services
        ;;
    restart)
        stop_services
        sleep 2
        start
        ;;
    status)
        status
        ;;
    logs)
        view_logs
        ;;
    *)
        usage
        exit 1
        ;;
esac
