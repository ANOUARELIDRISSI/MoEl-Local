#!/usr/bin/env python3
"""
MoEl Frontend - Flask Web Interface
User-friendly web interface for LLM interaction
"""

import os
import json
import logging
import requests
from datetime import datetime
from pathlib import Path

from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename

# Configure logging
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / f"frontend_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['UPLOAD_FOLDER'] = Path(__file__).parent.parent / 'uploads'
app.config['OUTPUT_FOLDER'] = Path(__file__).parent.parent / 'outputs'
app.config['UPLOAD_FOLDER'].mkdir(exist_ok=True)
app.config['OUTPUT_FOLDER'].mkdir(exist_ok=True)

# Backend API URL
BACKEND_URL = os.getenv('BACKEND_URL', 'http://localhost:8000')

ALLOWED_EXTENSIONS = {'txt', 'md', 'py', 'js', 'java', 'cpp', 'c', 'go', 'rs', 'json', 'xml', 'html', 'css'}


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')


@app.route('/api/health')
def health():
    """Frontend health check"""
    try:
        # Check backend health
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        backend_status = response.json()
        
        return jsonify({
            "status": "healthy",
            "frontend": "online",
            "backend": backend_status,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            "status": "degraded",
            "frontend": "online",
            "backend": "offline",
            "error": str(e)
        }), 503


@app.route('/api/inference', methods=['POST'])
def inference():
    """Single inference endpoint"""
    try:
        data = request.json
        
        # Validate request
        if not data or 'prompt' not in data:
            return jsonify({"error": "Prompt is required"}), 400
        
        # Send to backend
        response = requests.post(
            f"{BACKEND_URL}/inference",
            json={
                "prompt": data['prompt'],
                "max_tokens": data.get('max_tokens', 512),
                "temperature": data.get('temperature', 0.7),
                "task_type": data.get('task_type', 'general')
            },
            timeout=300  # 5 minutes timeout for long generations
        )
        
        response.raise_for_status()
        return jsonify(response.json())
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Backend request failed: {str(e)}")
        return jsonify({"error": f"Backend error: {str(e)}"}), 500
    except Exception as e:
        logger.error(f"Inference error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/batch-inference', methods=['POST'])
def batch_inference():
    """Batch inference endpoint"""
    try:
        data = request.json
        
        if not data or 'prompts' not in data:
            return jsonify({"error": "Prompts array is required"}), 400
        
        # Send to backend
        response = requests.post(
            f"{BACKEND_URL}/batch-inference",
            json={
                "prompts": data['prompts'],
                "default_prompt_template": data.get('default_prompt_template'),
                "max_tokens": data.get('max_tokens', 512),
                "temperature": data.get('temperature', 0.7),
                "task_type": data.get('task_type', 'general')
            },
            timeout=600  # 10 minutes for batch
        )
        
        response.raise_for_status()
        return jsonify(response.json())
    
    except Exception as e:
        logger.error(f"Batch inference error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/upload', methods=['POST'])
def upload_files():
    """Upload and process files"""
    try:
        if 'files' not in request.files:
            return jsonify({"error": "No files provided"}), 400
        
        files = request.files.getlist('files')
        task_type = request.form.get('task_type', 'general')
        max_tokens = int(request.form.get('max_tokens', 512))
        temperature = float(request.form.get('temperature', 0.7))
        default_prompt = request.form.get('default_prompt', '')
        
        if not files or files[0].filename == '':
            return jsonify({"error": "No files selected"}), 400
        
        # Save and process files
        file_data = []
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = app.config['UPLOAD_FOLDER'] / filename
                file.save(filepath)
                
                # Read content
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                file_data.append({
                    'filename': filename,
                    'content': content
                })
        
        if not file_data:
            return jsonify({"error": "No valid files uploaded"}), 400
        
        # Prepare backend request
        backend_files = []
        for fd in file_data:
            filepath = app.config['UPLOAD_FOLDER'] / fd['filename']
            backend_files.append(
                ('files', (fd['filename'], open(filepath, 'rb'), 'text/plain'))
            )
        
        # Send to backend
        response = requests.post(
            f"{BACKEND_URL}/upload-and-process",
            files=backend_files,
            data={
                'task_type': task_type,
                'max_tokens': max_tokens,
                'temperature': temperature,
                'default_prompt': default_prompt if default_prompt else None
            },
            timeout=600
        )
        
        # Close file handles
        for _, file_tuple in backend_files:
            file_tuple[1].close()
        
        response.raise_for_status()
        result = response.json()
        
        # Save results to output folder
        output_file = app.config['OUTPUT_FOLDER'] / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        return jsonify({
            **result,
            "output_file": str(output_file.name)
        })
    
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/download/<filename>')
def download_file(filename):
    """Download result file"""
    try:
        filepath = app.config['OUTPUT_FOLDER'] / secure_filename(filename)
        if not filepath.exists():
            return jsonify({"error": "File not found"}), 404
        
        return send_file(filepath, as_attachment=True)
    
    except Exception as e:
        logger.error(f"Download error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/backend-status')
def backend_status():
    """Get backend status"""
    try:
        response = requests.get(f"{BACKEND_URL}/", timeout=5)
        return jsonify(response.json())
    except Exception as e:
        return jsonify({
            "status": "offline",
            "error": str(e)
        }), 503


if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False
    )
