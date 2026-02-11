#!/usr/bin/env python3
"""
MoEl Backend - High-Performance LLM Inference Service
FastAPI backend with hardware optimization and batch processing
"""

import os
import sys
import json
import logging
import asyncio
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Configure logging
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / f"backend_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="MoEl LLM Inference API",
    description="High-performance local LLM inference service",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
llm_model = None
hardware_config = {}
executor = ThreadPoolExecutor(max_workers=4)


class InferenceRequest(BaseModel):
    """Single inference request model"""
    prompt: str = Field(..., description="Input prompt for the LLM")
    max_tokens: int = Field(512, description="Maximum tokens to generate")
    temperature: float = Field(0.7, description="Sampling temperature")
    task_type: str = Field("general", description="Task type: code_gen, code_review, translate, summarize, general")


class BatchInferenceRequest(BaseModel):
    """Batch inference request model"""
    prompts: List[str] = Field(..., description="List of prompts")
    default_prompt_template: Optional[str] = Field(None, description="Template to wrap each prompt")
    max_tokens: int = Field(512, description="Maximum tokens per generation")
    temperature: float = Field(0.7, description="Sampling temperature")
    task_type: str = Field("general", description="Task type")


class HardwareConfig(BaseModel):
    """Hardware configuration model"""
    device: str = Field("cpu", description="Device: cpu or cuda")
    gpu_layers: int = Field(0, description="Number of GPU layers")
    threads: int = Field(4, description="CPU threads")
    context_length: int = Field(2048, description="Context window size")


class LLMEngine:
    """Optimized LLM inference engine with hardware acceleration"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = config.get('device', 'cpu')
        self.model_path = config.get('model_path', None)
        self.model_name = config.get('model_name', 'gpt2')
        
        logger.info(f"Initializing LLM Engine with device: {self.device}")
        self._load_model()
    
    def _load_model(self):
        """Load the LLM model with optimal configuration"""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            
            logger.info(f"Loading model: {self.model_name}")
            
            # Configure device
            if self.device == 'cuda' and torch.cuda.is_available():
                device_map = 'auto'
                torch_dtype = torch.float16
                logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                device_map = 'cpu'
                torch_dtype = torch.float32
                logger.info("Using CPU inference")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                use_fast=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Model loading configuration
            model_kwargs = {
                'pretrained_model_name_or_path': self.model_name,
                'device_map': device_map,
                'torch_dtype': torch_dtype,
                'trust_remote_code': True,
                'low_cpu_mem_usage': True,
            }
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
            
            # Set to evaluation mode
            self.model.eval()
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7, 
                 task_type: str = "general") -> str:
        """Generate text from prompt with task-specific optimization"""
        try:
            import torch
            
            # Task-specific prompt engineering
            enhanced_prompt = self._enhance_prompt(prompt, task_type)
            
            # Tokenize
            inputs = self.tokenizer(
                enhanced_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.get('context_length', 2048)
            )
            
            # Move to device
            if self.device == 'cuda':
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    top_p=0.95,
                    top_k=50,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            # Decode
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the new generation
            result = generated_text[len(enhanced_prompt):].strip()
            
            return result
            
        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def _enhance_prompt(self, prompt: str, task_type: str) -> str:
        """Enhance prompt based on task type"""
        templates = {
            "code_gen": f"### Task: Generate clean, efficient code\n### Request: {prompt}\n### Code:\n",
            "code_review": f"### Task: Review the following code and provide feedback\n### Code:\n{prompt}\n### Review:\n",
            "translate": f"### Task: Translate the following text\n### Text: {prompt}\n### Translation:\n",
            "summarize": f"### Task: Provide a concise summary\n### Text: {prompt}\n### Summary:\n",
            "general": prompt
        }
        
        return templates.get(task_type, prompt)
    
    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Batch generation with parallel processing"""
        results = []
        for prompt in prompts:
            try:
                result = self.generate(prompt, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Batch generation error for prompt: {str(e)}")
                results.append(f"Error: {str(e)}")
        
        return results


@app.on_event("startup")
async def startup_event():
    """Initialize the LLM engine on startup"""
    global llm_model, hardware_config
    
    logger.info("Starting MoEl Backend Service")
    
    # Load hardware configuration
    config_file = Path(__file__).parent.parent / "config" / "hardware.json"
    
    if config_file.exists():
        with open(config_file, 'r') as f:
            hardware_config = json.load(f)
    else:
        hardware_config = {
            'device': 'cpu',
            'model_name': 'gpt2',
            'context_length': 2048,
            'threads': 4
        }
    
    logger.info(f"Hardware config: {hardware_config}")
    
    # Initialize LLM engine
    try:
        llm_model = LLMEngine(hardware_config)
        logger.info("LLM Engine initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize LLM Engine: {str(e)}")
        logger.warning("Backend will start but inference will fail until model is loaded")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "MoEl LLM Inference API",
        "version": "1.0.0",
        "model_loaded": llm_model is not None,
        "hardware": hardware_config
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    import torch
    
    return {
        "status": "healthy",
        "model_loaded": llm_model is not None,
        "device": hardware_config.get('device', 'cpu'),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/inference")
async def inference(request: InferenceRequest):
    """Single inference endpoint"""
    if llm_model is None:
        raise HTTPException(status_code=503, detail="LLM model not loaded")
    
    try:
        logger.info(f"Inference request - Task: {request.task_type}, Prompt length: {len(request.prompt)}")
        
        result = llm_model.generate(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            task_type=request.task_type
        )
        
        return {
            "status": "success",
            "result": result,
            "task_type": request.task_type,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Inference error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch-inference")
async def batch_inference(request: BatchInferenceRequest, background_tasks: BackgroundTasks):
    """Batch inference endpoint"""
    if llm_model is None:
        raise HTTPException(status_code=503, detail="LLM model not loaded")
    
    try:
        logger.info(f"Batch inference - {len(request.prompts)} prompts")
        
        # Apply template if provided
        if request.default_prompt_template:
            prompts = [request.default_prompt_template.format(content=p) for p in request.prompts]
        else:
            prompts = request.prompts
        
        # Generate results
        results = llm_model.batch_generate(
            prompts=prompts,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            task_type=request.task_type
        )
        
        return {
            "status": "success",
            "results": results,
            "count": len(results),
            "task_type": request.task_type,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Batch inference error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload-and-process")
async def upload_and_process(
    files: List[UploadFile] = File(...),
    task_type: str = "general",
    max_tokens: int = 512,
    temperature: float = 0.7,
    default_prompt: Optional[str] = None
):
    """Upload files and process them"""
    if llm_model is None:
        raise HTTPException(status_code=503, detail="LLM model not loaded")
    
    try:
        results = []
        
        for file in files:
            # Read file content
            content = await file.read()
            text_content = content.decode('utf-8')
            
            # Extract prompt from file or use default
            if default_prompt:
                prompt = default_prompt.format(content=text_content)
            else:
                # Look for prompt marker in file
                if "### PROMPT ###" in text_content:
                    prompt = text_content.split("### PROMPT ###")[1].split("### END PROMPT ###")[0].strip()
                else:
                    prompt = text_content
            
            # Generate
            result = llm_model.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                task_type=task_type
            )
            
            results.append({
                "filename": file.filename,
                "result": result
            })
        
        return {
            "status": "success",
            "results": results,
            "count": len(results),
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Upload and process error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/configure-hardware")
async def configure_hardware(config: HardwareConfig):
    """Reconfigure hardware settings (requires restart)"""
    global hardware_config
    
    config_file = Path(__file__).parent.parent / "config" / "hardware.json"
    config_file.parent.mkdir(exist_ok=True)
    
    hardware_config = config.dict()
    
    with open(config_file, 'w') as f:
        json.dump(hardware_config, f, indent=2)
    
    return {
        "status": "success",
        "message": "Hardware configuration updated. Please restart the backend for changes to take effect.",
        "config": hardware_config
    }


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
