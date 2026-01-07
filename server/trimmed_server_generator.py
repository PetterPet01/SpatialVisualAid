import os
import cv2
import numpy as np
import time
import json
import uuid
import logging
import asyncio
from typing import List, Optional, Dict, Any, Deque, Tuple
from contextlib import contextmanager
from collections import deque
from io import BytesIO

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import torch
from PIL import Image

# Import the Generator instead of custom LLM interfaces
from gemstral_client import Generator
from generalized_sg_generator_hf_yoloe import GeneralizedSceneGraphGenerator

# --- Configuration ---
CONFIG_PATH = os.environ.get("GSG_CONFIG_PATH", "configs/v2_hf_llm.py")
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", 8000))

# Generator Configuration
GATEWAY_BASE_URL = os.environ.get("GATEWAY_BASE_URL", "http://localhost:8543")
GATEWAY_API_KEY = os.environ.get("GATEWAY_API_KEY", "dummy")
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "mistral-medium-latest")
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.7"))
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "4096"))
GENERATOR_TIMEOUT = int(os.environ.get("GENERATOR_TIMEOUT", "120"))

# --- Logging & Profiling ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Server")

class PipelineProfiler:
    """Context manager for profiling code sections."""
    def __init__(self):
        self.timings = {}
    
    @contextmanager
    def profile(self, name: str):
        start = time.time()
        yield
        dur = time.time() - start
        self.timings[name] = dur
        logger.info(f"[PROFILE] {name}: {dur:.3f}s")

# --- Session Management ---
class SessionManager:
    """
    Manages conversation history per session ID with sliding window.
    """
    def __init__(self, max_history: int = 12):
        self.sessions: Dict[str, List[Dict[str, Any]]] = {}
        self.max_history = max_history
        
        self.system_prompt = {
            "role": "system", 
            "content": (
                "You are a smart AI assistant for a visually impaired user, speaking Vietnamese. "
                "You may receive 'Visual Scene Data' (JSON) or direct images. "
                "1. If you get JSON, use the spatial_properties (left, right, coordinates) to answer questions about location/size precisely. "
                "2. If you get an image, analyze it naturally. "
                "3. Keep answers concise and helpful for voice output. "
            )
        }

    def get_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get conversation history for a session, prepended with system prompt."""
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        return [self.system_prompt] + self.sessions[session_id]

    def add_message(self, session_id: str, role: str, content: Any):
        """Add a message to the conversation history."""
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        
        self.sessions[session_id].append({"role": role, "content": content})
        
        # Sliding window logic to prevent context overflow
        if len(self.sessions[session_id]) > self.max_history * 2:
            self.sessions[session_id] = self.sessions[session_id][-(self.max_history*2):]

# --- Global State ---
app = FastAPI(title="KWS Voice Assistant Server with Generator")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

generator: Optional[Generator] = None
scene_graph_generator: Optional[GeneralizedSceneGraphGenerator] = None
session_manager = SessionManager()

@app.on_event("startup")
async def startup_event():
    global generator, scene_graph_generator
    logger.info("="*80)
    logger.info("Server Starting...")

    # 1. Initialize Generator (connected to Unified AI Gateway)
    try:
        logger.info(f"Initializing Generator with Gateway: {GATEWAY_BASE_URL}")
        generator = Generator(
            base_url=GATEWAY_BASE_URL,
            api_key=GATEWAY_API_KEY,
            model_name=DEFAULT_MODEL,
            temperature=LLM_TEMPERATURE,
            max_new_tokens=LLM_MAX_TOKENS,
            timeout=GENERATOR_TIMEOUT
        )
        logger.info("✅ Generator initialized successfully")
    except Exception as e:
        logger.critical(f"FATAL: Generator initialization failed: {e}")
        generator = None

    # 2. Initialize Scene Graph Generator (for image processing)
    try:
        logger.info("Initializing Scene Graph Generator...")
        scene_graph_generator = GeneralizedSceneGraphGenerator(
            config_path=CONFIG_PATH,
            device="cuda" if torch.cuda.is_available() else "cpu",
            llm_interface=None  # We're using Generator instead
        )
        logger.info("✅ Scene Graph Generator initialized")
    except Exception as e:
        logger.critical(f"FATAL: SGG initialization failed: {e}")
        scene_graph_generator = None

    logger.info("="*80)

@app.post("/interact/")
async def interact_endpoint(
    session_id: str = Form(...),
    text_input: str = Form(...),
    image_mode: str = Form("process"),
    file: Optional[UploadFile] = File(None),
    model_name: Optional[str] = Form(None),
    temperature: Optional[float] = Form(None),
    max_tokens: Optional[int] = Form(None)
):
    """
    Main interaction endpoint: processes images and generates responses using Generator.
    
    Args:
        session_id: Unique session identifier for conversation history
        text_input: User's text input/question
        image_mode: "none" | "process" | "raw" | "both"
        file: Optional image file
        model_name: Optional override for model
        temperature: Optional override for temperature
        max_tokens: Optional override for max tokens
    """
    global generator, scene_graph_generator, session_manager
    
    if not generator:
        raise HTTPException(status_code=503, detail="Generator not initialized")

    profiler = PipelineProfiler()

    # 1. Process Image (if provided)
    user_message_content = None
    
    if file and image_mode != "none":
        with profiler.profile("Image Processing"):
            try:
                contents = await file.read()
                nparr = np.frombuffer(contents, np.uint8)
                image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if scene_graph_generator:
                    # Get structured scene graph data
                    img_result = scene_graph_generator.process_image_for_interaction(
                        image_bgr, 
                        mode=image_mode
                    )
                else:
                    # Fallback: just encode image as base64
                    _, buffer = cv2.imencode('.jpg', image_bgr)
                    img_base64 = __import__('base64').b64encode(buffer).decode('utf-8')
                    img_result = {
                        'raw_image_b64': img_base64,
                        'content': "Image provided (scene graph data unavailable)"
                    }

                # Construct multimodal payload based on mode
                if image_mode == "raw":
                    # Image only
                    user_message_content = [
                        {"type": "text", "text": f"User Request: {text_input}"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_result['raw_image_b64']}"}}
                    ]
                elif image_mode == "both":
                    # Scene data + image
                    user_message_content = [
                        {"type": "text", "text": f"Scene Data (JSON):\n{img_result['content']}\n\nUser Request: {text_input}"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_result['raw_image_b64']}"}}
                    ]
                else:  # "process"
                    # Scene data only
                    user_message_content = f"Visual Context:\n{img_result['content']}\n\nUser Request: {text_input}"

                logger.info(f"Image processed successfully in mode '{image_mode}'")

            except Exception as e:
                logger.error(f"Image processing error: {e}", exc_info=True)
                user_message_content = f"Error processing image. Request: {text_input}"

    # Fallback if no image
    if user_message_content is None:
        user_message_content = text_input

    # 2. Add user message to history
    session_manager.add_message(session_id, "user", user_message_content)

    # 3. Generate response
    try:
        with profiler.profile("LLM Generation"):
            history = session_manager.get_history(session_id)
            
            # Use optional parameter overrides or defaults
            gen_model = model_name or DEFAULT_MODEL
            gen_temperature = temperature if temperature is not None else LLM_TEMPERATURE
            gen_max_tokens = max_tokens or LLM_MAX_TOKENS
            
            # Call generator with conversation history and optional images
            response_text, updated_history = generator.generate(
                messages=history,
                model_name=gen_model,
                temperature=gen_temperature,
                max_new_tokens=gen_max_tokens
            )
            
            logger.info(f"Response generated successfully using model '{gen_model}'")

        # 4. Update session history and return
        session_manager.add_message(session_id, "assistant", response_text)

        return {
            "session_id": session_id,
            "reply": response_text,
            "model_used": gen_model,
            "profiling": profiler.timings
        }

    except Exception as e:
        logger.error(f"Generation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.post("/detect_intent/")
async def detect_intent_endpoint(
    text_input: str = Form(...),
    model_name: Optional[str] = Form(None)
):
    """
    Detect user intent using the Generator.
    Returns: {intent: str, confidence: float, model_used: str}
    """
    global generator

    if not generator:
        raise HTTPException(status_code=503, detail="Generator not initialized")

    if not text_input or not text_input.strip():
        raise HTTPException(status_code=400, detail="text_input cannot be empty")

    intent_system_prompt = (
        "You are an intent classifier. Classify the user input into one of these intents:\n"
        "1. 'REQUEST_IMAGE': User explicitly requests capturing a new photo using the camera (e.g., 'chụp ảnh', 'take a picture', 'open camera', 'capture an image'). "
        "This intent ONLY applies when the user is directly asking the system to take a new photo.\n"
        "2. 'ASK_QUESTION': User is asking a question, requesting information, or referring to analyzing/answering based on an image that has ALREADY been taken. "
        "Any informational query—even if about an image—is NOT a request to take a photo.\n"
        "3. 'UNKNOWN': Input doesn't clearly match any intent.\n\n"
        "Guidelines:\n"
        "- If the user asks a question about an existing or previously captured photo, classify it as 'ASK_QUESTION'.\n"
        "- ONLY classify as 'REQUEST_IMAGE' if the user explicitly asks to take or capture a photo.\n"
        "- Do not infer a capture request unless the user clearly states it.\n\n"
        "Respond with ONLY a JSON object: {\"intent\": \"<INTENT>\", \"confidence\": <0.0-1.0>}.\n"
        "Do not include any other text, markdown, or explanation."
    )


    try:
        gen_model = model_name or DEFAULT_MODEL
        
        messages = [
            {"role": "system", "content": intent_system_prompt},
            {"role": "user", "content": text_input}
        ]

        response_text, _ = generator.generate(
            messages=messages,
            model_name=gen_model,
            temperature=0.1,
            max_new_tokens=1024
        )

        # ======================= DEBUG LOGGING =======================
        logger.info("⬇️⬇️⬇️ FULL RAW API RESPONSE START ⬇️⬇️⬇️")
        logger.info(f"TYPE: {type(response_text)}")
        try:
            # Try to pretty print if it's a list/dict, otherwise print raw
            logger.info(f"CONTENT: {json.dumps(response_text, default=str)}") 
        except:
            logger.info(f"CONTENT: {response_text}")
        logger.info("⬆️⬆️⬆️ FULL RAW API RESPONSE END ⬆️⬆️⬆️")
        # =============================================================

        # --- Aggressive Cleaning Logic ---
        
        # 1. Flatten List to String
        if isinstance(response_text, list):
            full_text = " ".join([str(item) for item in response_text])
        else:
            full_text = str(response_text)

        # 2. Extract Text from Mistral "TextChunk" artifacts
        # Pattern: looks for text='...' or text="..." inside [TextChunk(...)]
        # We assume the actual JSON response is inside one of these text chunks.
        import re
        
        # Regex to capture the content inside text='...' or text="..."
        # Handles escaped quotes within the content
        text_chunk_pattern = r"text=['\"](.*?)['\"](?=[,)\]])"
        chunk_matches = re.findall(text_chunk_pattern, full_text)
        
        if chunk_matches:
            # If we found chunks, join them. This removes the "[ThinkChunk...]" wrapper entirely.
            logger.info(f"Found {len(chunk_matches)} text chunks. Joining them.")
            cleaned_text = " ".join(chunk_matches)
            # Unescape python string escapes (e.g. \\n -> \n, \\" -> ")
            cleaned_text = cleaned_text.replace("\\'", "'").replace('\\"', '"').replace("\\n", "\n")
            full_text = cleaned_text
        else:
            # If no specific TextChunk pattern found, try to clean the raw string
            # unescape anyway just in case it's a raw repr() string
            full_text = full_text.replace("\\'", "'").replace('\\"', '"').replace("\\n", "\n")

        # 3. Find JSON Object
        # Look for { ... "intent" ... }
        json_pattern = r'\{[^{}]*?[\"\']intent[\"\'][^{}]*?\}'
        matches = re.findall(json_pattern, full_text, re.DOTALL)
        
        result_json = None
        
        if matches:
            # Try the last match first (often the final answer)
            for match in reversed(matches):
                try:
                    result_json = json.loads(match)
                    break
                except:
                    import ast
                    try:
                        result_json = ast.literal_eval(match)
                        break
                    except:
                        continue

        if not result_json:
            logger.warning("Still no valid JSON found after cleaning.")
            return {
                "intent": "UNKNOWN", 
                "confidence": 0.0, 
                "model_used": gen_model, 
                "error": "JSON extraction failed",
                "debug_raw": str(response_text)[:100] # Return start of raw text for client debug
            }

        intent = result_json.get("intent", "UNKNOWN").upper()
        confidence = float(result_json.get("confidence", 0.5))
        
        return {
            "intent": intent,
            "confidence": confidence,
            "model_used": gen_model
        }

    except Exception as e:
        logger.error(f"Intent detection error: {e}", exc_info=True)
        return {"intent": "UNKNOWN", "confidence": 0.0, "model_used": gen_model, "error": str(e)}

@app.get("/health")
async def health_check():
    """Health check endpoint with server status."""
    if generator:
        available_models = generator.get_available_models()
        server_metrics = generator.get_server_metrics()
    else:
        available_models = []
        server_metrics = None

    return {
        "status": "ok",
        "generator_initialized": generator is not None,
        "scene_graph_initialized": scene_graph_generator is not None,
        "gateway_url": GATEWAY_BASE_URL,
        "available_models": available_models,
        "gateway_metrics": server_metrics,
        "default_model": DEFAULT_MODEL
    }

@app.get("/status")
async def status_endpoint():
    """Detailed server status including gateway information."""
    if generator:
        generator.print_server_status()
        return {"message": "Server status printed to logs"}
    else:
        raise HTTPException(status_code=503, detail="Generator not initialized")

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)