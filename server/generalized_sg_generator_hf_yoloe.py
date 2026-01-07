import os
import cv2
import numpy as np
from PIL import Image
import torch
import torchvision
import time
import json
import re
import warnings
import requests
import io
from mmengine import Config
import gc
import shutil
from datetime import datetime
from math import pi
from abc import ABC, abstractmethod
from typing import Optional, Dict, Tuple, List, Union
import base64
from sklearn.metrics.pairwise import cosine_similarity
import logging
from scipy.spatial import cKDTree  # <--- Added for fast density filter

import trimesh
from unik3d.models import UniK3D
from unik3d.utils.camera import OPENCV, Fisheye624, Pinhole, Spherical
import open3d as o3d
from wis3d import Wis3D
import matplotlib
from scipy.spatial.transform import Rotation
from collections import Counter

from ultralytics import YOLOE
from osdsynth.processor.captions import CaptionImage
from osdsynth.processor.pointcloud import PointCloudReconstruction
from osdsynth.processor.prompt import PromptGenerator as QAPromptGenerator
from osdsynth.processor.instruction import PromptGenerator as FactPromptGenerator
from osdsynth.utils.logger import SkipImageException, setup_logger

# Configure logging
logging.basicConfig(filename='llm_prompts.log', level=logging.INFO, format='%(asctime)s - %(message)s')

OSDSYNTH_AVAILABLE = True

# HuggingFace Transformers (optional)
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    import sys
    HF_TRANSFORMERS_AVAILABLE = True
except ImportError:
    HF_TRANSFORMERS_AVAILABLE = False
    warnings.warn("Hugging Face Transformers not found. Local LLM will not be available.")

# --- Helper: Image to Base64 ---
def encode_image_to_base64(image_input: Union[str, np.ndarray, Image.Image]) -> str:
    """Convert image to base64 string for VLM API calls."""
    try:
        if isinstance(image_input, str):
            with open(image_input, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        
        pil_image = None
        if isinstance(image_input, np.ndarray):
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
        elif isinstance(image_input, Image.Image):
            pil_image = image_input
        
        if pil_image:
            buffered = io.BytesIO()
            pil_image.save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
            
    except Exception as e:
        print(f"Error encoding image: {e}")
    return ""
def fast_density_filter(pcd, k=20, density_threshold=0.3):
    """
    Approximate density-based filter (faster replacement for remove_radius_outlier).
    Keeps points that have sufficiently high neighbor density.
    """
    if not pcd.has_points():
        return pcd

    points = np.asarray(pcd.points)
    
    try:
        tree = cKDTree(points)
        dists, _ = tree.query(points, k=k)
        # Average distance to k nearest neighbors
        mean_dists = np.mean(dists[:, 1:], axis=1)
        # Density approximation (inverse distance)
        density = 1.0 / (mean_dists + 1e-8)
        
        # Calculate dynamic threshold if not absolute
        threshold = np.mean(density) * density_threshold
        
        mask = density > threshold
        # print(f"⚡ Fast density filter kept {np.sum(mask):,}/{len(points):,} points")
        
        filtered = pcd.select_by_index(np.where(mask)[0])
        return filtered
    except Exception as e:
        print(f"Warning: fast_density_filter failed: {e}")
        return pcd

def process_pcd_for_unik3d(cfg, pcd):
    """
    Cleans a point cloud using Adaptive Statistical Outlier Removal 
    followed by the Fast Density Filter and Normal Estimation.
    
    This function is now applied to every detected object extracted from the scene.
    """
    if not pcd.has_points() or len(pcd.points) < 10: 
        return pcd

    points = np.asarray(pcd.points)
    n_points = len(points)
    
    # --- 1. Calculate Adaptive Parameters ---
    try:
        bounds = pcd.get_axis_aligned_bounding_box()
        size = bounds.get_max_bound() - bounds.get_min_bound()
        diag_len = np.linalg.norm(size)
        
        # Adaptive rules from the algorithm
        # nb_neighbors: Scale with points, clipped between 20 and 60
        nb_neighbors = int(np.clip(n_points * 0.001, 20, 60))
        
        # std_ratio: Scale with size (larger objects tolerate more variance)
        std_ratio = np.clip(diag_len / 5.0, 1.0, 3.0)
        
    except Exception:
        nb_neighbors = 20
        std_ratio = 1.5
        diag_len = 1.0

    # --- 2. Statistical Outlier Removal (Adaptive) ---
    try:
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    except RuntimeError:
        pass
    
    if not pcd.has_points(): return pcd

    # --- 3. Fast Density Filter ---
    # Using the algorithm provided. density_threshold=0.4 is the value from the main block
    pcd = fast_density_filter(pcd, k=20, density_threshold=cfg.get("density_threshold", 0.4))
    
    if not pcd.has_points(): return pcd

    # --- 4. Normal Estimation & Orientation ---
    # Essential for downstream oriented bounding box (OBB) calculation
    try:
        radius_normal = diag_len * 0.03
        if radius_normal <= 0: radius_normal = 0.1
        
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius_normal, max_nn=50
        ))
        
        # Orient normals consistently. 
        # Since we don't have a single camera origin for the whole scene easily accessible here,
        # we orient towards the bounding box center (standard assumption for objects)
        pcd.orient_normals_to_align_with_direction(orientation_reference=np.array([0., 0., 0.]))
        
        # Or standard tangent plane consistency
        # pcd.orient_normals_consistent_tangent_plane(10)
    except Exception as e:
        # If normal estimation fails, we proceed without normals
        pass

    return pcd

def pcd_denoise_dbscan_for_unik3d(pcd: o3d.geometry.PointCloud, eps=0.05, min_points=10) -> o3d.geometry.PointCloud:
    """Denoise point cloud using DBSCAN."""
    if not pcd.has_points() or len(pcd.points) < min_points: 
        return pcd
    try:
        labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    except RuntimeError:
        return pcd
    
    counts = Counter(labels)
    if -1 in counts: 
        del counts[-1]
    if not counts: 
        return o3d.geometry.PointCloud()
    
    largest_cluster_label = counts.most_common(1)[0][0]
    largest_cluster_indices = np.where(labels == largest_cluster_label)[0]
    if len(largest_cluster_indices) < min_points: 
        return o3d.geometry.PointCloud()
    
    return pcd.select_by_index(largest_cluster_indices)


# --- LLM Interface (Abstract Base Class) ---
class LLMInterface(ABC):
    """Abstract interface for LLM interactions."""
    
    @abstractmethod
    async def generate_answer(self, context_or_messages: Union[str, List[Dict]], 
                             question: Optional[str] = None, logger=None) -> str:
        """Generate an answer based on context and question, or from message history.
        
        Args:
            context_or_messages: Either a context string OR a list of message dicts with 'role' and 'content'
            question: Question string (used if context_or_messages is a string)
            logger: Optional logger instance
            
        Returns:
            Generated answer string
        """
        pass

    @abstractmethod
    def generate_json_qa(self, user_prompt: str, system_prompt: str, logger=None) -> Optional[Dict]:
        """Generate a JSON response (used for fact rephrasing)."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if LLM is available and initialized."""
        pass

# --- External API LLM Implementation ---

class ExternalAPILLM(LLMInterface):
    """LLM interface for external API servers with extended thinking support."""
    
    def __init__(self, api_url: str, model_name: str, api_key: Optional[str] = None, 
                custom_headers: Optional[Dict] = None, thinking_budget: Optional[int] = None,
                logger=None):
        """
        Args:
            api_url: API endpoint URL
            model_name: Model identifier
            api_key: Optional API key
            custom_headers: Optional custom headers
            thinking_budget: Optional thinking budget (for Gemini extended thinking)
                        Set to None to disable, or use a value like 10000-100000
            logger: Optional logger instance
        """
        self.api_url = api_url.rstrip("/")
        self.model_name = model_name
        self.api_key = api_key
        self.logger = logger
        self.thinking_budget = thinking_budget
        self.chat_endpoint = f"{self.api_url}/v1/chat/completions"
        self.prompt_logger = None  # <-- ADD THIS LINE
        
        self.headers = {"Content-Type": "application/json"}
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"
        if custom_headers:
            self.headers.update(custom_headers)




    async def generate_answer(self, context_or_messages: Union[str, List[Dict]], 
                            question: Optional[str] = None, logger=None) -> str:
        """Generate answer using external API."""
        current_logger = logger or self.logger
        request_id = None
        start_time = time.time()
        
        # Handle both legacy (context, question) and new (messages list) formats
        if isinstance(context_or_messages, list):
            messages = context_or_messages
        elif isinstance(context_or_messages, str) and question is not None:
            qa_system_prompt = (
                "You are an intelligent assistant helping a visually impaired person understand their surroundings. "
                "Based on the provided scene summary, answer the user's question concisely and helpfully."
            )
            messages = [
                {"role": "system", "content": qa_system_prompt},
                {"role": "user", "content": f"Scene Summary:\n{context_or_messages}\n\nQuestion: {question}"}
            ]
        else:
            raise ValueError("Must provide either messages list OR (context, question) tuple")
        
        # Filter out non-text content from messages
        filtered_messages = []
        for msg in messages:
            if isinstance(msg.get("content"), str):
                filtered_messages.append(msg)
            elif isinstance(msg.get("content"), list):
                text_content = ""
                for item in msg["content"]:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_content += item.get("text", "")
                if text_content:
                    filtered_messages.append({"role": msg.get("role"), "content": text_content})
            else:
                filtered_messages.append(msg)
        
        payload = {
            "model": self.model_name,
            "messages": filtered_messages,
            "max_tokens": 2048,
            "temperature": 0.6,
            "top_p": 0.9
        }
        
        # Log the request if prompt logger is available
        if self.prompt_logger:
            request_id = self.prompt_logger.log_external_api_request(
                model_name=self.model_name,
                messages=filtered_messages,
                max_tokens=payload.get('max_tokens'),
                temperature=payload.get('temperature'),
                thinking_budget=self.thinking_budget,
                context="QA Generation",
                logger=current_logger
            )
        
        # Add thinking budget if specified (Gemini API)
        if self.thinking_budget is not None:
            payload["generationConfig"] = {
                "thinkingConfig": {
                    "thinkingBudget": self.thinking_budget
                }
            }
        
        try:
            if current_logger:
                current_logger.info(f"Sending QA to external LLM API: {self.api_url}")
                if self.thinking_budget:
                    current_logger.info(f"Using thinking budget: {self.thinking_budget}")
            
            response = requests.post(self.chat_endpoint, json=payload, headers=self.headers, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            
            # Handle various API response formats robustly
            answer = self._extract_content_from_response(result)
            
            if not answer:
                print(result)
                if self.prompt_logger and request_id:
                    latency = time.time() - start_time
                    self.prompt_logger.log_external_api_response(
                        request_id=request_id,
                        response="",
                        latency_seconds=latency,
                        status='error',
                        error_message="Empty response received",
                        logger=current_logger
                    )
                return "The AI could not provide a response."
            
            # Try to salvage incomplete responses (e.g., due to token limits)
            answer = self._salvage_incomplete_response(answer, current_logger)
            
            # Log successful response
            if self.prompt_logger and request_id:
                latency = time.time() - start_time
                self.prompt_logger.log_external_api_response(
                    request_id=request_id,
                    response=answer,
                    latency_seconds=latency,
                    status='success',
                    logger=current_logger
                )
            
            if current_logger:
                current_logger.info(f"LLM API Answer: {answer}")
            return answer
            
        except requests.exceptions.ConnectionError as e:
            if current_logger:
                current_logger.error(f"Failed to connect to LLM API: {e}")
            if self.prompt_logger and request_id:
                latency = time.time() - start_time
                self.prompt_logger.log_external_api_response(
                    request_id=request_id,
                    response="",
                    latency_seconds=latency,
                    status='error',
                    error_message=f"Connection Error: {e}",
                    logger=current_logger
                )
            raise Exception(f"LLM Connection Error: {e}")
        except requests.exceptions.HTTPError as e:
            error_msg = f"HTTP {response.status_code}"
            if response.status_code == 401:
                if current_logger:
                    current_logger.error("LLM API authentication failed (401). Check your API key.")
                error_msg = "Authentication failed (401)"
                if self.prompt_logger and request_id:
                    latency = time.time() - start_time
                    self.prompt_logger.log_external_api_response(
                        request_id=request_id,
                        response="",
                        latency_seconds=latency,
                        status='error',
                        error_message=error_msg,
                        logger=current_logger
                    )
                raise Exception(f"LLM Authentication Error (401): Invalid API key")
            elif response.status_code == 429:
                if current_logger:
                    current_logger.error("LLM API rate limit exceeded (429).")
                error_msg = "Rate limit exceeded (429)"
                if self.prompt_logger and request_id:
                    latency = time.time() - start_time
                    self.prompt_logger.log_external_api_response(
                        request_id=request_id,
                        response="",
                        latency_seconds=latency,
                        status='error',
                        error_message=error_msg,
                        logger=current_logger
                    )
                # RE-RAISE for retry handler to catch
                raise Exception(f"LLM API Rate Limit (429): {response.text}")
            else:
                if current_logger:
                    current_logger.error(f"LLM API HTTP error {response.status_code}")
                if self.prompt_logger and request_id:
                    latency = time.time() - start_time
                    self.prompt_logger.log_external_api_response(
                        request_id=request_id,
                        response="",
                        latency_seconds=latency,
                        status='error',
                        error_message=error_msg,
                        logger=current_logger
                    )
                raise Exception(f"LLM API HTTP Error {response.status_code}: {response.text}")
        except requests.exceptions.Timeout as e:
            if current_logger:
                current_logger.error("LLM API request timed out.")
            if self.prompt_logger and request_id:
                latency = time.time() - start_time
                self.prompt_logger.log_external_api_response(
                    request_id=request_id,
                    response="",
                    latency_seconds=latency,
                    status='timeout',
                    error_message="Request timeout",
                    logger=current_logger
                )
            raise Exception(f"LLM API Timeout: {e}")
        except Exception as e:
            if current_logger:
                current_logger.error(f"External LLM API error: {e}", exc_info=True)
            if self.prompt_logger and request_id:
                latency = time.time() - start_time
                self.prompt_logger.log_external_api_response(
                    request_id=request_id,
                    response="",
                    latency_seconds=latency,
                    status='error',
                    error_message=str(e),
                    logger=current_logger
                )
            # Re-raise all exceptions so retry handler can decide what to do
            raise
        
    def _salvage_incomplete_response(self, text: str, logger=None) -> str:
        if not text or len(text.strip()) == 0:
            return "The AI could not provide a response."
        
        text = text.strip()
        
        # Common incomplete patterns and fixes
        incomplete_patterns = [
            (r'\.\.\.$', '.'),  # Replace ... at end with period
            (r'[,;]\s*$', '.'),  # Replace trailing comma/semicolon with period
            (r'\s+(and|or|but|because|however|therefore|thus|such as)\s*$', '.'),  # Remove dangling conjunctions
            (r'\s+\([^)]*$', '.'),  # Remove unclosed parenthetical
            (r'\s+\[[^\]]*$', '.'),  # Remove unclosed bracket
            (r'\s+["\']$', '.'),  # Remove unclosed quotes
        ]
        
        for pattern, replacement in incomplete_patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # If text ends with lowercase followed by no punctuation, add period
        if text and text[-1].isalnum() and not text.endswith(('.', '!', '?')):
            text += '.'
        
        # Remove orphaned single words at the end (likely incomplete)
        words = text.split()
        if len(words) > 1:
            # Check if last word is unusually short and preceded by a comma or incomplete phrase
            if len(words[-1]) < 4 and text.endswith(words[-1]) and (
                ',' in text[-20:] or text[-20:].count(' ') > 2
            ):
                # Keep it, but ensure the sentence before it is complete
                text = ' '.join(words[:-1])
                if text and not text.endswith(('.', '!', '?')):
                    text += '.'
        
        if logger:
            logger.info(f"Salvaged incomplete response. Length: {len(text)} chars")
        
        return text if text else "The AI could not provide a response."

    def _extract_content_from_response(self, result: dict, logger=None) -> str:
        """
        Robustly extract content from various API response formats.
        Handles standard OpenAI, Gemini, and other formats.
        """
        try:
            # Try standard OpenAI format: result["choices"][0]["message"]["content"]
            if "choices" in result and isinstance(result["choices"], list) and len(result["choices"]) > 0:
                choice = result["choices"][0]
                
                # Standard format
                if "message" in choice and isinstance(choice["message"], dict):
                    content = choice["message"].get("content", "")
                    if content:
                        return content.strip()
                
                # Alternative: content directly in choice
                if "content" in choice:
                    content = choice["content"]
                    if content:
                        return content.strip()
                
                # Alternative: text field
                if "text" in choice:
                    content = choice["text"]
                    if content:
                        return content.strip()
            
            # Try Gemini format: result["candidates"][0]["content"]["parts"][0]["text"]
            if "candidates" in result and isinstance(result["candidates"], list) and len(result["candidates"]) > 0:
                candidate = result["candidates"][0]
                
                if "content" in candidate and isinstance(candidate["content"], dict):
                    parts = candidate["content"].get("parts", [])
                    if parts and isinstance(parts, list):
                        for part in parts:
                            if isinstance(part, dict) and "text" in part:
                                text = part["text"]
                                if text:
                                    return text.strip()
            
            # Try generic "content" field
            if "content" in result:
                content = result["content"]
                if isinstance(content, str):
                    return content.strip()
                elif isinstance(content, dict) and "text" in content:
                    return content["text"].strip()
            
            # Try "text" field
            if "text" in result:
                text = result["text"]
                if isinstance(text, str):
                    return text.strip()
            
            # Log the structure for debugging
            if logger:
                logger.warning(f"Could not parse response structure. Keys: {list(result.keys())}")
            
            return ""
        
        except Exception as e:
            if logger:
                logger.error(f"Error extracting content from response: {e}", exc_info=True)
            return ""
    
    def generate_json_qa(self, user_prompt: str, system_prompt: str, logger=None) -> Optional[Dict]:
        """Generate JSON response via external API."""
        current_logger = logger or self.logger
        request_id = None
        start_time = time.time()
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": 2048,
            "temperature": 0.2,
            "top_p": 0.9
        }
        
        # Log the request
        if self.prompt_logger:
            request_id = self.prompt_logger.log_json_qa_request(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                model_name=self.model_name,
                context="JSON QA Generation",
                logger=current_logger
            )
        
        # Add thinking budget if specified
        if self.thinking_budget is not None:
            payload["generationConfig"] = {
                "thinkingConfig": {
                    "thinkingBudget": self.thinking_budget
                }
            }
        
        try:
            response = requests.post(self.chat_endpoint, json=payload, headers=self.headers, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            answer_text = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            
            if not answer_text:
                if self.prompt_logger and request_id:
                    latency = time.time() - start_time
                    self.prompt_logger.log_external_api_response(
                        request_id=request_id,
                        response="",
                        latency_seconds=latency,
                        status='error',
                        error_message="Empty response",
                        logger=current_logger
                    )
                return None
            
            parsed_json = self._parse_json_response(answer_text, current_logger)
            
            if self.prompt_logger and request_id:
                latency = time.time() - start_time
                self.prompt_logger.log_external_api_response(
                    request_id=request_id,
                    response=json.dumps(parsed_json) if parsed_json else answer_text,
                    latency_seconds=latency,
                    status='success',
                    logger=current_logger
                )
            
            return parsed_json
            
        except Exception as e:
            if current_logger:
                current_logger.error(f"Error generating JSON QA via API: {e}")
            if self.prompt_logger and request_id:
                latency = time.time() - start_time
                self.prompt_logger.log_external_api_response(
                    request_id=request_id,
                    response="",
                    latency_seconds=latency,
                    status='error',
                    error_message=str(e),
                    logger=current_logger
                )
            return None
    
    def _parse_json_response(self, text: str, logger=None) -> Optional[Dict]:
        """Extract and parse JSON from LLM response."""
        try:
            match_json_block = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
            if match_json_block:
                json_str = match_json_block.group(1)
            else:
                first_brace = text.find('{')
                last_brace = text.rfind('}')
                if first_brace != -1 and last_brace != -1:
                    json_str = text[first_brace:last_brace + 1]
                else:
                    return None
            
            return json.loads(json_str)
        except Exception as e:
            if logger:
                logger.warning(f"Could not parse JSON response: {e}")
            return None
    
    def is_available(self) -> bool:
        """Check if external API is reachable."""
        try:
            response = requests.get(f"{self.api_url}/health", headers=self.headers, timeout=5)
            return response.status_code == 200
        except Exception:
            return False
                     
# --- Local HuggingFace LLM Implementation ---
class LocalHFLLM(LLMInterface):
    """LLM interface for locally hosted HuggingFace models."""
    
    def __init__(self, model_name: str, device: str, logger=None):
        self.model_name = model_name
        self.device = device
        self.logger = logger
        self.pipeline = None
        self.tokenizer = None
        self._initialize()
    
    def _initialize(self):
        """Initialize the HuggingFace pipeline and tokenizer."""
        try:
            if self.logger:
                self.logger.info(f"Initializing local LLM: {self.model_name} on {self.device}")
            
            self.pipeline = pipeline(
                "text2text-generation",
                model=self.model_name,
                device=0 if self.device == "cuda" else -1,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            if self.logger:
                self.logger.info(f"Local LLM initialized successfully.")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to initialize local LLM: {e}", exc_info=True)
            self.pipeline = None
            self.tokenizer = None
    
    async def generate_answer(self, context_or_messages: Union[str, List[Dict]], 
                             question: Optional[str] = None, logger=None) -> str:
        """Generate answer using local LLM."""
        if not self.is_available():
            return "Local LLM model is not available."
        
        # Handle both legacy (context, question) and new (messages list) formats
        if isinstance(context_or_messages, list):
            # New format: messages list passed directly
            messages = context_or_messages
        elif isinstance(context_or_messages, str) and question is not None:
            # Legacy format: context and question strings
            qa_system_prompt = (
                "You are an intelligent assistant helping a visually impaired person understand their surroundings. "
                "Based on the provided scene summary, answer the user's question concisely and helpfully."
            )
            messages = [
                {"role": "system", "content": qa_system_prompt},
                {"role": "user", "content": f"Scene Summary:\n{context_or_messages}\n\nQuestion: {question}"}
            ]
        else:
            raise ValueError("Must provide either messages list OR (context, question) tuple")
        
        try:
            if hasattr(self.tokenizer, 'apply_chat_template'):
                prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                prompt = f"{qa_system_prompt}\n{context}\n{question}"
            
            generated = self.pipeline(
                prompt,
                max_length=2048,
                temperature=0.6,
                do_sample=True,
                repetition_penalty=1.1
            )
            
            answer = generated[0].get("generated_text", "").strip() if generated else ""
            if answer.startswith(prompt):
                answer = answer[len(prompt):].strip()
            
            for eos_str in ["<|im_end|>", "<|eot_id|>", "</s>"]:
                if answer.endswith(eos_str):
                    answer = answer[:-len(eos_str)].strip()
            
            return answer if answer else "The AI could not provide a specific answer."
            
        except Exception as e:
            if logger or self.logger:
                (logger or self.logger).error(f"Local LLM error: {e}", exc_info=True)
            return "An error occurred while generating an answer."
    
    def generate_json_qa(self, user_prompt: str, system_prompt: str, logger=None) -> Optional[Dict]:
        """Generate JSON response via local LLM."""
        if not self.is_available():
            return None
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            if hasattr(self.tokenizer, 'apply_chat_template'):
                prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                prompt = f"{system_prompt}\n{user_prompt}"
            
            generated = self.pipeline(
                prompt,
                max_length=2048,
                temperature=0.2,
                do_sample=True,
                repetition_penalty=1.1
            )
            
            answer = generated[0].get("generated_text", "").strip() if generated else ""
            if answer.startswith(prompt):
                answer = answer[len(prompt):].strip()
            
            return self._parse_json_response(answer, logger or self.logger)
            
        except Exception as e:
            if logger or self.logger:
                (logger or self.logger).error(f"Error generating JSON QA via local LLM: {e}")
            return None
    
    def _parse_json_response(self, text: str, logger=None) -> Optional[Dict]:
        """Extract and parse JSON from LLM response."""
        try:
            match_json_block = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
            if match_json_block:
                json_str = match_json_block.group(1)
            else:
                first_brace = text.find('{')
                last_brace = text.rfind('}')
                if first_brace != -1 and last_brace != -1:
                    json_str = text[first_brace:last_brace + 1]
                else:
                    return None
            
            return json.loads(json_str)
        except Exception as e:
            if logger:
                logger.warning(f"Could not parse JSON response: {e}")
            return None
    
    def is_available(self) -> bool:
        """Check if local LLM is initialized."""
        return self.pipeline is not None and self.tokenizer is not None

# --- Helper functions ---
def blackout_nonmasked_area(image_pil: Image.Image, mask_crop: np.ndarray) -> Image.Image:
    """Blacks out the non-masked area of a cropped image."""
    image_np = np.array(image_pil)
    if image_np.shape[:2] != mask_crop.shape:
        warnings.warn(f"Shape mismatch in blackout. Skipping.")
        return image_pil
    
    mask_bool = mask_crop.astype(bool)
    if image_np.ndim == 3 and image_np.shape[2] >= 3:
        mask_3channel = np.stack([mask_bool]*image_np.shape[2], axis=-1)
    else: 
        mask_3channel = mask_bool

    blacked_out_image_np = np.where(mask_3channel, image_np, 0)
    return Image.fromarray(blacked_out_image_np.astype(np.uint8))

def draw_red_outline(image_pil: Image.Image, mask_crop: np.ndarray, outline_thickness: int = 2) -> Image.Image:
    """Draws a red outline around the masked area."""
    image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    if image_cv.shape[:2] != mask_crop.shape:
        warnings.warn(f"Shape mismatch. Skipping outline.")
        return image_pil

    mask_uint8 = mask_crop.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if image_cv.shape[2] == 4: 
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGBA2BGR)

    cv2.drawContours(image_cv, contours, -1, (0, 0, 255), outline_thickness) 
    return Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))

def crop_image_and_mask(image: Image, mask: np.ndarray, x1: int, y1: int, x2: int, y2: int, padding: int = 0):
    """Crop the image and mask with padding."""
    image_np_original = np.array(image)
    if image_np_original.shape[:2] != mask.shape:
        print(f"Critical: Shape mismatch: Image {image_np_original.shape[:2]} != Mask {mask.shape}")
        return None, None

    x1_pad = max(0, x1 - padding)
    y1_pad = max(0, y1 - padding)
    x2_pad = min(image_np_original.shape[1], x2 + padding)
    y2_pad = min(image_np_original.shape[0], y2 + padding)
    x1_pad, y1_pad, x2_pad, y2_pad = round(x1_pad), round(y1_pad), round(x2_pad), round(y2_pad)

    image_crop_np = image_np_original[y1_pad:y2_pad, x1_pad:x2_pad]
    mask_crop_np = mask[y1_pad:y2_pad, x1_pad:x2_pad]

    if image_crop_np.shape[:2] != mask_crop_np.shape:
        print(f"Cropped shape mismatch")
        return None, None
    
    image_crop_pil = Image.fromarray(image_crop_np)
    return image_crop_pil, mask_crop_np

def visualize_bboxes_in_wis3d(wis3d_instance, valid_detections_dicts):
    """
    Add 3D bounding boxes to Wis3D visualization.
    
    Args:
        wis3d_instance: Wis3D visualization instance
        valid_detections_dicts: List of detection dictionaries with bbox info
    """
    import matplotlib
    import numpy as np
    cmap = matplotlib.colormaps.get_cmap("turbo")
    
    for i, det_dict in enumerate(valid_detections_dicts):
        class_name = det_dict.get("class_name", f"object_{i}")
        color = cmap(i / max(len(valid_detections_dicts), 1))[:3]
        
        # Try oriented bbox first, fall back to axis-aligned
        obb = det_dict.get("oriented_bbox")
        aabb = det_dict.get("axis_aligned_bbox")
        
        if obb and not obb.is_empty():
            # Get oriented bounding box corners
            bbox_points = np.asarray(obb.get_box_points())
            
            # Define the 12 edges of a box (connecting 8 corners)
            edges = [
                [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
                [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
                [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical edges
            ]
            
            # Collect all start and end points for this bbox
            start_points = []
            end_points = []
            for start_idx, end_idx in edges:
                start_points.append(bbox_points[start_idx])
                end_points.append(bbox_points[end_idx])
            
            start_points = np.array(start_points)
            end_points = np.array(end_points)
            
            # Add all edges as lines with single call
            wis3d_instance.add_lines(
                start_points,
                end_points,
                colors=np.tile(color, (len(edges), 1)),
                name=f"{i:02d}_{class_name}_bbox"
            )
        
        elif aabb and not aabb.is_empty():
            # For axis-aligned bbox, get min/max corners
            min_bound = np.asarray(aabb.get_min_bound())
            max_bound = np.asarray(aabb.get_max_bound())
            
            # Create 8 corners of axis-aligned box
            bbox_points = np.array([
                [min_bound[0], min_bound[1], min_bound[2]],
                [max_bound[0], min_bound[1], min_bound[2]],
                [max_bound[0], max_bound[1], min_bound[2]],
                [min_bound[0], max_bound[1], min_bound[2]],
                [min_bound[0], min_bound[1], max_bound[2]],
                [max_bound[0], min_bound[1], max_bound[2]],
                [max_bound[0], max_bound[1], max_bound[2]],
                [min_bound[0], max_bound[1], max_bound[2]]
            ])
            
            edges = [
                [0, 1], [1, 2], [2, 3], [3, 0],
                [4, 5], [5, 6], [6, 7], [7, 4],
                [0, 4], [1, 5], [2, 6], [3, 7]
            ]
            
            # Collect all start and end points
            start_points = []
            end_points = []
            for start_idx, end_idx in edges:
                start_points.append(bbox_points[start_idx])
                end_points.append(bbox_points[end_idx])
            
            start_points = np.array(start_points)
            end_points = np.array(end_points)
            
            wis3d_instance.add_lines(
                start_points,
                end_points,
                colors=np.tile(color, (len(edges), 1)),
                name=f"{i:02d}_{class_name}_bbox",
                width=30.0  # <--- Add this parameter to make lines thicker
            )


def export_scene_with_bboxes_to_ply(output_path, global_points, global_colors, 
                                     valid_detections_dicts):
    """
    Export the global point cloud with 3D bounding boxes to a PLY file.
    
    Args:
        output_path: Path to save PLY file
        global_points: Nx3 array of global scene points
        global_colors: Nx3 array of RGB colors (0-1 range)
        valid_detections_dicts: List of detection dictionaries
    """
    import open3d as o3d
    import matplotlib
    
    # Create combined point cloud
    combined_pcd = o3d.geometry.PointCloud()
    combined_pcd.points = o3d.utility.Vector3dVector(global_points)
    combined_pcd.colors = o3d.utility.Vector3dVector(global_colors)
    
    # Add bounding box edges as line segments
    cmap = matplotlib.colormaps.get_cmap("turbo")
    all_bbox_points = []
    all_bbox_colors = []
    
    for i, det_dict in enumerate(valid_detections_dicts):
        color = cmap(i / max(len(valid_detections_dicts), 1))[:3]
        
        obb = det_dict.get("oriented_bbox")
        aabb = det_dict.get("axis_aligned_bbox")
        
        if obb and not obb.is_empty():
            bbox_points = np.asarray(obb.get_box_points())
        elif aabb and not aabb.is_empty():
            min_bound = np.asarray(aabb.get_min_bound())
            max_bound = np.asarray(aabb.get_max_bound())
            bbox_points = np.array([
                [min_bound[0], min_bound[1], min_bound[2]],
                [max_bound[0], min_bound[1], min_bound[2]],
                [max_bound[0], max_bound[1], min_bound[2]],
                [min_bound[0], max_bound[1], min_bound[2]],
                [min_bound[0], min_bound[1], max_bound[2]],
                [max_bound[0], min_bound[1], max_bound[2]],
                [max_bound[0], max_bound[1], max_bound[2]],
                [min_bound[0], max_bound[1], max_bound[2]]
            ])
        else:
            continue
        
        # Create dense line segments for each edge
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7]
        ]
        
        for start, end in edges:
            # Interpolate points along edge for visibility
            num_points = 20
            t = np.linspace(0, 1, num_points)
            edge_points = bbox_points[start] + t[:, None] * (bbox_points[end] - bbox_points[start])
            all_bbox_points.append(edge_points)
            all_bbox_colors.append(np.tile(color, (num_points, 1)))
    
    # Add bbox edge points to main point cloud
    if all_bbox_points:
        bbox_points_array = np.vstack(all_bbox_points)
        bbox_colors_array = np.vstack(all_bbox_colors)
        
        combined_points = np.vstack([
            np.asarray(combined_pcd.points),
            bbox_points_array
        ])
        combined_colors = np.vstack([
            np.asarray(combined_pcd.colors),
            bbox_colors_array
        ])
        
        combined_pcd.points = o3d.utility.Vector3dVector(combined_points)
        combined_pcd.colors = o3d.utility.Vector3dVector(combined_colors)
    
    # Save to PLY
    o3d.io.write_point_cloud(output_path, combined_pcd)
    print(f"Saved scene with bounding boxes to: {output_path}")

def crop_detections_with_xyxy(cfg, image_pil, detections_list):
    """Crop detections with XYXY coordinates."""
    for idx, detection in enumerate(detections_list):
        x1, y1, x2, y2 = detection["xyxy"]
        if image_pil.size[1] != detection["mask"].shape[0] or image_pil.size[0] != detection["mask"].shape[1]:
            resized_mask = cv2.resize(detection["mask"].astype(np.uint8), 
                                     (image_pil.size[0], image_pil.size[1]), 
                                     interpolation=cv2.INTER_NEAREST)
            detection["mask"] = resized_mask.astype(bool)

        image_crop, mask_crop = crop_image_and_mask(image_pil, detection["mask"], 
                                                    int(x1), int(y1), int(x2), int(y2), 
                                                    padding=cfg.get("crop_padding", 10))
        
        if image_crop is None or mask_crop is None:
            detections_list[idx]["image_crop"] = None
            detections_list[idx]["mask_crop"] = None
            detections_list[idx]["image_crop_modified"] = None
            continue

        if cfg.get("masking_option") == "blackout":
            image_crop_modified = blackout_nonmasked_area(image_crop, mask_crop)
        elif cfg.get("masking_option") == "red_outline":
            image_crop_modified = draw_red_outline(image_crop, mask_crop)
        else: 
            image_crop_modified = image_crop 
        
        detections_list[idx]["image_crop"] = image_crop
        detections_list[idx]["mask_crop"] = mask_crop 
        detections_list[idx]["image_crop_modified"] = image_crop_modified
    
    return detections_list

def add_camera_frustum_at_origin(wis3d_inst, target_center, name="camera_origin", scale=0.1):
    """
    Draws a camera frustum at (0,0,0) and a line connecting it to the target object center.
    Assumes points are in Camera Coordinates (Z-forward).
    """
    # 1. Define frustum corners (Pyramid shape at origin)
    # Standard OpenCV Camera frame: X Right, Y Down, Z Forward
    origin = np.array([0, 0, 0])
    
    # Frustum plane corners at distance 'scale'
    w, h = scale, scale * 0.75  # 4:3 aspect ratio
    z = scale
    
    tl = np.array([-w, -h, z]) # Top-Left
    tr = np.array([w, -h, z])  # Top-Right
    br = np.array([w, h, z])   # Bottom-Right
    bl = np.array([-w, h, z])  # Bottom-Left
    
    # 2. Draw Frustum Lines
    frustum_points = np.array([origin, tl, tr, br, bl])
    
    # Edges connecting origin to plane
    edges_origin = [[0, 1], [0, 2], [0, 3], [0, 4]]
    # Edges connecting plane corners
    edges_plane = [[1, 2], [2, 3], [3, 4], [4, 1]]
    
    all_edges = edges_origin + edges_plane
    
    start_points = []
    end_points = []
    
    for s, e in all_edges:
        start_points.append(frustum_points[s])
        end_points.append(frustum_points[e])
        
    # Draw Frustum with THICKNESS
    wis3d_inst.add_lines(
        np.array(start_points),
        np.array(end_points),
        colors=np.tile(np.array([0, 1, 1]), (len(start_points), 1)),
        name=f"{name}_frustum",
        width=50.0  # <--- Increased Line Width (Default is usually 1.0)
    )

    # Draw Viewing Ray with THICKNESS
    wis3d_inst.add_lines(
        np.array([origin]),
        np.array([target_center]),
        colors=np.array([[1, 0, 1]]),
        name=f"{name}_viewing_ray",
        width=100.0 # <--- Even thicker for the ray
    )
def generate_solid_tube_points(start_pt, end_pt, color, radius=0.02, density_per_meter=5000):
    """
    Generates a dense, solid cylinder of points between start_pt and end_pt.
    """
    vec = end_pt - start_pt
    length = np.linalg.norm(vec)
    
    if length < 1e-6:
        return np.array([]), np.array([])

    direction = vec / length
    
    # Calculate number of points based on length to ensure consistent density
    num_points = int(length * density_per_meter)
    num_points = max(num_points, 100) # Minimum points per edge

    # 1. Create an arbitrary coordinate system (basis) around the direction vector
    # Find a vector not parallel to direction
    not_v = np.array([1, 0, 0])
    if abs(np.dot(direction, not_v)) > 0.9:
        not_v = np.array([0, 1, 0])
        
    n1 = np.cross(direction, not_v)
    n1 /= np.linalg.norm(n1)
    n2 = np.cross(direction, n1)

    # 2. Generate random cylindrical coordinates
    # t: distance along the line (0 to length)
    t = np.random.uniform(0, length, num_points)
    # theta: angle around the line (0 to 2pi)
    theta = np.random.uniform(0, 2 * np.pi, num_points)
    # r: distance from center (0 to radius). 
    # sqrt() ensures uniform distribution in the circle area (solid look)
    r = np.sqrt(np.random.uniform(0, 1, num_points)) * radius

    # 3. Convert to 3D coordinates
    # P = Start + (Forward * t) + (Right * r * cos) + (Up * r * sin)
    points = (start_pt + 
              np.outer(t, direction) + 
              np.outer(r * np.cos(theta), n1) + 
              np.outer(r * np.sin(theta), n2))
    
    colors = np.tile(color, (num_points, 1))
    
    return points, colors

def generate_dense_edge_points(start_pt, end_pt, color, num_points=200, thickness=0.01):
    """
    Generates a dense cylinder of points between start and end to simulate a thick line in PLY.
    """
    # 1. Linear interpolation (the spine of the line)
    t = np.linspace(0, 1, num_points)
    spine_points = start_pt + t[:, None] * (end_pt - start_pt)
    
    # 2. Add Gaussian noise to create thickness/volume
    # We generate random offsets perpendicular to the line would be ideal, 
    # but simple spherical noise is usually sufficient for visualization 
    # and much faster to compute.
    noise = np.random.normal(scale=thickness, size=spine_points.shape)
    
    # Apply noise to create "thickness"
    dense_points = spine_points + noise
    
    # Create colors array
    dense_colors = np.tile(color, (num_points, 1))
    
    return dense_points, dense_colors

def export_target_scene_to_ply(output_path, global_points, global_colors, 
                               valid_detections_dicts, target_class_name):
    """
    Exports the point cloud with THICK, SOLID TUBE bounding boxes.
    """
    import open3d as o3d
    
    # Create combined point cloud
    combined_pcd = o3d.geometry.PointCloud()
    combined_pcd.points = o3d.utility.Vector3dVector(global_points)
    combined_pcd.colors = o3d.utility.Vector3dVector(global_colors)
    
    bbox_points_list = []
    bbox_colors_list = []
    
    target_found = False
    
    for i, det_dict in enumerate(valid_detections_dicts):
        if det_dict.get("class_name") != target_class_name:
            continue
            
        target_found = True
        
        # Use bright Red for target
        color = np.array([1.0, 0.0, 0.0]) 
        
        obb = det_dict.get("oriented_bbox")
        if obb and not obb.is_empty():
            bbox_corners = np.asarray(obb.get_box_points())
            
            edges = [
                [0, 1], [1, 2], [2, 3], [3, 0],
                [4, 5], [5, 6], [6, 7], [7, 4],
                [0, 4], [1, 5], [2, 6], [3, 7]
            ]
            
            for start, end in edges:
                # --- UPDATED HERE ---
                # radius=0.03  -> 3cm thick tubes (very visible)
                # density=10000 -> 10k points per meter (very dense)
                edge_pts, edge_cols = generate_solid_tube_points(
                    bbox_corners[start], 
                    bbox_corners[end], 
                    color, 
                    radius=0.03,           # Adjust thickness here
                    density_per_meter=10000 # Adjust density here
                )
                
                if len(edge_pts) > 0:
                    bbox_points_list.append(edge_pts)
                    bbox_colors_list.append(edge_cols)
        
        break 

    if bbox_points_list:
        combined_points = np.vstack([np.asarray(combined_pcd.points)] + bbox_points_list)
        combined_colors = np.vstack([np.asarray(combined_pcd.colors)] + bbox_colors_list)
        
        combined_pcd.points = o3d.utility.Vector3dVector(combined_points)
        combined_pcd.colors = o3d.utility.Vector3dVector(combined_colors)
    
    o3d.io.write_point_cloud(output_path, combined_pcd)
    print(f"Saved scene with SOLID '{target_class_name}' BBox to: {output_path}")
class DetectedObject:
    """Represents a detected object with 2D and 3D information."""
    
    def __init__(self,
                 class_name: str,
                 description: str,
                 segmentation_mask_2d: np.ndarray,
                 bounding_box_2d: np.ndarray,
                 point_cloud_3d: o3d.geometry.PointCloud,
                 bounding_box_3d_oriented: o3d.geometry.OrientedBoundingBox,
                 bounding_box_3d_axis_aligned: o3d.geometry.AxisAlignedBoundingBox,
                 image_crop_pil: Image.Image = None):
        self.class_name = class_name
        self.description = description
        self.segmentation_mask_2d = segmentation_mask_2d 
        self.bounding_box_2d = bounding_box_2d
        self.point_cloud_3d = point_cloud_3d
        self.bounding_box_3d_oriented = bounding_box_3d_oriented
        self.bounding_box_3d_axis_aligned = bounding_box_3d_axis_aligned
        self.image_crop_pil = image_crop_pil

    @property
    def center(self) -> np.ndarray:
        """Calculate and return the 3D center of the bounding box."""
        if self.bounding_box_3d_oriented and not self.bounding_box_3d_oriented.is_empty():
            return np.asarray(self.bounding_box_3d_oriented.center)
        elif self.bounding_box_3d_axis_aligned and not self.bounding_box_3d_axis_aligned.is_empty():
            return np.asarray(self.bounding_box_3d_axis_aligned.get_center())
        elif self.point_cloud_3d and self.point_cloud_3d.has_points():
            return np.mean(np.asarray(self.point_cloud_3d.points), axis=0)
        else:
            return np.array([0, 0, 0])

    @property
    def volume(self) -> float:
        """Calculate and return the volume of the 3D bounding box."""
        if self.bounding_box_3d_oriented and not self.bounding_box_3d_oriented.is_empty():
            return float(self.bounding_box_3d_oriented.volume())
        elif self.bounding_box_3d_axis_aligned and not self.bounding_box_3d_axis_aligned.is_empty():
            return float(self.bounding_box_3d_axis_aligned.volume())
        else:
            return 0.0

    @property
    def color_rgb(self) -> np.ndarray:
        """Calculate and return the average color from the point cloud."""
        if self.point_cloud_3d and self.point_cloud_3d.has_colors():
            return np.mean(np.asarray(self.point_cloud_3d.colors), axis=0)
        else:
            return np.array([0.5, 0.5, 0.5])

    def __repr__(self):
        num_points = len(self.point_cloud_3d.points) if self.point_cloud_3d and self.point_cloud_3d.has_points() else 0
        return (f"<DetectedObject: {self.class_name} "
                f"(Desc: '{self.description[:30]}...'), "
                f"3D_pts: {num_points}>")


def pcd_denoise_dbscan_for_unik3d(pcd: o3d.geometry.PointCloud, eps=0.05, min_points=10) -> o3d.geometry.PointCloud:
    """Denoise point cloud using DBSCAN."""
    if not pcd.has_points() or len(pcd.points) < min_points: 
        return pcd
    try:
        labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    except RuntimeError:
        return pcd
    
    counts = Counter(labels)
    if -1 in counts: 
        del counts[-1]
    if not counts: 
        return o3d.geometry.PointCloud()
    
    largest_cluster_label = counts.most_common(1)[0][0]
    largest_cluster_indices = np.where(labels == largest_cluster_label)[0]
    if len(largest_cluster_indices) < min_points: 
        return o3d.geometry.PointCloud()
    
    return pcd.select_by_index(largest_cluster_indices)

def get_bounding_box_for_unik3d(cfg, pcd):
    """Get oriented and axis-aligned bounding boxes."""
    if not pcd.has_points() or len(pcd.points) < 3:
        aabb = o3d.geometry.AxisAlignedBoundingBox()
        obb = o3d.geometry.OrientedBoundingBox()
        return aabb, obb
    
    axis_aligned_bbox = pcd.get_axis_aligned_bounding_box()
    try:
        oriented_bbox = pcd.get_oriented_bounding_box(robust=cfg.get("obb_robust", True))
    except RuntimeError:
        oriented_bbox = o3d.geometry.OrientedBoundingBox.create_from_axis_aligned_bounding_box(axis_aligned_bbox)
    
    return axis_aligned_bbox, oriented_bbox

def color_by_instance_for_unik3d(pcds):
    """Color point clouds by instance."""
    if not pcds: 
        return []
    cmap = matplotlib.colormaps.get_cmap("turbo")
    instance_colors = cmap(np.linspace(0, 1, len(pcds)))
    colored_pcds = []
    for i, pcd_original in enumerate(pcds):
        if pcd_original.has_points():
            pcd_copy = o3d.geometry.PointCloud(pcd_original)
            pcd_copy.colors = o3d.utility.Vector3dVector(np.tile(instance_colors[i, :3], (len(pcd_copy.points), 1)))
            colored_pcds.append(pcd_copy)
        else:
            colored_pcds.append(o3d.geometry.PointCloud())
    return colored_pcds

def oriented_bbox_to_center_euler_extent_for_unik3d(bbox_center, box_R, bbox_extent):
    """Convert oriented bbox to center, euler, extent."""
    center = np.asarray(bbox_center)
    extent = np.asarray(bbox_extent)
    eulers = Rotation.from_matrix(box_R.copy()).as_euler("XYZ")
    return center, eulers, extent

def axis_aligned_bbox_to_center_euler_extent_for_unik3d(min_coords, max_coords):
    """Convert axis-aligned bbox to center, euler, extent."""
    center = tuple((min_val + max_val) / 2.0 for min_val, max_val in zip(min_coords, max_coords))
    eulers = (0.0, 0.0, 0.0)
    extent = tuple(abs(max_val - min_val) for min_val, max_val in zip(min_coords, max_coords))
    return center, eulers, extent

LLM_HF_SYSTEM_PROMPT = (
    "You are an expert at rephrasing factual statements about objects in an image into natural-sounding question-answer pairs. "
    "Output ONLY a single JSON object with 'Question' and 'Answer' keys. Do not include any other text."
)

def prepare_llm_prompts_from_facts(facts, detection_list_dicts):
    """Prepare LLM prompts from facts."""
    batched_instructions = []
    for fact_instruction in facts:
        i_regions_found = re.findall(r"<region(\d+)>", fact_instruction)
        region_to_tag = {}
        valid_regions_in_fact = True
        for r_idx_str in i_regions_found:
            r_idx = int(r_idx_str)
            if 0 <= r_idx < len(detection_list_dicts):
                region_to_tag[r_idx] = detection_list_dicts[r_idx]["class_name"]
            else:
                valid_regions_in_fact = False
                break
        if not valid_regions_in_fact: 
            continue
        object_references = []
        unique_region_indices = sorted(list(set(map(int, i_regions_found))))
        for r_idx in unique_region_indices:
            if r_idx in region_to_tag: 
                object_references.append(f"<region{r_idx}> {region_to_tag[r_idx]}")
        object_reference_str = ", ".join(object_references)
        new_instruction_for_llm = f"[Objects]: {object_reference_str}. [Description]: {fact_instruction}"
        batched_instructions.append(new_instruction_for_llm)
    return batched_instructions

def parse_qas_from_vqa_results(vqa_results):
    """Parse QAs from VQA results."""
    conversations = []
    for item in vqa_results:
        qa_pair = item[0]
        conversations.append(qa_pair)
    return conversations

def instantiate_model(model_name):
    """Instantiate UniK3D model."""
    type_ = model_name[0].lower()
    name = f"unik3d-vit{type_}"
    model = UniK3D.from_pretrained(f"lpiccinelli/{name}")
    model.resolution_level = 9
    model.interpolation_mode = "bilinear"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    return model

warnings.filterwarnings("ignore")

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import threading

class LLMPromptLogger:
    """Comprehensive logger for LLM prompts, requests, and responses."""
    
    def __init__(self, log_dir: str = "./llm_logs", enable_file_logging: bool = True, 
                 enable_console_logging: bool = False):
        """
        Initialize the LLM prompt logger.
        
        Args:
            log_dir: Directory to save logs
            enable_file_logging: Whether to save logs to files
            enable_console_logging: Whether to print logs to console
        """
        self.log_dir = Path(log_dir)
        self.enable_file_logging = enable_file_logging
        self.enable_console_logging = enable_console_logging
        self.lock = threading.Lock()
        
        if self.enable_file_logging:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.session_log_dir = self.log_dir / f"session_{self.session_id}"
            self.session_log_dir.mkdir(exist_ok=True)
            self.prompts_file = self.session_log_dir / "prompts.jsonl"
            self.summary_file = self.session_log_dir / "summary.json"
        
        self.request_counter = 0
        self.requests_log = []
    
    def log_prompt(self, prompt_data: Dict[str, Any]) -> None:
        """
        Log a single prompt/request to LLM.
        
        Args:
            prompt_data: Dictionary containing:
                - request_id: Unique identifier
                - timestamp: Timestamp of request
                - llm_type: 'external_api' or 'local_hf'
                - model_name: Name of the model
                - system_prompt: System prompt (optional)
                - user_prompt: User prompt
                - messages: List of message dicts (alternative to system/user prompts)
                - max_tokens: Max tokens parameter
                - temperature: Temperature parameter
                - thinking_budget: Thinking budget if applicable
                - context: Optional context about the request
                - response: LLM response (optional, added later)
                - response_tokens: Number of tokens in response
                - latency_seconds: Request latency
                - status: 'success', 'error', 'timeout', etc.
                - error_message: Error details if applicable
        """
        with self.lock:
            self.request_counter += 1
            
            # Add request counter if not present
            if 'request_id' not in prompt_data:
                prompt_data['request_id'] = self.request_counter
            
            # Add timestamp if not present
            if 'timestamp' not in prompt_data:
                prompt_data['timestamp'] = datetime.now().isoformat()
            
            self.requests_log.append(prompt_data)
            
            # File logging
            if self.enable_file_logging:
                self._write_to_jsonl(prompt_data)
            
            # Console logging
            if self.enable_console_logging:
                self._print_prompt(prompt_data)
    
    def _write_to_jsonl(self, prompt_data: Dict[str, Any]) -> None:
        """Write prompt data to JSONL file."""
        try:
            with open(self.prompts_file, 'a') as f:
                f.write(json.dumps(prompt_data) + '\n')
        except Exception as e:
            print(f"Error writing to prompt log: {e}")
    
    def _print_prompt(self, prompt_data: Dict[str, Any]) -> None:
        """Print prompt to console."""
        print(f"\n{'='*80}")
        print(f"LLM Request #{prompt_data.get('request_id', '?')}")
        print(f"{'='*80}")
        print(f"Model: {prompt_data.get('model_name', 'Unknown')}")
        print(f"Status: {prompt_data.get('status', 'Unknown')}")
        if 'user_prompt' in prompt_data:
            print(f"Prompt: {prompt_data['user_prompt'][:200]}...")
        if 'response' in prompt_data:
            print(f"Response: {prompt_data['response'][:200]}...")
        print(f"Latency: {prompt_data.get('latency_seconds', 'N/A')}s")
        print(f"{'='*80}\n")
    
    def log_external_api_request(self, model_name: str, messages: List[Dict], 
                                 max_tokens: int, temperature: float,
                                 thinking_budget: Optional[int] = None,
                                 context: Optional[str] = None,
                                 logger=None) -> int:
        """
        Log an external API request before making it.
        
        Returns:
            request_id for tracking
        """
        request_id = self.request_counter + 1
        
        # Extract system and user prompts from messages
        system_prompt = None
        user_prompt = None
        for msg in messages:
            if msg.get('role') == 'system':
                system_prompt = msg.get('content', '')
            elif msg.get('role') == 'user':
                user_prompt = msg.get('content', '')
        
        prompt_data = {
            'request_id': request_id,
            'timestamp': datetime.now().isoformat(),
            'llm_type': 'external_api',
            'model_name': model_name,
            'system_prompt': system_prompt,
            'user_prompt': user_prompt,
            'messages': messages,  # Log the full messages list
            'messages_count': len(messages),
            'max_tokens': max_tokens,
            'temperature': temperature,
            'thinking_budget': thinking_budget,
            'context': context,
            'status': 'pending'
        }
        
        self.log_prompt(prompt_data)
        
        if logger:
            logger.info(f"LLM Request #{request_id}: {model_name} - {context or 'no context'}")
        
        return request_id
    
    def log_external_api_response(self, request_id: int, response: str, 
                                 latency_seconds: float, status: str = 'success',
                                 error_message: Optional[str] = None,
                                 logger=None) -> None:
        """Log response to an external API request."""
        with self.lock:
            # Find and update the request
            for req in self.requests_log:
                if req.get('request_id') == request_id:
                    req['response'] = response[:1000]  # Store first 1000 chars
                    req['response_full_length'] = len(response)
                    req['latency_seconds'] = latency_seconds
                    req['status'] = status
                    if error_message:
                        req['error_message'] = error_message
                    
                    # Re-write updated record
                    if self.enable_file_logging:
                        self._write_to_jsonl(req)
                    
                    if logger:
                        logger.info(f"LLM Response #{request_id}: Status={status}, "
                                  f"Latency={latency_seconds:.2f}s, Response_len={len(response)}")
                    break
    
    def log_local_llm_request(self, model_name: str, prompt: str,
                             max_tokens: int, temperature: float,
                             context: Optional[str] = None,
                             logger=None) -> int:
        """Log a local HF LLM request."""
        request_id = self.request_counter + 1
        
        prompt_data = {
            'request_id': request_id,
            'timestamp': datetime.now().isoformat(),
            'llm_type': 'local_hf',
            'model_name': model_name,
            'user_prompt': prompt[:500],  # Store first 500 chars
            'prompt_full_length': len(prompt),
            'max_tokens': max_tokens,
            'temperature': temperature,
            'context': context,
            'status': 'pending'
        }
        
        self.log_prompt(prompt_data)
        
        if logger:
            logger.info(f"Local LLM Request #{request_id}: {model_name}")
        
        return request_id
    
    def log_local_llm_response(self, request_id: int, response: str,
                              latency_seconds: float, status: str = 'success',
                              error_message: Optional[str] = None,
                              logger=None) -> None:
        """Log response from local LLM."""
        with self.lock:
            for req in self.requests_log:
                if req.get('request_id') == request_id:
                    req['response'] = response[:1000]
                    req['response_full_length'] = len(response)
                    req['latency_seconds'] = latency_seconds
                    req['status'] = status
                    if error_message:
                        req['error_message'] = error_message
                    
                    if self.enable_file_logging:
                        self._write_to_jsonl(req)
                    
                    if logger:
                        logger.info(f"Local LLM Response #{request_id}: Status={status}, "
                                  f"Latency={latency_seconds:.2f}s")
                    break
    
    def log_json_qa_request(self, user_prompt: str, system_prompt: str,
                           model_name: str, context: Optional[str] = None,
                           logger=None) -> int:
        """Log a JSON QA generation request."""
        request_id = self.request_counter + 1
        
        prompt_data = {
            'request_id': request_id,
            'timestamp': datetime.now().isoformat(),
            'llm_type': 'json_qa',
            'model_name': model_name,
            'system_prompt': system_prompt[:200],
            'user_prompt': user_prompt[:500],
            'prompt_type': 'json_qa',
            'context': context,
            'status': 'pending'
        }
        
        self.log_prompt(prompt_data)
        
        if logger:
            logger.info(f"JSON QA Request #{request_id}: {model_name}")
        
        return request_id
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate a summary of all logged requests."""
        with self.lock:
            total_requests = len(self.requests_log)
            
            # Count by type and status
            by_type = {}
            by_status = {}
            by_model = {}
            
            total_latency = 0
            successful_requests = 0
            
            for req in self.requests_log:
                llm_type = req.get('llm_type', 'unknown')
                status = req.get('status', 'unknown')
                model = req.get('model_name', 'unknown')
                
                by_type[llm_type] = by_type.get(llm_type, 0) + 1
                by_status[status] = by_status.get(status, 0) + 1
                by_model[model] = by_model.get(model, 0) + 1
                
                if status == 'success':
                    successful_requests += 1
                    if 'latency_seconds' in req:
                        total_latency += req['latency_seconds']
            
            summary = {
                'session_id': self.session_id if self.enable_file_logging else 'console_only',
                'start_time': datetime.now().isoformat(),
                'total_requests': total_requests,
                'successful_requests': successful_requests,
                'failed_requests': total_requests - successful_requests,
                'success_rate': (successful_requests / total_requests * 100) if total_requests > 0 else 0,
                'average_latency_seconds': (total_latency / successful_requests) if successful_requests > 0 else 0,
                'requests_by_type': by_type,
                'requests_by_status': by_status,
                'requests_by_model': by_model
            }
            
            if self.enable_file_logging:
                self._save_summary(summary)
            
            return summary
    
    def _save_summary(self, summary: Dict[str, Any]) -> None:
        """Save summary to JSON file."""
        try:
            with open(self.summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
        except Exception as e:
            print(f"Error writing summary: {e}")
    
    def get_logs(self, request_id: Optional[int] = None) -> Union[Dict, List[Dict]]:
        """Retrieve logs by request ID or all logs."""
        with self.lock:
            if request_id is not None:
                for req in self.requests_log:
                    if req.get('request_id') == request_id:
                        return req
                return None
            return self.requests_log.copy()
    
    def export_logs(self, filepath: str) -> None:
        """Export all logs to a file."""
        with self.lock:
            try:
                with open(filepath, 'w') as f:
                    json.dump(self.requests_log, f, indent=2)
                print(f"Logs exported to {filepath}")
            except Exception as e:
                print(f"Error exporting logs: {e}")

def visualize_detected_objects_bboxes(wis3d_instance, detected_objects_list):
    """
    Add 3D bounding boxes to Wis3D visualization from DetectedObject instances.
    
    Args:
        wis3d_instance: Wis3D visualization instance
        detected_objects_list: List of DetectedObject instances
    """
    import matplotlib
    import numpy as np
    cmap = matplotlib.colormaps.get_cmap("turbo")
    
    # Define the 12 edges of a box (connecting 8 corners)
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
        [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
        [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical edges
    ]
    
    for i, detected_obj in enumerate(detected_objects_list):
        class_name = detected_obj.class_name
        color = cmap(i / max(len(detected_objects_list), 1))[:3]
        
        # Try oriented bbox first, fall back to axis-aligned
        obb = detected_obj.bounding_box_3d_oriented
        aabb = detected_obj.bounding_box_3d_axis_aligned
        
        bbox_points = None
        bbox_type = None
        
        if obb and not obb.is_empty():
            # Get oriented bounding box corners
            bbox_points = np.asarray(obb.get_box_points())
            bbox_type = "OBB"
        elif aabb and not aabb.is_empty():
            # For axis-aligned bbox, get min/max corners
            min_bound = np.asarray(aabb.get_min_bound())
            max_bound = np.asarray(aabb.get_max_bound())
            
            # Create 8 corners of axis-aligned box
            bbox_points = np.array([
                [min_bound[0], min_bound[1], min_bound[2]],
                [max_bound[0], min_bound[1], min_bound[2]],
                [max_bound[0], max_bound[1], min_bound[2]],
                [min_bound[0], max_bound[1], min_bound[2]],
                [min_bound[0], min_bound[1], max_bound[2]],
                [max_bound[0], min_bound[1], max_bound[2]],
                [max_bound[0], max_bound[1], max_bound[2]],
                [min_bound[0], max_bound[1], max_bound[2]]
            ])
            bbox_type = "AABB"
        else:
            # Skip if no valid bounding box
            continue
        
        # Collect all start and end points for this bbox
        start_points = []
        end_points = []
        for start_idx, end_idx in edges:
            start_points.append(bbox_points[start_idx])
            end_points.append(bbox_points[end_idx])
        
        start_points = np.array(start_points)
        end_points = np.array(end_points)
        
        # Add all edges as lines with single call
        wis3d_instance.add_lines(
            start_points,
            end_points,
            colors=np.tile(color, (len(edges), 1)),
            name=f"{i:02d}_{class_name}_{bbox_type}"
        )
        
        # Optional: Add center point as sphere
        center = detected_obj.center
        wis3d_instance.add_spheres(
            center.reshape(1, 3),
            radius=0.02,
            colors=np.array([color]),
            name=f"{i:02d}_{class_name}_center"
        )


def export_scene_with_bboxes_to_ply(output_path, global_points, global_colors, 
                                     valid_detections_dicts):
    """
    Export the global point cloud with 3D bounding boxes to a PLY file.
    
    Args:
        output_path: Path to save PLY file
        global_points: Nx3 array of global scene points
        global_colors: Nx3 array of RGB colors (0-1 range)
        valid_detections_dicts: List of detection dictionaries
    """
    import open3d as o3d
    import matplotlib
    
    # Create combined point cloud
    combined_pcd = o3d.geometry.PointCloud()
    combined_pcd.points = o3d.utility.Vector3dVector(global_points)
    combined_pcd.colors = o3d.utility.Vector3dVector(global_colors)
    
    # Add bounding box edges as line segments
    cmap = matplotlib.colormaps.get_cmap("turbo")
    all_bbox_points = []
    all_bbox_colors = []
    
    for i, det_dict in enumerate(valid_detections_dicts):
        color = cmap(i / max(len(valid_detections_dicts), 1))[:3]
        
        obb = det_dict.get("oriented_bbox")
        aabb = det_dict.get("axis_aligned_bbox")
        
        if obb and not obb.is_empty():
            bbox_points = np.asarray(obb.get_box_points())
        elif aabb and not aabb.is_empty():
            min_bound = np.asarray(aabb.get_min_bound())
            max_bound = np.asarray(aabb.get_max_bound())
            bbox_points = np.array([
                [min_bound[0], min_bound[1], min_bound[2]],
                [max_bound[0], min_bound[1], min_bound[2]],
                [max_bound[0], max_bound[1], min_bound[2]],
                [min_bound[0], max_bound[1], min_bound[2]],
                [min_bound[0], min_bound[1], max_bound[2]],
                [max_bound[0], min_bound[1], max_bound[2]],
                [max_bound[0], max_bound[1], max_bound[2]],
                [min_bound[0], max_bound[1], max_bound[2]]
            ])
        else:
            continue
        
        # Create dense line segments for each edge
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7]
        ]
        
        for start, end in edges:
            # Interpolate points along edge for visibility
            num_points = 20
            t = np.linspace(0, 1, num_points)
            edge_points = bbox_points[start] + t[:, None] * (bbox_points[end] - bbox_points[start])
            all_bbox_points.append(edge_points)
            all_bbox_colors.append(np.tile(color, (num_points, 1)))
    
    # Add bbox edge points to main point cloud
    if all_bbox_points:
        bbox_points_array = np.vstack(all_bbox_points)
        bbox_colors_array = np.vstack(all_bbox_colors)
        
        combined_points = np.vstack([
            np.asarray(combined_pcd.points),
            bbox_points_array
        ])
        combined_colors = np.vstack([
            np.asarray(combined_pcd.colors),
            bbox_colors_array
        ])
        
        combined_pcd.points = o3d.utility.Vector3dVector(combined_points)
        combined_pcd.colors = o3d.utility.Vector3dVector(combined_colors)
    
    # Save to PLY
    o3d.io.write_point_cloud(output_path, combined_pcd)
    print(f"Saved scene with bounding boxes to: {output_path}")

def export_scene_with_all_solid_obbs(output_path, global_points, global_colors, valid_detections_dicts):
    """
    Exports the global point cloud with THICK, SOLID TUBE Oriented Bounding Boxes for all objects.
    """
    import open3d as o3d
    import matplotlib
    
    print(f"Generating solid OBB visualization for {len(valid_detections_dicts)} objects...")

    # 1. Base Scene
    combined_points_list = [global_points]
    combined_colors_list = [global_colors]
    
    # Color map for distinct objects
    cmap = matplotlib.colormaps.get_cmap("turbo")
    
    # 2. Iterate over all detections
    for i, det_dict in enumerate(valid_detections_dicts):
        
        # Get Oriented BBox
        obb = det_dict.get("oriented_bbox")
        if obb is None or obb.is_empty():
            continue

        # Get distinct color for this object
        # i / len ensures spread across the colormap
        color = np.array(cmap(i / max(len(valid_detections_dicts), 1))[:3])
        
        # Get the 8 corners of the box
        bbox_corners = np.asarray(obb.get_box_points())
        
        # The 12 edges connecting the 8 corners
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0], # Bottom face
            [4, 5], [5, 6], [6, 7], [7, 4], # Top face
            [0, 4], [1, 5], [2, 6], [3, 7]  # Verticals
        ]
        
        for start, end in edges:
            edge_pts, edge_cols = generate_solid_tube_points(
                bbox_corners[start], 
                bbox_corners[end], 
                color, 
                radius=0.015,           # 1.5cm thickness (Adjust as needed)
                density_per_meter=8000  # High density for solid look
            )
            
            if len(edge_pts) > 0:
                combined_points_list.append(edge_pts)
                combined_colors_list.append(edge_cols)

    # 3. Stack and Save
    final_points = np.vstack(combined_points_list)
    final_colors = np.vstack(combined_colors_list)
    
    combined_pcd = o3d.geometry.PointCloud()
    combined_pcd.points = o3d.utility.Vector3dVector(final_points)
    combined_pcd.colors = o3d.utility.Vector3dVector(final_colors)
    
    o3d.io.write_point_cloud(output_path, combined_pcd)
    print(f"Saved scene with SOLID OBBs to: {output_path}")

def export_target_class_solid_obbs(output_path, global_points, global_colors, detected_objects_list, target_class_name):
    """
    Exports the scene with THICK SOLID OBBs only for objects matching target_class_name.
    
    Args:
        output_path: Where to save the PLY.
        global_points: Numpy array (N, 3) of scene points.
        global_colors: Numpy array (N, 3) of scene colors.
        detected_objects_list: List of DetectedObject instances.
        target_class_name: String (case-insensitive) of the class to highlight (e.g., "chair").
    """
    import open3d as o3d
    import numpy as np

    print(f"Attempting to highlight class: '{target_class_name}'...")

    # 1. Start with the Base Scene
    combined_points_list = [global_points]
    combined_colors_list = [global_colors]
    
    # Define a specific highlight color (e.g., Bright Red)
    highlight_color = np.array([1.0, 0.0, 0.0]) 
    
    match_count = 0

    # 2. Iterate over detected objects
    for obj in detected_objects_list:
        
        # --- FILTERING LOGIC ---
        # Compare class names (case-insensitive)
        if obj.class_name.lower() != target_class_name.lower():
            continue
            
        obb = obj.bounding_box_3d_oriented
        if obb is None or obb.is_empty():
            continue

        match_count += 1
        
        # 3. Draw the Box
        bbox_corners = np.asarray(obb.get_box_points())
        
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0], # Bottom
            [4, 5], [5, 6], [6, 7], [7, 4], # Top
            [0, 4], [1, 5], [2, 6], [3, 7]  # Sides
        ]
        
        for start, end in edges:
            edge_pts, edge_cols = generate_solid_tube_points(
                bbox_corners[start], 
                bbox_corners[end], 
                highlight_color, 
                radius=0.02,            # Slightly thicker (2cm) for emphasis
                density_per_meter=10000 
            )
            
            if len(edge_pts) > 0:
                combined_points_list.append(edge_pts)
                combined_colors_list.append(edge_cols)

    if match_count == 0:
        print(f"Warning: No objects found matching class '{target_class_name}'. Saving raw scene only.")
    else:
        print(f"Highlighted {match_count} instance(s) of '{target_class_name}'.")

    # 4. Save
    final_points = np.vstack(combined_points_list)
    final_colors = np.vstack(combined_colors_list)
    
    combined_pcd = o3d.geometry.PointCloud()
    combined_pcd.points = o3d.utility.Vector3dVector(final_points)
    combined_pcd.colors = o3d.utility.Vector3dVector(final_colors)
    
    o3d.io.write_point_cloud(output_path, combined_pcd)
    print(f"Saved to: {output_path}")

def generate_camera_frustum_points(scale=0.1, color=np.array([0, 1, 1])):
    """
    Generates a solid tube representation of a camera at (0,0,0).
    Color defaults to Cyan.
    """
    # Camera Origin
    origin = np.array([0.0, 0.0, 0.0])
    
    # Frustum Plane (Image Plane representation)
    # Assuming Z-forward (standard for UniK3D/Camera coords)
    w = scale        # width
    h = scale * 0.75 # aspect ratio 4:3
    z = scale        # focal length visual
    
    tl = np.array([-w, -h, z]) # Top-Left
    tr = np.array([ w, -h, z]) # Top-Right
    br = np.array([ w,  h, z]) # Bottom-Right
    bl = np.array([-w,  h, z]) # Bottom-Left
    
    corners = [tl, tr, br, bl]
    
    points_list = []
    colors_list = []
    
    # 1. Connect Origin to 4 corners
    for corner in corners:
        pts, cols = generate_solid_tube_points(
            origin, corner, color, radius=0.005, density_per_meter=10000
        )
        if len(pts) > 0:
            points_list.append(pts)
            colors_list.append(cols)
            
    # 2. Connect corners to make a rectangle
    edges = [(0,1), (1,2), (2,3), (3,0)] # tl->tr->br->bl->tl
    for i, j in edges:
        pts, cols = generate_solid_tube_points(
            corners[i], corners[j], color, radius=0.005, density_per_meter=10000
        )
        if len(pts) > 0:
            points_list.append(pts)
            colors_list.append(cols)

    if not points_list:
        return np.array([]), np.array([])
        
    return np.vstack(points_list), np.vstack(colors_list)

def export_scene_with_camera_focus(output_path, global_points, global_colors, 
                                   detected_objects_list, target_class_name):
    """
    Exports:
    1. The Point Cloud
    2. The Target Object BBox (Red Solid Tubes)
    3. The Camera Frustum at 0,0,0 (Cyan Solid Tubes)
    4. Projection Rays from Camera to Object (Yellow Solid Tubes)
    """
    import open3d as o3d
    import numpy as np

    print(f"Generating Camera Focus visualization for '{target_class_name}'...")

    # --- 1. Base Scene ---
    combined_points_list = [global_points]
    combined_colors_list = [global_colors]
    
    # Colors
    color_target = np.array([1.0, 0.0, 0.0]) # Red for Object
    color_camera = np.array([0.0, 1.0, 1.0]) # Cyan for Camera
    color_rays   = np.array([1.0, 0.8, 0.0]) # Gold/Yellow for rays

    target_found = False
    
    # --- 2. Find Target & Draw Geometry ---
    for obj in detected_objects_list:
        if obj.class_name.lower() != target_class_name.lower():
            continue
            
        obb = obj.bounding_box_3d_oriented
        if obb is None or obb.is_empty():
            continue

        target_found = True
        
        # A. Draw Target Object BBox (Thick Red)
        bbox_corners = np.asarray(obb.get_box_points())
        box_edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7]
        ]
        
        for start, end in box_edges:
            pts, cols = generate_solid_tube_points(
                bbox_corners[start], bbox_corners[end], 
                color_target, radius=0.02, density_per_meter=8000
            )
            if len(pts) > 0:
                combined_points_list.append(pts)
                combined_colors_list.append(cols)

        # B. Draw Camera Frustum at Origin
        cam_pts, cam_cols = generate_camera_frustum_points(scale=0.15, color=color_camera)
        if len(cam_pts) > 0:
            combined_points_list.append(cam_pts)
            combined_colors_list.append(cam_cols)

        # C. Draw Projection Rays (Camera Origin -> Object Corners)
        # We connect (0,0,0) to the 8 corners of the object
        camera_origin = np.array([0.0, 0.0, 0.0])
        
        for corner in bbox_corners:
            # Make rays slightly thinner than the object box so they don't obscure it
            pts, cols = generate_solid_tube_points(
                camera_origin, corner, 
                color_rays, radius=0.008, density_per_meter=5000
            )
            if len(pts) > 0:
                combined_points_list.append(pts)
                combined_colors_list.append(cols)

        # Stop after the first matching instance to avoid visual chaos
        # (Remove 'break' if you want to visualize rays to ALL instances of the class)
        break 

    if not target_found:
        print(f"Warning: Target '{target_class_name}' not found. Exporting raw scene.")

    # --- 3. Save ---
    final_points = np.vstack(combined_points_list)
    final_colors = np.vstack(combined_colors_list)
    
    combined_pcd = o3d.geometry.PointCloud()
    combined_pcd.points = o3d.utility.Vector3dVector(final_points)
    combined_pcd.colors = o3d.utility.Vector3dVector(final_colors)
    
    o3d.io.write_point_cloud(output_path, combined_pcd)
    print(f"Saved Camera Focus visualization to: {output_path}")

def _export_captured_and_segmented_images(self, image_bgr, detection_list, filename_prefix):
    """
    Export both the original captured image AND the YOLOE segmentation visualization.
    
    Args:
        image_bgr: Original captured image in BGR format
        detection_list: List of detection dictionaries with masks and boxes
        filename_prefix: Prefix for output filenames
    
    Returns:
        Dictionary with paths to exported files
    """
    try:
        debug_folder = self._create_debug_folder()
        exported_files = {}
        
        # 1. Export Original Captured Image
        original_path = os.path.join(
            debug_folder,
            f"{filename_prefix}_original_captured.jpg"
        )
        cv2.imwrite(original_path, image_bgr)
        exported_files['original'] = original_path
        self.logger.info(f"✓ Saved original image to: {original_path}")
        
        # 2. Export YOLOE Segmentation Visualization
        vis_image = self._visualize_yoloe_results(image_bgr, detection_list, filename_prefix)
        segmentation_path = os.path.join(
            debug_folder,
            f"{filename_prefix}_yoloe_segmentation.jpg"
        )
        cv2.imwrite(segmentation_path, vis_image)
        exported_files['segmentation'] = segmentation_path
        self.logger.info(f"✓ Saved YOLOE segmentation to: {segmentation_path}")
        
        # 3. Export Individual Masks
        masks_folder = os.path.join(debug_folder, f"{filename_prefix}_masks")
        os.makedirs(masks_folder, exist_ok=True)
        
        mask_paths = []
        for idx, detection in enumerate(detection_list):
            mask = detection["mask"].astype(np.uint8) * 255
            class_name = detection.get("class_name", f"obj_{idx}")
            mask_path = os.path.join(
                masks_folder,
                f"{idx:03d}_{class_name}_mask.png"
            )
            cv2.imwrite(mask_path, mask)
            mask_paths.append(mask_path)
        
        exported_files['masks_folder'] = masks_folder
        exported_files['individual_masks'] = mask_paths
        self.logger.info(f"✓ Saved {len(detection_list)} masks to: {masks_folder}")
        
        # 4. Export Metadata
        metadata = {
            "filename_prefix": filename_prefix,
            "timestamp": datetime.now().isoformat(),
            "num_detections": len(detection_list),
            "image_shape": {
                "height": image_bgr.shape[0],
                "width": image_bgr.shape[1],
                "channels": image_bgr.shape[2]
            },
            "exported_files": {
                "original": original_path,
                "segmentation": segmentation_path,
                "masks_folder": masks_folder
            },
            "detections": []
        }
        
        for idx, detection in enumerate(detection_list):
            det_meta = {
                "id": idx,
                "class_name": detection.get("class_name", "unknown"),
                "confidence": float(detection.get("confidence", 0.0)),
                "bounding_box_xyxy": [float(x) for x in detection["xyxy"]],
                "mask_area_pixels": int(np.sum(detection["mask"])),
                "mask_file": mask_paths[idx]
            }
            metadata["detections"].append(det_meta)
        
        metadata_path = os.path.join(
            debug_folder,
            f"{filename_prefix}_export_metadata.json"
        )
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        exported_files['metadata'] = metadata_path
        self.logger.info(f"✓ Saved metadata to: {metadata_path}")
        
        # 5. Create Summary Report
        summary = f"""
════════════════════════════════════════════════════════════════
YOLOE SEGMENTATION EXPORT SUMMARY
════════════════════════════════════════════════════════════════
Timestamp: {metadata['timestamp']}
Prefix: {filename_prefix}

EXPORTED FILES:
  📷 Original Image:    {original_path}
  🎨 Segmentation:      {segmentation_path}
  📁 Masks Folder:      {masks_folder}
  📋 Metadata:          {metadata_path}

DETECTIONS: {len(detection_list)} objects
"""
        for idx, det in enumerate(detection_list):
            summary += f"  [{idx}] {det.get('class_name')} (conf: {det.get('confidence', 0):.2f})\n"
        
        summary += "════════════════════════════════════════════════════════════════\n"
        print(summary)
        
        return exported_files
        
    except Exception as e:
        self.logger.error(f"Error exporting images: {e}", exc_info=True)
        return {}


class GeneralizedSceneGraphGenerator:
    """Main scene graph generator with modular LLM support."""

    def __init__(self, config_path="config/v2_hf_llm.py", device="cuda", 
                llm_interface: Optional[LLMInterface] = None):
        """Initialize the generator.
        
        Args:
            config_path: Path to the configuration file
            device: Device to use ('cuda' or 'cpu')
            llm_interface: Optional pre-configured LLM interface. If None, attempts to initialize from config.
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        self.cfg = Config.fromfile(config_path)
        self.device = device
        self.logger = setup_logger(name="GeneralizedSceneGraphGeneratorHF")
        self.timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        self.llm_interface = llm_interface

        # Initialize prompt logger
        log_dir = self.cfg.get("llm_logs_dir", "./llm_logs")
        self.prompt_logger = LLMPromptLogger(
            log_dir=log_dir,
            enable_file_logging=self.cfg.get("enable_llm_file_logging", True),
            enable_console_logging=self.cfg.get("enable_llm_console_logging", False)
        )
        self.logger.info(f"LLM Prompt Logger initialized. Logs will be saved to: {log_dir}")

        # Initialize YOLOE
        self.yoloe_model_path = self.cfg.get("yoloe_model_path", "yoloe-l-seg.pt")
        try:
            self.logger.info(f"Initializing YOLOE model from: {self.yoloe_model_path}")
            self.yoloe_model = YOLOE(self.yoloe_model_path)
            self.logger.info("Successfully initialized YOLOE model.")
        except Exception as e:
            self.logger.error(f"Failed to initialize YOLOE model: {e}")
            raise RuntimeError("Could not initialize YOLOE model.")
        
        # Initialize UniK3D
        try:
            self.logger.info("Initializing UniK3D model...")
            self.unik3d_model = instantiate_model(self.cfg.get("unik3d_model_size", "Small"))
            self.logger.info("Successfully initialized UniK3D model")
        except Exception as e:
            self.logger.error(f"Failed to initialize UniK3D model: {e}")
            raise RuntimeError("Could not initialize UniK3D model")

        # Initialize other components
        self.captioner = CaptionImage(self.cfg, self.logger, self.device, 
                                    init_lava=self.cfg.get("init_lava_on_startup", False))
        self.qa_prompter = QAPromptGenerator(self.cfg, self.logger, self.device)
        self.fact_prompter = FactPromptGenerator(self.cfg, self.logger, self.device)

        # Initialize LLM if not provided
        if self.llm_interface is None:
            self._initialize_default_llm()

        # Setup visualization
        default_wis3d_folder = os.path.join(self.cfg.get("log_dir", "./temp_outputs/log"), 
                                        f"Wis3D_Generalized_HF_{self.timestamp}")
        self.cfg.wis3d_folder = self.cfg.get("wis3d_folder", default_wis3d_folder)
        os.makedirs(self.cfg.wis3d_folder, exist_ok=True)
        self.cfg.vis = self.cfg.get("vis", False)
        
    def _initialize_default_llm(self):
        """Initialize default LLM from config if not provided."""
        try:
            llm_api_url = self.cfg.get("llm_api_url")
            llm_api_model = self.cfg.get("llm_api_model")
            llm_api_key = self.cfg.get("llm_api_key")
            llm_api_headers = self.cfg.get("llm_api_headers", {})
            llm_local_model = self.cfg.get("llm_local_model")

            if llm_api_url:
                self.logger.info(f"Initializing external LLM API: {llm_api_url}")
                self.llm_interface = ExternalAPILLM(
                    llm_api_url, llm_api_model, api_key=llm_api_key,
                    custom_headers=llm_api_headers, logger=self.logger,
                    thinking_budget=0
                )
                # Attach prompt logger to the LLM interface
                if hasattr(self, 'prompt_logger'):
                    self.llm_interface.prompt_logger = self.prompt_logger
                
                if self.llm_interface.is_available():
                    self.logger.info(f"External LLM API initialized successfully.")
                else:
                    self.logger.warning("External LLM API health check failed. Trying local LLM.")
                    self.llm_interface = None
            
            if self.llm_interface is None and llm_local_model and HF_TRANSFORMERS_AVAILABLE:
                self.logger.info(f"Initializing local LLM: {llm_local_model}")
                llm_device = "cuda" if torch.cuda.is_available() else "cpu"
                self.llm_interface = LocalHFLLM(llm_local_model, llm_device, logger=self.logger)
                if self.llm_interface.is_available():
                    self.logger.info("Local LLM initialized successfully.")
                else:
                    self.llm_interface = None
                    self.logger.warning("Failed to initialize local LLM.")
        except Exception as e:
            self.logger.warning(f"Could not initialize LLM: {e}")
            self.llm_interface = None
            
    def set_llm_interface(self, llm_interface: Optional[LLMInterface]):
        """Set or replace the LLM interface at runtime."""
        self.llm_interface = llm_interface
        if llm_interface:
            self.logger.info(f"LLM interface updated to: {type(llm_interface).__name__}")
        else:
            self.logger.info("LLM interface disabled.")

    def _load_image(self, image_input):
        """Load and prepare image."""
        if isinstance(image_input, str):
            if not os.path.exists(image_input): 
                raise FileNotFoundError(f"Image not found at {image_input}")
            image_bgr = cv2.imread(image_input)
            if image_bgr is None: 
                raise ValueError(f"Could not read image from {image_input}")
        elif isinstance(image_input, np.ndarray): 
            image_bgr = image_input.copy()
        else: 
            raise TypeError("image_input must be a file path (str) or NumPy array (BGR).")
        
        image_bgr = image_bgr[:, :, :3]
        h, w = image_bgr.shape[:2]
        if h == 0 or w == 0: 
            raise ValueError("Image has zero height or width.")
        
        target_h = self.cfg.get("image_resize_height", 640)
        if h != target_h:
            scale = target_h / h
            target_w = int(w * scale)
            image_bgr_resized = cv2.resize(image_bgr, (target_w, target_h))
        else:
            image_bgr_resized = image_bgr
        
        return image_bgr_resized

    def _get_object_classes(self, image_rgb_pil, custom_vocabulary=None):
        """Get object classes to detect."""
        yoloe_class_names = list(self.yoloe_model.names.values())

        if custom_vocabulary:
            if not isinstance(custom_vocabulary, list) or not all(isinstance(s, str) for s in custom_vocabulary):
                self.logger.warning("custom_vocabulary must be a list of strings.")
                return yoloe_class_names
            if not custom_vocabulary:
                return yoloe_class_names
            
            valid_custom_classes = [cls.lower() for cls in custom_vocabulary 
                                   if cls.lower() in map(str.lower, yoloe_class_names)]
            yoloe_names_lower_map = {name.lower(): name for name in yoloe_class_names}
            valid_custom_classes_cased = [yoloe_names_lower_map[cls_lower] for cls_lower in valid_custom_classes]

            if not valid_custom_classes_cased:
                self.logger.warning(f"No custom vocabulary classes detected. Using all YOLOE classes.")
                return yoloe_class_names
            
            self.logger.info(f"Using custom vocabulary: {valid_custom_classes_cased}")
            return valid_custom_classes_cased
        else:
            return yoloe_class_names

    def _create_debug_folder(self):
        """Create debug logging folder if it doesn't exist."""
        debug_folder = self.cfg.get("debug_folder", "./debug_logs")
        os.makedirs(debug_folder, exist_ok=True)
        return debug_folder

    def _visualize_yoloe_results(self, image_bgr, detection_list, filename_prefix):
        """Visualize YOLOE segmentation results on image.
        
        Args:
            image_bgr: Input image in BGR format
            detection_list: List of detection dictionaries with masks and boxes
            filename_prefix: Prefix for output filename
        
        Returns:
            Visualization image as numpy array (BGR)
        """
        vis_image = image_bgr.copy().astype(np.float32)
        h, w = image_bgr.shape[:2]
        
        # Generate distinct colors for each object
        cmap = matplotlib.colormaps.get_cmap("tab20")
        num_objects = len(detection_list)
        colors = [(cmap(i / max(num_objects, 1))[:3]) for i in range(num_objects)]
        
        # Draw masks with semi-transparency
        for idx, detection in enumerate(detection_list):
            mask = detection["mask"].astype(np.uint8)
            color_rgb = colors[idx % len(colors)]
            color_bgr = (int(color_rgb[2] * 255), int(color_rgb[1] * 255), int(color_rgb[0] * 255))
            
            # Apply colored overlay to masked area
            mask_indices = mask > 0
            vis_image[mask_indices] = (
                0.6 * vis_image[mask_indices] + 0.4 * np.array(color_bgr, dtype=np.float32)
            )
            
            # Draw contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis_image, contours, -1, color_bgr, 2)
            
            # Draw bounding box
            x1, y1, x2, y2 = detection["xyxy"]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color_bgr, 2)
            
            # Draw label
            class_name = detection.get("class_name", f"obj_{idx}")
            confidence = detection.get("confidence", 0.0)
            label_text = f"{class_name} ({confidence:.2f})"
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            text_size = cv2.getTextSize(label_text, font, font_scale, thickness)[0]
            
            # Background for text
            cv2.rectangle(
                vis_image,
                (x1, max(y1 - text_size[1] - 4, 0)),
                (x1 + text_size[0] + 4, y1),
                color_bgr,
                -1
            )
            
            # Text
            cv2.putText(
                vis_image,
                label_text,
                (x1 + 2, y1 - 2),
                font,
                font_scale,
                (255, 255, 255),
                thickness
            )
        
        return vis_image.astype(np.uint8)

    def _export_yoloe_debug_visualization(self, image_bgr, detection_list, filename_prefix):
        """Export YOLOE segmentation visualization to debug folder.
        
        Args:
            image_bgr: Input image in BGR format
            detection_list: List of detection dictionaries with masks and boxes
            filename_prefix: Prefix for output filename
        
        Returns:
            Path to saved visualization image
        """
        try:
            debug_folder = self._create_debug_folder()
            
            # Create visualized image
            vis_image = self._visualize_yoloe_results(image_bgr, detection_list, filename_prefix)
            
            # Save visualization
            output_path = os.path.join(
                debug_folder,
                f"{filename_prefix}_yoloe_segmentation.jpg"
            )
            cv2.imwrite(output_path, vis_image)
            self.logger.info(f"Saved YOLOE segmentation visualization to: {output_path}")
            
            # Also save individual masks
            masks_folder = os.path.join(debug_folder, f"{filename_prefix}_masks")
            os.makedirs(masks_folder, exist_ok=True)
            
            for idx, detection in enumerate(detection_list):
                mask = detection["mask"].astype(np.uint8) * 255
                class_name = detection.get("class_name", f"obj_{idx}")
                mask_path = os.path.join(
                    masks_folder,
                    f"{idx:03d}_{class_name}_mask.png"
                )
                cv2.imwrite(mask_path, mask)
            
            self.logger.info(f"Saved {len(detection_list)} individual masks to: {masks_folder}")
            
            # Save metadata
            metadata = {
                "filename_prefix": filename_prefix,
                "num_detections": len(detection_list),
                "image_shape": image_bgr.shape,
                "detections": []
            }
            
            for idx, detection in enumerate(detection_list):
                det_meta = {
                    "id": idx,
                    "class_name": detection.get("class_name", "unknown"),
                    "confidence": float(detection.get("confidence", 0.0)),
                    "xyxy": [float(x) for x in detection["xyxy"]],
                    "mask_area_pixels": int(np.sum(detection["mask"]))
                }
                metadata["detections"].append(det_meta)
            
            metadata_path = os.path.join(
                debug_folder,
                f"{filename_prefix}_yoloe_metadata.json"
            )
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"Saved YOLOE metadata to: {metadata_path}")
            
            return output_path
        
        except Exception as e:
            self.logger.error(f"Error exporting YOLOE visualization: {e}", exc_info=True)
            return None

    def _segment_image(self, image_bgr, classes_to_detect, filename_prefix="unknown_image"):
        """Segment image using YOLOE (modified to include debug export)."""
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_rgb_pil = Image.fromarray(image_rgb)

        target_class_indices = None
        if classes_to_detect:
            name_to_idx_map = {name: idx for idx, name in self.yoloe_model.names.items()}
            target_class_indices = [name_to_idx_map[name] for name in classes_to_detect if name in name_to_idx_map]
        
        yolo_results_list = self.yoloe_model.predict(
            source=image_bgr.copy(),
            classes=target_class_indices,
            conf=self.cfg.get("yoloe_confidence_threshold", 0.6)
        )

        if not yolo_results_list or not yolo_results_list[0].boxes or yolo_results_list[0].boxes.shape[0] == 0:
            del yolo_results_list
            if torch.cuda.is_available(): 
                torch.cuda.empty_cache()
                gc.collect()
            raise SkipImageException("No objects detected by YOLOE.")

        res = yolo_results_list[0]
        if res.masks is None or res.masks.data is None or res.masks.data.shape[0] == 0:
            del yolo_results_list, res
            if torch.cuda.is_available(): 
                torch.cuda.empty_cache()
                gc.collect()
            raise SkipImageException("No masks found in YOLOE results.")

        boxes_xyxy = res.boxes.xyxy.cpu().numpy()
        confidences = res.boxes.conf.cpu().numpy()
        class_ids = res.boxes.cls.cpu().numpy().astype(int)
        masks_data_np = res.masks.data.cpu().numpy()
        
        h_img, w_img = image_bgr.shape[:2]
        processed_masks = []
        for i in range(masks_data_np.shape[0]):
            mask_i = masks_data_np[i, :, :]
            if mask_i.shape[0] != h_img or mask_i.shape[1] != w_img:
                mask_i = cv2.resize(mask_i, (w_img, h_img), interpolation=cv2.INTER_NEAREST)
            binary_mask = (mask_i > self.cfg.get("yoloe_mask_threshold", 0.5)).astype(bool)
            processed_masks.append(binary_mask)
        
        if not processed_masks:
            del yolo_results_list, res, masks_data_np, boxes_xyxy, confidences, class_ids
            if torch.cuda.is_available(): 
                torch.cuda.empty_cache()
                gc.collect()
            raise SkipImageException("No masks could be processed.")
        
        masks_np_full_image = np.stack(processed_masks)
        detected_class_names = [self.yoloe_model.names[cid] for cid in class_ids]

        detection_list = []
        for i in range(len(boxes_xyxy)):
            detection_list.append({
                "xyxy": boxes_xyxy[i],
                "mask": masks_np_full_image[i],
                "subtracted_mask": masks_np_full_image[i].copy(),
                "confidence": confidences[i],
                "class_name": detected_class_names[i],
                "class_id": class_ids[i]
            })

        del res, masks_data_np, boxes_xyxy, confidences, class_ids, processed_masks, yolo_results_list
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        if not detection_list:
            raise SkipImageException("No detections formulated into list.")
        
        # Export YOLOE debug visualization
        # Export both original and segmented images
        # if self.cfg.get("export_yoloe_debug_viz", False):
        exported_files = self._export_captured_and_segmented_images(
            image_bgr, detection_list, filename_prefix
        )
        # Store paths for later access
        self._last_export_paths = exported_files
        # Filter by area
        filtered_detection_list = []
        min_area = self.cfg.get("min_mask_area_pixel", 100)
        for det_entry in detection_list:
            mask_area = np.sum(det_entry["mask"])
            if mask_area >= min_area:
                filtered_detection_list.append(det_entry)
        
        if not filtered_detection_list:
            if torch.cuda.is_available(): 
                torch.cuda.empty_cache()
                gc.collect()
            raise SkipImageException("No detections remaining after area filtering.")
        
        detection_list = filtered_detection_list
        detection_list = sorted(detection_list, key=lambda d: np.sum(d["mask"]), reverse=True)
        
        # Dilation
        if self.cfg.get("yoloe_mask_dilate_iterations", 0) > 0:
            kernel_size = self.cfg.get("yoloe_mask_dilate_kernel_size", 3)
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            iterations = self.cfg.get("yoloe_mask_dilate_iterations")
            for det_entry in detection_list:
                mask_uint8 = det_entry["mask"].astype(np.uint8)
                dilated_mask = cv2.dilate(mask_uint8, kernel, iterations=iterations)
                det_entry["mask"] = dilated_mask.astype(bool)
                det_entry["subtracted_mask"] = dilated_mask.astype(bool)

        # Crop detections
        if OSDSYNTH_AVAILABLE:
            detection_list = crop_detections_with_xyxy(self.cfg, image_rgb_pil, detection_list)
        else:
            for det_entry in detection_list:
                det_entry["image_crop"] = None
                det_entry["mask_crop"] = None
                det_entry["image_crop_modified"] = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        return detection_list

    def export_llm_logs(self, filepath: str = None) -> str:
        """Export all logged LLM prompts and responses to a file.
        
        Args:
            filepath: Path to save logs. If None, saves to timestamped file in llm_logs dir.
        
        Returns:
            Path where logs were saved
        """
        if not hasattr(self, 'prompt_logger'):
            self.logger.warning("Prompt logger not available.")
            return None
        
        if filepath is None:
            log_dir = self.cfg.get("llm_logs_dir", "./llm_logs")
            filepath = os.path.join(log_dir, f"exported_logs_{self.timestamp}.json")
        
        self.prompt_logger.export_logs(filepath)
        self.logger.info(f"LLM logs exported to: {filepath}")
        return filepath

    def _process_common(self, image_input, custom_vocabulary=None, **kwargs):
        """
        Main pipeline: 
        1. YOLOE Segmentation
        2. UniK3D Projection
        3. Extraction & Cleaning (Applying User Algorithm)
        4. Bounding Box & Data Creation
        """
        image_bgr = self._load_image(image_input)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        prefix = f"img_{self.timestamp}"

        # 1. Detection
        classes = list(self.yoloe_model.names.values())
        res = self.yoloe_model.predict(image_bgr, classes=None, conf=0.5)
        if not res or not res[0].boxes: raise SkipImageException("No objects")
        
        boxes, masks_raw, clss = res[0].boxes.xyxy.cpu().numpy(), res[0].masks.data.cpu().numpy(), res[0].boxes.cls.cpu().numpy()
        h, w = image_bgr.shape[:2]

        detection_list = []
        for i in range(len(boxes)):
            m = cv2.resize(masks_raw[i], (w, h), interpolation=cv2.INTER_NEAREST) > 0.5
            if np.sum(m) < 100: continue
            detection_list.append({
                "xyxy": boxes[i], "mask": m, "class_name": self.yoloe_model.names[int(clss[i])]
            })

        # 2. UniK3D Projection
        img_tensor = torch.from_numpy(image_rgb).permute(2,0,1).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            out = self.unik3d_model.infer(img_tensor, normalize=True)
        pts_global = out["points"].squeeze().permute(1,2,0).cpu().numpy() # H, W, 3

        # Wis3D Setup
        wis3d = Wis3D(self.cfg.wis3d_folder, prefix) if self.cfg.get("vis", False) else None
        if wis3d:
            wis3d.add_point_cloud(pts_global.reshape(-1,3), image_rgb.reshape(-1,3)/255.0, name="scene_raw")

        valid_detections = []
        
        # 3. Extraction & Downstream Cleaning
        for det in detection_list:
            # Mask points for this specific object
            obj_pts = pts_global[det["mask"]]
            obj_cols = image_rgb[det["mask"]] / 255.0
            
            if len(obj_pts) < 20: continue
            
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(obj_pts)
            pcd.colors = o3d.utility.Vector3dVector(obj_cols)
            
            # --- APPLY ALGORITHM HERE ---
            # This calls the updated function with Adaptive SOR + Fast Density + Normals
            cleaned_pcd = process_pcd_for_unik3d(self.cfg, pcd)
            
            # Check if cleaning removed too many points
            if len(cleaned_pcd.points) < 10: continue
            
            # 4. Downstream Tasks (BBox, Volume)
            # These are now based on the cleaned cloud
            aabb, obb = get_bounding_box_for_unik3d(self.cfg, cleaned_pcd)
            
            det["pcd"] = cleaned_pcd
            det["axis_aligned_bbox"] = aabb
            det["oriented_bbox"] = obb
            valid_detections.append(det)

        if not valid_detections: raise SkipImageException("No valid 3D objects after cleaning.")

        # Crops & Captioning
        crop_detections_with_xyxy(self.cfg, Image.fromarray(image_rgb), valid_detections)
        captioned = self.captioner.process_local_caption(valid_detections)

        # Final Object List
        objects = []
        for d in captioned:
            objects.append(DetectedObject(
                d.get("class_name"), d.get("caption"), d["mask"], d["xyxy"],
                d["pcd"], d["oriented_bbox"], d["axis_aligned_bbox"], d.get("image_crop")
            ))

        # Visuals
        # Define output path for the ply
        ply_output_path = os.path.join(
            self.cfg.get("log_dir", "./temp_outputs"), 
            f"{prefix}_scene_with_OBBs.ply"
        )
        
        # Flatten global points for export (H, W, 3) -> (N, 3)
        flat_points = pts_global.reshape(-1, 3)
        flat_colors = image_rgb.reshape(-1, 3) / 255.0

        target_class = "car"

        # Call the new export function
        export_target_class_solid_obbs(
            "highlighted_couch.ply", 
            flat_points, 
            flat_colors, 
            objects, # The list of DetectedObject instances
            target_class  # The target class
        )

        # Visuals
        # Define output path for the ply
        ply_output_path = os.path.join(
            self.cfg.get("log_dir", "./temp_outputs"), 
            f"{prefix}_camera_focus_visualization.ply"
        )

        # --- NEW CALL HERE ---
        # Change "couch" to whatever class you want to debug/visualize
        
        export_scene_with_camera_focus(
            ply_output_path, 
            flat_points, 
            flat_colors, 
            objects, 
            target_class
        )

        return objects, captioned, prefix
        
    def generate_detected_objects_from_image(self, image_input, custom_vocabulary=None, **kwargs):
        """Generate detected objects from image."""
        try:
            detected_objects_list, _, _ = self._process_common(image_input, custom_vocabulary, **kwargs)
            return detected_objects_list
        except SkipImageException as e:
            self.logger.warning(f"Object detection skipped: {e}")
            return []
        except Exception as e:
            self.logger.error(f"Error during object detection: {e}", exc_info=True)
            raise
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

    def generate_scene_narrative(self, captioned_detections_dicts: List[Dict], template_facts: List[str]) -> str:
        """Generate scene narrative using LLM."""
        if not self.llm_interface or not self.llm_interface.is_available():
            self.logger.warning("LLM not available for scene narrative.")
            obj_descs = [f"{det.get('class_name')} (described as: {det.get('caption')})"
                        for det in captioned_detections_dicts]
            narrative = "Objects detected: " + "; ".join(obj_descs) if obj_descs else "No objects detected."
            if template_facts:
                narrative += " Relationships: " + ". ".join(template_facts)
            return narrative.strip()

        narrative_system_prompt = (
            "You are describing visual scenes for visually impaired people. "
            "Generate a single, cohesive paragraph summarizing the scene based on detected objects and relationships."
        )

        object_details_parts = []
        if captioned_detections_dicts:
            for i, det_dict in enumerate(captioned_detections_dicts):
                object_details_parts.append(
                    f"<region{i}> is a {det_dict.get('class_name')} (described as: {det_dict.get('caption')})")
            object_details_str = "\n".join(object_details_parts)
        else:
            object_details_str = "No distinct objects identified."
        
        facts_str = ". ".join(template_facts) if template_facts else "No specific relationships identified."

        user_prompt_content = (
            f"Objects:\n{object_details_str}\n\n"
            f"Relationships:\n{facts_str}\n\n"
            f"Provide a narrative summary paragraph of the scene."
        )

        try:
            if isinstance(self.llm_interface, LocalHFLLM):
                import asyncio
                loop = asyncio.new_event_loop()
                narrative_text = loop.run_until_complete(
                    self.llm_interface.generate_answer(object_details_str, facts_str, self.logger)
                )
                loop.close()
            else:
                narrative_text = self.llm_interface.generate_answer(
                    object_details_str, facts_str, self.logger)
            
            return narrative_text if narrative_text else "Unable to generate narrative."
        
        except Exception as e:
            self.logger.error(f"Error generating narrative: {e}", exc_info=True)
            return "Error generating scene narrative."

    def generate_facts(self, image_input, custom_vocabulary=None, run_llm_rephrase=False, **kwargs):
        """Generate facts from image."""
        try:
            detected_objects_list, detection_list_dicts, filename_prefix = self._process_common(
                image_input, custom_vocabulary, **kwargs)
            
            if not detected_objects_list:
                return [], [], [], ""

            template_facts = self.fact_prompter.evaluate_predicates_on_pairs(detection_list_dicts)
            rephrased_qas = []
            
            if run_llm_rephrase and template_facts and self.llm_interface:
                llm_prompts = prepare_llm_prompts_from_facts(template_facts, detection_list_dicts)
                rephrased_qas = self._run_llm_rephrasing(llm_prompts)
            
            scene_narrative = ""
            if self.llm_interface and (detected_objects_list or template_facts):
                scene_narrative = self.generate_scene_narrative(detection_list_dicts, template_facts)

            return detected_objects_list, template_facts, rephrased_qas, scene_narrative
        
        except SkipImageException as e:
            self.logger.warning(f"Fact generation skipped: {e}")
            return [], [], [], ""
        except Exception as e:
            self.logger.error(f"Error during fact generation: {e}", exc_info=True)
            return [], [], [], ""
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

    def generate_qa(self, image_input, custom_vocabulary=None, **kwargs):
        """Generate QAs from image."""
        try:
            detected_objects_list, detection_list_dicts, filename_prefix = self._process_common(
                image_input, custom_vocabulary, **kwargs)
            
            if not detected_objects_list:
                return [], [], ""

            vqa_results = self.qa_prompter.evaluate_predicates_on_pairs(detection_list_dicts)
            template_qas = parse_qas_from_vqa_results(vqa_results)
            
            scene_narrative = ""
            if self.llm_interface and detected_objects_list:
                scene_narrative = self.generate_scene_narrative(detection_list_dicts, [])

            return detected_objects_list, template_qas, scene_narrative
        
        except SkipImageException as e:
            self.logger.warning(f"QA generation skipped: {e}")
            return [], [], ""
        except Exception as e:
            self.logger.error(f"Error during QA generation: {e}", exc_info=True)
            return [], [], ""
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

    def serialize_scene_graph(self, detected_objects: List[DetectedObject]) -> str:
        """Serialize detected objects to minimal JSON scene graph.
        
        Each object contains only:
        - label: object class name
        - centroid: 3D center point [x, y, z]
        - dimensions: [width, height, depth] of bounding box
        """
        if not detected_objects:
            return json.dumps({
                "objects": [],
                "timestamp": datetime.now().isoformat()
            }, indent=2)

        scene_data = []
        for idx, obj in enumerate(detected_objects):
            # Get 3D center
            centroid = obj.center.tolist()
            
            # Get 3D dimensions from oriented or axis-aligned bounding box
            if obj.bounding_box_3d_oriented and not obj.bounding_box_3d_oriented.is_empty():
                extent = obj.bounding_box_3d_oriented.extent
            elif obj.bounding_box_3d_axis_aligned and not obj.bounding_box_3d_axis_aligned.is_empty():
                extent = obj.bounding_box_3d_axis_aligned.get_extent()
            else:
                extent = [0, 0, 0]
            
            obj_data = {
                # "id": idx,
                "label": obj.class_name,
                "centroid": [round(x, 3) for x in centroid],
                "dimensions": [round(x, 3) for x in extent]
            }
            scene_data.append(obj_data)

        return json.dumps({
            "objects": scene_data,
            "timestamp": datetime.now().isoformat()
        }, indent=2)
    
    def serialize_scene_for_llm(self, detected_objects: List[DetectedObject]) -> str:
        """Generate LLM-optimized scene description."""
        lines = [f"SCENE: {len(detected_objects)} objects detected\n"]
        
        for idx, obj in enumerate(detected_objects):
            lines.append(f"OBJECT {idx}: {obj.class_name}")
            lines.append(f"  - Description: {obj.description}")
            
            # Center position
            center = obj.center
            center_rounded = [round(float(x), 2) for x in center]
            lines.append(f"  - Position: {center_rounded}")
            
            # Size/extent
            if obj.bounding_box_3d_oriented and not obj.bounding_box_3d_oriented.is_empty():
                extent = obj.bounding_box_3d_oriented.extent
            elif obj.bounding_box_3d_axis_aligned and not obj.bounding_box_3d_axis_aligned.is_empty():
                extent = obj.bounding_box_3d_axis_aligned.get_extent()
            else:
                extent = [0.0, 0.0, 0.0]
            
            size_rounded = [round(float(x), 2) for x in extent]
            lines.append(f"  - Size (W×H×D): {size_rounded}")
            
            # Volume
            lines.append(f"  - Volume: {round(obj.volume, 3)}m³\n")
        
        return "\n".join(lines)

    def process_image_for_interaction(self, image_input, mode="process") -> Dict:
        """Process image for interactive VLM/LLM applications."""
        result = {
            "type": "text_only",
            "content": "",
            "raw_image_b64": None
        }
        try:
            # 1. SGG Processing
            if mode in ["process", "both"]:
                try:
                    detected_objs, _, _ = self._process_common(image_input)
                    serialized_data = self.serialize_scene_for_llm(detected_objs)
                    print(serialized_data)
                    result["content"] = f"Structured Visual Scene Data (JSON):\n{serialized_data}\n"
                except SkipImageException:
                    result["content"] = "Visual Scan: No distinct objects found."

            # 2. Raw Encoding
            if mode in ["raw", "both"]:
                b64_str = encode_image_to_base64(image_input)
                result["raw_image_b64"] = b64_str
                result["type"] = "multimodal"

            return result

        except Exception as e:
            self.logger.error(f"Interaction Processing Error: {e}", exc_info=True)
            return {
                "type": "error", 
                "content": f"Error processing image: {str(e)}",
                "raw_image_b64": None
            }
    
    def _run_llm_rephrasing(self, llm_prompts):
        """Run LLM rephrasing for facts."""
        if not self.llm_interface or not self.llm_interface.is_available():
            self.logger.warning("LLM not available for rephrasing.")
            return []
        
        rephrased_conversations = []
        max_retries = self.cfg.get("llm_max_retries", 2)
        
        for user_prompt_text in llm_prompts:
            success = False
            
            for attempt in range(max_retries):
                try:
                    json_response = self.llm_interface.generate_json_qa(
                        user_prompt_text, LLM_HF_SYSTEM_PROMPT, self.logger)
                    
                    if not json_response:
                        continue
                    
                    question = json_response.get("Question", "").strip()
                    answer = json_response.get("Answer", "").strip()
                    
                    if question and answer:
                        rephrased_conversations.append((question, answer))
                        self.logger.info(f"Rephrased => Q: {question} || A: {answer}")
                        success = True
                        break
                
                except Exception as e:
                    self.logger.debug(f"Rephrasing attempt {attempt + 1} failed: {e}")
            
            if not success:
                self.logger.warning(f"Failed to rephrase: {user_prompt_text[:100]}")
        
        return rephrased_conversations

    def __del__(self):
        """Cleanup resources."""
        # Generate and save final summary before cleanup
        if hasattr(self, 'prompt_logger'):
            try:
                summary = self.prompt_logger.generate_summary()
                self.logger.info(f"LLM Logging Summary: {summary['total_requests']} total requests, "
                            f"{summary['successful_requests']} successful")
            except Exception as e:
                self.logger.warning(f"Error generating prompt logger summary: {e}")
        
        if hasattr(self, 'unik3d_model'):
            del self.unik3d_model
        if hasattr(self, 'yoloe_model'):
            del self.yoloe_model
        if hasattr(self, 'captioner'):
            del self.captioner
        if hasattr(self, 'qa_prompter'):
            del self.qa_prompter
        if hasattr(self, 'fact_prompter'):
            del self.fact_prompter
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            self.logger.info("Resources cleaned up.")
