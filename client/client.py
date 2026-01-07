import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import sounddevice as sd
import cv2
import requests
import threading
import queue
import signal
import uuid
import difflib
import re
import traceback
import logging
import struct
import io
import datetime
import asyncio
import websockets
import json
import base64
import pyaudio
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
import warnings
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Suppress warnings for cleaner console output
warnings.filterwarnings('ignore')

# ==============================================================================
# --- 1. LOGGING SETUP ---
# ==============================================================================
LOG_DIR = Path("/tmp/voice_assistant_logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / f"assistant.log"

logger = logging.getLogger("voice_assistant")
logger.setLevel(logging.DEBUG)
fmt = logging.Formatter("%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s", "%Y-%m-%d %H:%M:%S")

# Console Handler (Info level to keep it readable)
sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(fmt)
sh.setLevel(logging.INFO)

# File Handler (Debug level for troubleshooting)
fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
fh.setFormatter(fmt)
fh.setLevel(logging.DEBUG)

logger.handlers = []
logger.addHandler(sh)
logger.addHandler(fh)

def log_exc(label="Exception"):
    logger.error("%s: %s", label, traceback.format_exc())

# ==============================================================================
# --- 2. DEPENDENCY & GLOBAL CONFIG ---
# ==============================================================================

# Check for NVIDIA Riva Client
try:
    import riva.client
    import riva.client.audio_io
    from riva.client.argparse_utils import add_asr_config_argparse_parameters, add_connection_argparse_parameters
    RIVA_AVAILABLE = True
except ImportError:
    RIVA_AVAILABLE = False
    logger.critical("FATAL: NVIDIA Riva client not found. Please install: pip install nvidia-riva-client")

# Check for PiCamera2
try:
    from picamera2 import Picamera2
    PICAM2_AVAILABLE = True
except ImportError:
    PICAM2_AVAILABLE = False

# Configuration
SERVER_URL = "http://192.168.20.156:8443/interact/"
PIPER_TTS_URL = "ws://localhost:9090"  # Piper TTS WebSocket server URL
CAMERA_CONFIG = {'width': 640, 'height': 480, 'warmup_time': 2, 'index': 0}

# Inter-Thread Queues
TRANSCRIPT_QUEUE = queue.Queue()  # ASR -> Logic
TTS_QUEUE = queue.Queue()         # Logic -> TTS
ASR_CONTROL_QUEUE = queue.Queue() # Logic -> ASR (for restart signals)

# ==============================================================================
# --- 2.5 WAKE WORD DETECTION ---
# ==============================================================================

# ==============================================================================
# ENHANCED WAKE WORD DETECTION WITH AUTO-TRIMMING
# ==============================================================================

class WakeWordDetector:
    """
    Detects wake word/phrase using fuzzy matching.
    Activates the assistant for ONE command, then automatically deactivates.
    Automatically trims the wake word and any preceding text from the result.
    """
    def __init__(self, 
                 wake_phrases: List[str] = None,
                 similarity_threshold: float = 0.75,
                 logger_instance=None):
        """
        Args:
            wake_phrases: List of wake phrases to detect (default: ['xin chào'])
            similarity_threshold: Minimum similarity ratio to trigger (0.0-1.0)
        """
        self.wake_phrases = wake_phrases or ['xin chào', 'xin chao', 'chào', 'chao']
        self.similarity_threshold = similarity_threshold
        self.logger = logger_instance or logger
        
        self.is_active = False
        self.lock = threading.Lock()
        
        # Statistics
        self.stats = {
            "total_checked": 0,
            "activations": 0,
            "deactivations": 0,
            "trimmed_texts": 0
        }
        
        self.logger.info(f"[WakeWord] Initialized with phrases: {self.wake_phrases}")
        self.logger.info(f"[WakeWord] Threshold: {self.similarity_threshold}")
        self.logger.info(f"[WakeWord] Auto-trim enabled: YES")
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        # Remove extra whitespace and convert to lowercase
        text = ' '.join(text.lower().split())
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        return text
    
    def _calculate_similarity(self, text: str, wake_phrase: str) -> float:
        """Calculate similarity between text and wake phrase."""
        text_norm = self._normalize_text(text)
        phrase_norm = self._normalize_text(wake_phrase)
        
        # Check for exact substring match first
        if phrase_norm in text_norm:
            return 1.0
        
        # Use SequenceMatcher for fuzzy matching
        matcher = difflib.SequenceMatcher(None, text_norm, phrase_norm)
        return matcher.ratio()
    
    def _find_wake_word_position(self, text: str, wake_phrase: str) -> Optional[Tuple[int, int]]:
        """
        Find the position of wake word in text.
        Returns: (start_pos, end_pos) in the ORIGINAL text, or None if not found.
        """
        text_lower = text.lower()
        phrase_lower = wake_phrase.lower()
        
        # First try exact match in lowercase
        pos = text_lower.find(phrase_lower)
        if pos != -1:
            return (pos, pos + len(wake_phrase))
        
        # Try fuzzy matching by sliding window
        words = text.split()
        phrase_words = wake_phrase.split()
        
        for i in range(len(words) - len(phrase_words) + 1):
            window = ' '.join(words[i:i + len(phrase_words)])
            similarity = self._calculate_similarity(window, wake_phrase)
            
            if similarity >= self.similarity_threshold:
                # Found fuzzy match, calculate position in original text
                start_pos = len(' '.join(words[:i]))
                if i > 0:
                    start_pos += 1  # Account for space before
                end_pos = start_pos + len(window)
                return (start_pos, end_pos)
        
        return None
    
    def check_wake_word(self, text: str) -> Tuple[bool, float, Optional[str], Optional[Tuple[int, int]]]:
        """
        Check if text contains wake word.
        Returns: (detected, similarity, matched_phrase, position)
        where position is (start_pos, end_pos) in original text
        """
        if not text or not text.strip():
            return False, 0.0, None, None
        
        self.stats["total_checked"] += 1
        
        best_similarity = 0.0
        best_phrase = None
        best_position = None
        
        # Check against all wake phrases
        for phrase in self.wake_phrases:
            similarity = self._calculate_similarity(text, phrase)
            
            if similarity >= self.similarity_threshold:
                position = self._find_wake_word_position(text, phrase)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_phrase = phrase
                    best_position = position
        
        detected = best_similarity >= self.similarity_threshold
        
        if detected:
            self.logger.info(
                f"[WakeWord] Detected! Phrase: '{best_phrase}' "
                f"(similarity: {best_similarity:.2f}) at position {best_position} "
                f"in '{text[:50]}...'"
            )
        
        return detected, best_similarity, best_phrase, best_position
    
    def trim_wake_word(self, text: str) -> str:
        """
        Remove wake word and any text before it from the input.
        Returns the trimmed text (command only).
        
        Example:
            Input: "xin chào chụp ảnh" -> Output: "chụp ảnh"
            Input: "hello xin chào what's the weather" -> Output: "what's the weather"
        """
        detected, similarity, phrase, position = self.check_wake_word(text)
        
        if not detected or position is None:
            # No wake word found, return original text
            return text
        
        start_pos, end_pos = position
        
        # Get everything after the wake word
        trimmed = text[end_pos:].strip()
        
        self.stats["trimmed_texts"] += 1
        
        self.logger.info(
            f"[WakeWord] Trimmed: '{text}' -> '{trimmed}' "
            f"(removed: '{text[:end_pos]}')"
        )
        
        return trimmed
    
    def activate(self):
        """Activate the assistant for ONE command."""
        with self.lock:
            was_active = self.is_active
            self.is_active = True
            self.stats["activations"] += 1
        
        if not was_active:
            self.logger.info(f"[WakeWord] Assistant ACTIVATED (will process next command)")
    
    def deactivate(self):
        """Deactivate the assistant after command is processed."""
        with self.lock:
            if self.is_active:
                self.is_active = False
                self.stats["deactivations"] += 1
                self.logger.info("[WakeWord] Assistant DEACTIVATED (waiting for wake word)")
    
    def is_assistant_active(self) -> bool:
        """Check if assistant is currently active."""
        with self.lock:
            return self.is_active
    
    def process_transcript(self, text: str) -> Tuple[bool, str]:
        """
        Process transcript and activate if wake word detected.
        Returns: (should_process, cleaned_text)
        
        - should_process: True if assistant should process this text
        - cleaned_text: Text with wake word trimmed (if detected), or original text
        """
        # Check if wake word is in text
        detected, similarity, phrase, position = self.check_wake_word(text)
        
        if detected:
            # Activate and trim the wake word
            self.activate()
            cleaned_text = self.trim_wake_word(text)
            return True, cleaned_text
        
        # If not detected, check if assistant is already active
        if self.is_assistant_active():
            return True, text
        
        return False, text
    
    def get_stats(self) -> dict:
        """Get wake word statistics."""
        with self.lock:
            stats = self.stats.copy()
            stats["is_active"] = self.is_active
            return stats
        
# Global System State
class SystemState:
    def __init__(self):
        self.is_running = True
        self.is_speaking = False
        self.restart_asr = False
        self.lock = threading.Lock()
        self.current_session_id = str(uuid.uuid4())
        self.conversation_image = None  # Store the image for current conversation

    def set_speaking(self, speaking: bool):
        with self.lock:
            self.is_speaking = speaking

    def check_speaking(self):
        with self.lock:
            return self.is_speaking

    def request_asr_restart(self):
        """Request ASR engine to restart."""
        with self.lock:
            self.restart_asr = True
        logger.info("[SystemState] ASR restart requested")

    def check_and_clear_restart(self):
        """Check if restart is requested and clear the flag."""
        with self.lock:
            if self.restart_asr:
                self.restart_asr = False
                return True
            return False

    def start_new_conversation(self, image: Optional[np.ndarray] = None):
        """Start a new conversation with a fresh session ID and optional image."""
        with self.lock:
            self.current_session_id = str(uuid.uuid4())
            self.conversation_image = image
        logger.info(f"[SystemState] New conversation started: {self.current_session_id}")
        if image is not None:
            logger.info(f"[SystemState] Conversation image stored (shape: {image.shape})")

    def get_session_id(self) -> str:
        """Get current session ID."""
        with self.lock:
            return self.current_session_id

    def get_conversation_image(self) -> Optional[np.ndarray]:
        """Get the image associated with current conversation."""
        with self.lock:
            return self.conversation_image.copy() if self.conversation_image is not None else None

    def stop(self):
        with self.lock:
            self.is_running = False

state = SystemState()

# ==============================================================================
# --- 3. PIPER TTS INTEGRATION ---
# ==============================================================================

class PiperTTSClient:
    """
    Client for Piper TTS with real-time streaming via WebSocket.
    Adapted to work with threading instead of asyncio for compatibility.
    """
    def __init__(self, server_url="ws://localhost:9090"):
        self.server_url = server_url
        self.audio_queue = queue.Queue()
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.is_playing = False
        self.metadata = None
        self.loop = None
        self.ws_thread = None
        
    def audio_player_thread(self):
        """Thread to play audio from queue"""
        while self.is_playing:
            try:
                chunk = self.audio_queue.get(timeout=0.1)
                if chunk is None:  # End signal
                    break
                if self.stream:
                    self.stream.write(chunk)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"[PiperTTS] Audio playback error: {e}")
                break
        
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        logger.debug("[PiperTTS] Audio player thread stopped")
    
    async def _synthesize_async(self, text):
        """Async method to send text to server and receive audio stream"""
        try:
            logger.debug(f"[PiperTTS] Connecting to {self.server_url}...")
            
            async with websockets.connect(self.server_url) as websocket:
                logger.debug("[PiperTTS] Connected! Sending text for synthesis...")
                
                # Send synthesis request
                await websocket.send(json.dumps({
                    'type': 'synthesize',
                    'text': text
                }))
                
                # Start audio player thread
                self.is_playing = True
                state.set_speaking(True)
                player_thread = threading.Thread(target=self.audio_player_thread, daemon=True)
                player_thread.start()
                
                # Receive and process messages
                async for message in websocket:
                    data = json.loads(message)
                    msg_type = data.get('type')
                    
                    if msg_type == 'metadata':
                        # Initialize audio stream with metadata
                        self.metadata = data
                        logger.info(f"[PiperTTS] Audio format: {data['sample_rate']}Hz, {data['channels']} channel(s)")
                        
                        self.stream = self.p.open(
                            format=pyaudio.paInt16,
                            channels=data['channels'],
                            rate=data['sample_rate'],
                            output=True,
                            frames_per_buffer=1024
                        )
                        logger.debug("[PiperTTS] Playing audio...")
                        
                    elif msg_type == 'audio':
                        # Decode and queue audio chunk
                        audio_data = base64.b64decode(data['data'])
                        self.audio_queue.put(audio_data)
                        
                    elif msg_type == 'complete':
                        logger.debug("[PiperTTS] Synthesis complete")
                        self.audio_queue.put(None)  # Signal end
                        break
                        
                    elif msg_type == 'error':
                        logger.error(f"[PiperTTS] Error from server: {data['message']}")
                        self.is_playing = False
                        break
                
                # Wait for playback to finish
                player_thread.join()
                self.is_playing = False
                state.set_speaking(False)
                logger.debug("[PiperTTS] Playback finished")
                
        except ConnectionRefusedError:
            logger.error(f"[PiperTTS] Could not connect to server at {self.server_url}")
            logger.error("[PiperTTS] Make sure the Piper TTS server is running!")
            self.is_playing = False
            state.set_speaking(False)
        except Exception as e:
            logger.error(f"[PiperTTS] Synthesis error: {e}")
            log_exc("PiperTTS Synthesis")
            self.is_playing = False
            state.set_speaking(False)
    
    def synthesize(self, text):
        """
        Synchronous wrapper for async synthesize method.
        Runs in a new event loop in the current thread.
        """
        if not text.strip():
            return
        
        try:
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._synthesize_async(text))
            loop.close()
        except Exception as e:
            logger.error(f"[PiperTTS] Synthesis wrapper error: {e}")
            log_exc("PiperTTS Wrapper")
    
    def close(self):
        """Clean up resources"""
        self.is_playing = False
        if self.stream:
            try:
                self.stream.close()
            except Exception:
                pass
        try:
            self.p.terminate()
        except Exception:
            pass
        logger.debug("[PiperTTS] Client closed")


# ==============================================================================
# --- 4. IMPROVED FUZZY OVERLAP REMOVAL WITH TIME-AWARE STRATEGY ---
# ==============================================================================

class DiffStrategy(Enum):
    """Strategy for detecting new content in ASR results."""
    FUZZY_MATCH = "fuzzy_match"
    TIME_BASED = "time_based"
    COMBINED = "combined"
    EXPIRED = "expired"


@dataclass
class OverlapAnalysis:
    """Result of overlap analysis."""
    strategy_used: str
    match_ratio: float
    cut_position: int
    new_content: str
    confidence: float
    debug_info: dict


class ImprovedFuzzyOverlapManager:
    """Time-Aware Fuzzy Overlap Removal for ASR Streams with Decoder History."""
    
    def __init__(
        self,
        fuzzy_threshold: float = 0.70,
        expiry_seconds: float = 15.0,
        time_based_threshold_seconds: float = 2.0,
        min_new_content_chars: int = 3,
        logger_instance=None
    ):
        self.fuzzy_threshold = fuzzy_threshold
        self.expiry_seconds = expiry_seconds
        self.time_based_threshold_seconds = time_based_threshold_seconds
        self.min_new_content_chars = min_new_content_chars
        self.logger = logger_instance or logger
        
        # Last executed command storage
        self.last_executed_raw = ""
        self.last_executed_time = 0.0
        self.last_executed_length = 0
        
        # Statistics for debugging
        self.stats = {
            "total_processed": 0,
            "cleaned": 0,
            "time_based_cutoff": 0,
            "fuzzy_matched": 0,
            "no_overlap": 0,
            "expired": 0
        }

    def record_executed_command(self, raw_text: str):
        """Record a command that has been executed and acted upon."""
        self.last_executed_raw = raw_text.strip()
        self.last_executed_time = time.time()
        self.last_executed_length = len(self.last_executed_raw)
        
        self.logger.info(
            f"[OverlapMgr] Command recorded: '{self.last_executed_raw[:60]}...' "
            f"({self.last_executed_length} chars)"
        )

    def _calculate_match_ratio(self, text1: str, text2: str) -> float:
        """Calculate how similar two texts are using SequenceMatcher."""
        matcher = difflib.SequenceMatcher(None, text1, text2)
        return matcher.ratio()

    def _find_best_match(
        self,
        last_text: str,
        new_text: str
    ) -> Tuple[int, float, int]:
        """Find where the last executed command appears in the new text."""
        if not last_text or not new_text or len(last_text) > len(new_text):
            return 0, 0.0, 0
        
        best_ratio = 0.0
        best_start = 0
        best_end = 0
        
        window_size = len(last_text)
        
        for i in range(len(new_text) - window_size + 1):
            window = new_text[i:i + window_size]
            ratio = self._calculate_match_ratio(last_text, window)
            
            if ratio > best_ratio or (ratio == best_ratio and i < best_start):
                best_ratio = ratio
                best_start = i
                best_end = i + window_size
        
        return best_start, best_ratio, best_end

    def _time_since_last_command(self) -> float:
        """Get seconds since last command was executed."""
        if not self.last_executed_time:
            return float('inf')
        return time.time() - self.last_executed_time

    def _is_expired(self) -> bool:
        """Check if last command memory has expired."""
        return self._time_since_last_command() > self.expiry_seconds

    def analyze_overlap(self, new_text: str) -> OverlapAnalysis:
        """Analyze overlap between new ASR result and last executed command."""
        new_text = new_text.strip()
        self.stats["total_processed"] += 1
        
        debug_info = {
            "new_text_length": len(new_text),
            "last_executed_length": self.last_executed_length,
            "time_since_last": self._time_since_last_command(),
            "expired": self._is_expired()
        }
        
        if not self.last_executed_raw or self._is_expired():
            self.stats["expired"] += 1
            return OverlapAnalysis(
                strategy_used="expired",
                match_ratio=0.0,
                cut_position=0,
                new_content=new_text,
                confidence=1.0,
                debug_info=debug_info
            )
        
        match_start, match_ratio, match_end = self._find_best_match(
            self.last_executed_raw,
            new_text
        )
        
        time_since = self._time_since_last_command()
        use_time_based = time_since > self.time_based_threshold_seconds
        
        debug_info.update({
            "match_start": match_start,
            "match_end": match_end,
            "match_ratio": match_ratio,
            "use_time_based": use_time_based
        })
        
        overall_similarity = self._calculate_match_ratio(self.last_executed_raw, new_text)
        if overall_similarity > 0.95:
            strategy = DiffStrategy.COMBINED
            return OverlapAnalysis(
                strategy_used="near_identical",
                match_ratio=overall_similarity,
                cut_position=len(new_text),
                new_content="",
                confidence=0.95,
                debug_info=debug_info
            )
        
        if use_time_based and match_ratio >= 0.70:
            strategy = DiffStrategy.TIME_BASED
            cutoff_chars = int(self.last_executed_length * 0.85)
            cut_position = min(cutoff_chars, len(new_text))
            new_content = new_text[cut_position:].strip()
            confidence = 0.75
            self.stats["time_based_cutoff"] += 1
            
        elif match_ratio >= self.fuzzy_threshold:
            strategy = DiffStrategy.FUZZY_MATCH
            cut_position = match_end
            new_content = new_text[cut_position:].strip()
            confidence = min(1.0, match_ratio)
            self.stats["fuzzy_matched"] += 1
            
        else:
            strategy = DiffStrategy.COMBINED
            cut_position = 0
            new_content = new_text
            confidence = 1.0
            self.stats["no_overlap"] += 1
        
        if new_content and len(new_content) >= self.min_new_content_chars:
            self.stats["cleaned"] += 1
        
        debug_info.update({
            "strategy": strategy.value,
            "cut_position": cut_position,
            "new_content_length": len(new_content),
            "confidence": confidence
        })
        
        return OverlapAnalysis(
            strategy_used=strategy.value,
            match_ratio=match_ratio,
            cut_position=cut_position,
            new_content=new_content,
            confidence=confidence,
            debug_info=debug_info
        )

    def clean(self, text: str) -> Tuple[str, float]:
        """Clean ASR text by removing overlap with last executed command."""
        analysis = self.analyze_overlap(text)
        
        if analysis.new_content:
            self.logger.debug(
                f"[OverlapMgr] Clean via {analysis.strategy_used} "
                f"(ratio={analysis.match_ratio:.2f}, conf={analysis.confidence:.2f}): "
                f"'{analysis.new_content[:50]}...'"
            )
        
        return analysis.new_content, analysis.confidence

    def get_stats(self) -> dict:
        """Get statistics about overlap removal."""
        return self.stats.copy()

    def reset_stats(self):
        """Reset statistics counters."""
        for key in self.stats:
            self.stats[key] = 0

    def clear_memory(self):
        """Clear the last executed command memory."""
        self.last_executed_raw = ""
        self.last_executed_time = 0.0
        self.last_executed_length = 0
        self.logger.debug("[OverlapMgr] Memory cleared")

# ==============================================================================
# --- 5. NLP & INTENT CLASSIFICATION (SERVER-SIDE) ---
# ==============================================================================

class UserIntent(Enum):
    REQUEST_IMAGE = "request_image"
    ASK_QUESTION = "ask_question"
    UNKNOWN = "unknown"


class ServerIntentClassifier:
    """Intent classification using server-side /detect_intent/ endpoint."""
    
    def __init__(
        self,
        server_url: str,
        timeout: int = 10,
        logger_instance=None
    ):
        # Extract base URL from server_url (remove /interact/ if present)
        self.base_url = server_url.rsplit("/interact/", 1)[0]
        self.intent_url = f"{self.base_url}/detect_intent/"
        self.timeout = timeout
        self.logger = logger_instance or logger
        
        self.logger.info(f"ServerIntentClassifier initialized: {self.intent_url}")
    
    def _map_intent_string_to_enum(self, intent_str: str) -> UserIntent:
        """Map server intent string to UserIntent enum."""
        intent_map = {
            "REQUEST_IMAGE": UserIntent.REQUEST_IMAGE,
            "ASK_QUESTION": UserIntent.ASK_QUESTION,
            "UNKNOWN": UserIntent.UNKNOWN
        }
        return intent_map.get(intent_str.upper(), UserIntent.UNKNOWN)
    
    def classify(self, user_input: str) -> Tuple[UserIntent, float, str]:
        """Classify user intent using server endpoint."""
        if not user_input or not user_input.strip():
            return UserIntent.UNKNOWN, 0.0, "empty_input"
        
        try:
            payload = {"text_input": user_input}
            response = requests.post(
                self.intent_url,
                data=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            intent_str = result.get("intent", "UNKNOWN")
            confidence = float(result.get("confidence", 0.5))
            
            confidence = max(0.0, min(1.0, confidence))
            intent = self._map_intent_string_to_enum(intent_str)
            
            self.logger.info(
                f"Intent classification: {intent.name} (confidence: {confidence:.2f}) | "
                f"Input: '{user_input[:50]}...'"
            )
            
            return intent, confidence, "server_side"
            
        except requests.exceptions.Timeout:
            self.logger.error("Server intent detection timed out")
            return UserIntent.UNKNOWN, 0.5, "timeout"
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Server intent detection failed: {e}")
            return UserIntent.UNKNOWN, 0.5, "api_error"
        except Exception as e:
            self.logger.error(f"Intent classification error: {e}")
            return UserIntent.UNKNOWN, 0.5, "error"


class FastIntentClassifier:
    """Fast fallback intent classifier using keyword matching (no network)."""
    
    def __init__(self):
        self.image_keywords = [
            'chụp', 'ảnh', 'photo', 'picture', 'camera', 'hình',
            'screenshot', 'capture', 'tấm hình',
            'chụp ảnh', 'lấy ảnh', 'chụp hình'
        ]
        self.question_keywords = [
            'gì', 'nào', 'như thế nào', 'tại sao', 'khi nào',
            'ai', 'ở đâu', 'bao nhiêu', 'cái gì', 'what', 'why',
            'when', 'where', 'how', 'who', 'hỏi', 'bảo', '?',
            'kể chuyện', 'giải thích', 'tính toán', 'thời tiết'
        ]
    
    def classify(self, text: str) -> Tuple[UserIntent, float, str]:
        """Fast classification without network call."""
        text_lower = text.lower().strip()
        
        # Check for image request keywords
        image_matches = sum(1 for kw in self.image_keywords if kw in text_lower)
        # Check for question keywords
        question_matches = sum(1 for kw in self.question_keywords if kw in text_lower)
        
        if image_matches > 0 and image_matches >= question_matches:
            confidence = min(0.95, 0.7 + (image_matches * 0.1))
            return UserIntent.REQUEST_IMAGE, confidence, "fast_keyword"
        elif question_matches > 0:
            confidence = min(0.95, 0.6 + (question_matches * 0.1))
            return UserIntent.ASK_QUESTION, confidence, "fast_keyword"
        
        return UserIntent.UNKNOWN, 0.3, "fast_keyword"


class LinguisticCompletionDetector:
    """Detects when user has finished speaking based on linguistic stability."""
    
    def __init__(self, stability_window=0.7):
        self.stability_window = stability_window
        self.last_text = ""
        self.stable_since = 0.0

    def update(self, text: str) -> bool:
        """Update with new text and return True if text is stable."""
        if not text:
            return False
        
        if text == self.last_text:
            if (time.time() - self.stable_since) >= self.stability_window:
                return True
        else:
            self.last_text = text
            self.stable_since = time.time()
        return False

    def reset(self):
        """Reset detector state."""
        self.last_text = ""
        self.stable_since = 0.0

# ==============================================================================
# --- 6. HARDWARE ABSTRACTION (CAMERA) ---
# ==============================================================================
class CameraManager:
    def __init__(self, config): 
        self.config = config
    def initialize(self) -> bool: 
        raise NotImplementedError
    def capture(self) -> Optional[np.ndarray]: 
        raise NotImplementedError
    def cleanup(self): 
        pass

class PiCameraManager(CameraManager):
    def __init__(self, config):
        super().__init__(config)
        self.picam2 = None
    
    def initialize(self) -> bool:
        if not PICAM2_AVAILABLE: 
            return False
        try:
            self.picam2 = Picamera2()
            cam_config = self.picam2.create_preview_configuration(
                main={
                    "size": (self.config['width'], self.config['height']), 
                    "format": "RGB888"
                }
            )
            self.picam2.configure(cam_config)
            self.picam2.start()
            time.sleep(self.config['warmup_time'])
            return True
        except Exception as e:
            logger.error(f"PiCamera Error: {e}")
            return False
    
    def capture(self) -> Optional[np.ndarray]:
        if not self.picam2:
            return None
        
        # Capture the frame (RGB888 format from PiCamera2)
        frame = self.picam2.capture_array()
        
        # Flip vertically to correct upside-down camera
        frame = cv2.flip(frame, -1)  # 0 = flip around x-axis (vertical flip)
        
        # Convert RGB to BGR for OpenCV compatibility
        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        return frame
    
    def cleanup(self):
        if self.picam2: 
            self.picam2.stop()
            self.picam2.close()
class OpenCVCameraManager(CameraManager):
    def __init__(self, config):
        super().__init__(config)
        self.cap = None
    def initialize(self) -> bool:
        try:
            self.cap = cv2.VideoCapture(self.config['index'])
            if not self.cap.isOpened(): return False
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['width'])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['height'])
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            time.sleep(self.config['warmup_time'])
            return True
        except Exception as e:
            logger.error(f"OpenCV Error: {e}")
            return False
    
    def capture_fresh(self, flush_count: int = 10) -> Optional[np.ndarray]:
        """Capture the freshest frame by flushing buffer."""
        if not self.cap: 
            return None
        
        for _ in range(flush_count):
            ret, _ = self.cap.read()
            if not ret:
                break
        
        ret, frame = self.cap.read()
        return frame if ret else None
    
    def capture(self) -> Optional[np.ndarray]:
        if not self.cap: return None
        ret, frame = self.cap.read()
        return frame if ret else None
    def cleanup(self):
        if self.cap: self.cap.release()

# ==============================================================================
# --- 7. IMAGE EXPORT UTILITY ---
# ==============================================================================
class ImageExporter:
    """Utility for exporting captured images to local directory for debugging."""
    
    def __init__(self, export_dir: Optional[str] = None, enabled: bool = False):
        self.enabled = enabled
        if export_dir:
            self.export_dir = Path(export_dir)
        else:
            self.export_dir = Path("/tmp/voice_assistant_images")
        
        if self.enabled:
            self.export_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Image export enabled: {self.export_dir}")
    
    def export_image(self, frame: np.ndarray, label: str = "capture") -> Optional[str]:
        """Export an image frame to local directory."""
        if not self.enabled or frame is None:
            return None
        
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"{label}_{timestamp}.jpg"
            filepath = self.export_dir / filename
            
            success = cv2.imwrite(str(filepath), frame)
            if success:
                logger.info(f"[ImageExport] Saved image: {filepath}")
                return str(filepath)
            else:
                logger.warning(f"[ImageExport] Failed to write image: {filepath}")
                return None
        except Exception as e:
            logger.error(f"[ImageExport] Error exporting image: {e}")
            return None

# ==============================================================================
# --- 8. SERVER CLIENT ---
# ==============================================================================

class ServerClient:
    """Handles communication with the central AI server."""
    
    def __init__(self, url: str, timeout: int = 30):
        self.url = url
        self.base_url = url.rsplit("/interact/", 1)[0]
        self.timeout = timeout
        self.session = self._create_session()

    def _create_session(self) -> requests.Session:
        """Create a requests session with retry strategy."""
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def send_text_query(self, text: str, session_id: str) -> Tuple[str, float]:
        """Send text query to server. Returns (reply_text, latency_ms)"""
        start = time.time()
        try:
            payload = {"session_id": session_id, "text_input": text}
            resp = self.session.post(self.url, data=payload, timeout=self.timeout)
            resp.raise_for_status()
            reply = resp.json().get("reply", "")
            return reply, (time.time() - start) * 1000
        except requests.exceptions.Timeout:
            logger.error("[ServerClient] Text query timed out")
            return "", (time.time() - start) * 1000
        except requests.exceptions.RequestException as e:
            logger.error(f"[ServerClient] Text query failed: {e}")
            return "", (time.time() - start) * 1000
        except Exception as e:
            logger.error(f"[ServerClient] Unexpected error: {e}")
            return "", (time.time() - start) * 1000

    def send_image_query(self, image: np.ndarray, text: str, session_id: str) -> Tuple[str, float]:
        """Send image query to server. Returns (reply_text, latency_ms)"""
        start = time.time()
        try:
            # Encode image
            success, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not success:
                logger.error("[ServerClient] Image encoding failed")
                return "Encoding failed", 0.0
            
            files = {'file': ('image.jpg', encoded.tobytes(), 'image/jpeg')}
            data = {"session_id": session_id, "text_input": text, "image_mode": "both"}
            
            resp = self.session.post(self.url, data=data, files=files, timeout=self.timeout)
            resp.raise_for_status()
            reply = resp.json().get("reply", "")
            return reply, (time.time() - start) * 1000
        except requests.exceptions.Timeout:
            logger.error("[ServerClient] Image query timed out")
            return "", (time.time() - start) * 1000
        except requests.exceptions.RequestException as e:
            logger.error(f"[ServerClient] Image query failed: {e}")
            return "", (time.time() - start) * 1000
        except Exception as e:
            logger.error(f"[ServerClient] Unexpected error: {e}")
            return "", (time.time() - start) * 1000

# ==============================================================================
# --- 9. THREADS ---
# ==============================================================================

# --- Thread A: Async ASR Engine (Producer) with Restart Capability ---
class AsyncASREngine(threading.Thread):
    """
    Keeps microphone open perpetually. Streams audio to Riva.
    Pushes transcripts to TRANSCRIPT_QUEUE.
    Supports restart mechanism: stops current stream and starts fresh.
    """
    def __init__(self, auth, args):
        super().__init__(name="ASR-Engine")
        self.auth = auth
        self.args = args
        self.daemon = True
        self.current_stream = None
        self.stream_lock = threading.Lock()

    def run(self):
        logger.info("ASR Engine started. Listening...")
        while state.is_running:
            try:
                if state.check_and_clear_restart():
                    logger.info("[ASR] Restart requested. Stopping current stream...")
                    self._stop_current_stream()
                    time.sleep(0.5)
                    logger.info("[ASR] Restarting ASR stream...")
                
                self._streaming_loop()
            except Exception:
                log_exc("ASR Stream Error")
                time.sleep(2.0)

    def _stop_current_stream(self):
        """Stop the current audio stream cleanly."""
        with self.stream_lock:
            if self.current_stream:
                try:
                    self.current_stream.__exit__(None, None, None)
                    logger.debug("[ASR] Current stream stopped")
                except Exception as e:
                    logger.warning(f"[ASR] Error stopping stream: {e}")
                finally:
                    self.current_stream = None

    def _streaming_loop(self):
        asr_service = riva.client.ASRService(self.auth)
        config = riva.client.StreamingRecognitionConfig(
            config=riva.client.RecognitionConfig(
                encoding=riva.client.AudioEncoding.LINEAR_PCM,
                language_code=self.args.language_code,
                max_alternatives=1,
                enable_automatic_punctuation=True,
                sample_rate_hertz=self.args.sample_rate_hz,
            ),
            interim_results=True,
        )
        riva.client.add_endpoint_parameters_to_config(
            config,
            start_history=300,
            start_threshold=0.5,
            stop_history=1000,
            stop_history_eou=500,
            stop_threshold=0.3,
            stop_threshold_eou=0.5
        )
        
        with riva.client.audio_io.MicrophoneStream(
            rate=self.args.sample_rate_hz,
            chunk=self.args.file_streaming_chunk,
            device=self.args.input_device
        ) as audio_chunk_iterator:
            
            with self.stream_lock:
                self.current_stream = audio_chunk_iterator
            
            response_generator = asr_service.streaming_response_generator(
                audio_chunks=audio_chunk_iterator,
                streaming_config=config
            )

            for response in response_generator:
                if not state.is_running:
                    break
                
                if state.restart_asr:
                    logger.info("[ASR] Restart signal detected, breaking stream loop")
                    break
                
                if not response.results:
                    continue

                result = response.results[0]
                transcript = result.alternatives[0].transcript
                is_final = result.is_final

                TRANSCRIPT_QUEUE.put({
                    "text": transcript,
                    "is_final": is_final,
                    "timestamp": time.time()
                })
            
            with self.stream_lock:
                self.current_stream = None

# --- Thread B: TTS Worker (Consumer) ---
class TTSWorker(threading.Thread):
    """Handles text-to-speech via Piper TTS (WebSocket streaming)."""
    def __init__(self, tts_client: PiperTTSClient):
        super().__init__(name="TTS-Worker")
        self.tts_client = tts_client
        self.daemon = True

    def run(self):
        logger.info("TTS Worker ready.")
        while state.is_running:
            try:
                text = TTS_QUEUE.get(timeout=0.5)
                logger.info(f"🤖 Speaking: {text[:50]}...")
                
                # Synchronous call that blocks until TTS completes
                self.tts_client.synthesize(text)
                
                TTS_QUEUE.task_done()

            except queue.Empty:
                continue
            except Exception:
                log_exc("TTS Execution Error")

# --- Thread C: Logic Processor (Orchestrator) ---
# ==============================================================================
# UPDATED LOGIC PROCESSOR WITH WAKE WORD TRIMMING
# ==============================================================================

class LogicProcessor(threading.Thread):
    """
    The Brain. Uses wake word detection with auto-trimming before processing commands.
    """
    def __init__(self, wake_word_detector, server_classifier, fast_classifier, 
                 server_client, camera_mgr, image_exporter, consecutive_unknown_threshold=3):
        super().__init__(name="Logic-Processor")
        self.wake_word_detector = wake_word_detector
        self.server_classifier = server_classifier
        self.fast_classifier = fast_classifier
        self.server_client = server_client
        self.camera = camera_mgr
        self.image_exporter = image_exporter
        self.scanner = LinguisticCompletionDetector()
        self.overlap_mgr = ImprovedFuzzyOverlapManager(
            fuzzy_threshold=0.70,
            expiry_seconds=15.0,
            time_based_threshold_seconds=2.0,
            min_new_content_chars=3,
            logger_instance=logger
        )
        self.consecutive_unknown_threshold = consecutive_unknown_threshold
        self.consecutive_unknown_count = 0
        self.daemon = True

    def run(self):
        logger.info("Logic Processor ready. Waiting for wake word...")
        while state.is_running:
            try:
                payload = TRANSCRIPT_QUEUE.get(timeout=1.0)
            except queue.Empty:
                continue

            raw_text = payload['text']
            is_final = payload['is_final']

            # Skip processing if TTS is speaking
            if state.check_speaking():
                self.scanner.reset()
                continue

            # WAKE WORD DETECTION WITH AUTO-TRIMMING - Primary gate
            should_process, cleaned_text = self.wake_word_detector.process_transcript(raw_text)
            
            if not should_process:
                # Not active, just log and skip
                logger.debug(f"[Logic] Inactive (no wake word): '{raw_text[:30]}...'")
                continue
            
            # If we reach here, assistant is active and text is cleaned of wake word
            logger.debug(f"[Logic] Processing (ACTIVE): '{cleaned_text[:50]}...'")
            
            # Apply overlap manager to the already-cleaned text
            final_text, confidence = self.overlap_mgr.clean(cleaned_text)
            if not final_text:
                continue

            should_execute = False
            
            if is_final:
                logger.info(f"✓ ASR Final (ACTIVE): '{final_text}'")
                should_execute = True
            elif self.scanner.update(final_text):
                logger.info(f"✓ ASR Stable (ACTIVE): '{final_text}'")
                should_execute = True

            if should_execute:
                executed = self.process_command(final_text)
                if executed:
                    self.overlap_mgr.record_executed_command(raw_text)
                    logger.info(f"[Logic] Command executed. Deactivating assistant and restarting ASR...")
                    self.consecutive_unknown_count = 0
                    
                    # Deactivate wake word after successful command
                    self.wake_word_detector.deactivate()
                    
                    state.request_asr_restart()
                    self.scanner.reset()
                    self._drain_queue()
                else:
                    self.consecutive_unknown_count += 1
                    logger.debug(f"[Logic] UNKNOWN intent ({self.consecutive_unknown_count}/{self.consecutive_unknown_threshold})")
                    
                    if self.consecutive_unknown_count >= self.consecutive_unknown_threshold:
                        logger.warning(f"[Logic] Consecutive UNKNOWN limit reached. Deactivating and restarting ASR...")
                        self.consecutive_unknown_count = 0
                        
                        # Deactivate wake word after failed attempts
                        self.wake_word_detector.deactivate()
                        
                        state.request_asr_restart()
                        self.scanner.reset()
                        self._drain_queue()
                        self.speak("Xin lỗi, tôi không hiểu. Vui lòng nói lại.")

    def _drain_queue(self):
        """Drain remaining items in queue."""
        while not TRANSCRIPT_QUEUE.empty():
            try:
                TRANSCRIPT_QUEUE.get_nowait()
            except queue.Empty:
                break
        logger.debug("[Logic] Queue drained after command execution")

    def process_command(self, text):
        """Classify intent using server-side classifier only."""
        
        # Use server-side classification exclusively
        try:
            intent, confidence, method = self.server_classifier.classify(text)
            logger.info(f"[Intent] {intent.name} ({confidence:.2f}) via {method}")
        except Exception as e:
            logger.error(f"[Intent] Server classification failed: {e}")
            # If server fails, treat as unknown and return False
            return False
        
        # Require minimum confidence
        if confidence < 0.60:
            logger.debug(f"[Intent] Confidence too low ({confidence:.2f}), ignoring")
            return False

        if intent == UserIntent.REQUEST_IMAGE:
            self.action_camera(text)
            return True
        elif intent == UserIntent.ASK_QUESTION:
            self.action_question(text)
            return True
        
        return False

    def speak(self, text):
        TTS_QUEUE.put(text)

    def action_camera(self, text):
        self.speak("Được, đang chụp.")
        
        if not self.camera:
            self.speak("Lỗi: Không tìm thấy camera.")
            return

        self._drain_queue()
        
        if hasattr(self.camera, 'capture_fresh'):
            logger.info("[Action] Using fresh frame capture with buffer flush")
            frame = self.camera.capture_fresh(flush_count=15)
        else:
            frame = self.camera.capture()
        
        if frame is not None:
            export_path = self.image_exporter.export_image(frame, label="request_image")
            if export_path:
                logger.info(f"[Action] Image exported to: {export_path}")
            
            state.start_new_conversation(image=frame)
            logger.info(f"[Action] New conversation initiated with image, session: {state.get_session_id()}")
            
            # Add length constraint to prompt
            prompt_with_constraint = f"Mô tả cảnh này. [Trả lời ngắn gọn trong khoảng 50 từ]"
            
            threading.Thread(
                target=self._network_send_image,
                args=(frame, prompt_with_constraint),
                daemon=True
            ).start()
        else:
            self.speak("Lỗi: Không chụp được ảnh.")

    def action_question(self, text):
        self.speak("Để tôi xem.")
        
        # Add length constraint to prompt
        prompt_with_constraint = f"{text} [Trả lời ngắn gọn trong khoảng 50 từ]"
        
        threading.Thread(
            target=self._network_send_text,
            args=(prompt_with_constraint,),
            daemon=True
        ).start()

    def _network_send_image(self, image, text):
        """Send image to server using ServerClient."""
        try:
            session_id = state.get_session_id()
            logger.debug(f"[Network] Sending image query for session: {session_id[:8]}...")
            
            reply, latency = self.server_client.send_image_query(image, text, session_id)
            
            logger.info(f"[Network] Image query responded in {latency:.0f}ms")
            
            if reply:
                logger.debug(f"[Network] Server reply: {reply[:100]}...")
                self.speak(reply)
            else:
                logger.warning("[Network] Server returned empty reply")
                self.speak("Xin lỗi, tôi không nhận được phản hồi từ máy chủ.")
                
        except Exception as e:
            logger.error(f"[Network] Image send error: {e}")
            self.speak("Tôi gặp lỗi khi gửi ảnh.")

    def _network_send_text(self, text):
        """Send text to server using ServerClient."""
        try:
            session_id = state.get_session_id()
            logger.debug(f"[Network] Sending text query: {text[:50]}... (session: {session_id[:8]})")
            
            reply, latency = self.server_client.send_text_query(text, session_id)
            
            logger.info(f"[Network] Text query responded in {latency:.0f}ms")
            
            if reply:
                logger.debug(f"[Network] Server reply: {reply[:100]}...")
                self.speak(reply)
            else:
                logger.warning("[Network] Server returned empty reply")
                self.speak("Xin lỗi, tôi không nhận được phản hồi từ máy chủ.")
                
        except Exception as e:
            logger.error(f"[Network] Text send error: {e}")
            self.speak("Tôi gặp lỗi kết nối.")

    def print_stats(self):
        """Print overlap manager and wake word statistics."""
        # Overlap manager stats
        stats = self.overlap_mgr.get_stats()
        logger.info("=== Overlap Manager Statistics ===")
        logger.info(f"Total processed: {stats['total_processed']}")
        logger.info(f"Cleaned: {stats['cleaned']}")
        logger.info(f"Fuzzy matched: {stats['fuzzy_matched']}")
        logger.info(f"Time-based cutoff: {stats['time_based_cutoff']}")
        logger.info(f"No overlap: {stats['no_overlap']}")
        logger.info(f"Expired: {stats['expired']}")
        
        # Wake word stats
        wake_stats = self.wake_word_detector.get_stats()
        logger.info("=== Wake Word Detector Statistics ===")
        logger.info(f"Total checked: {wake_stats['total_checked']}")
        logger.info(f"Activations: {wake_stats['activations']}")
        logger.info(f"Deactivations: {wake_stats['deactivations']}")
        logger.info(f"Trimmed texts: {wake_stats['trimmed_texts']}")
        logger.info(f"Currently active: {wake_stats['is_active']}")

# ==============================================================================
# --- 10. MAIN ENTRY POINT ---
# ==============================================================================
def main():
    if not RIVA_AVAILABLE:
        print("FATAL: NVIDIA Riva Client library is missing.")
        sys.exit(1)

    default_device_info = riva.client.audio_io.get_default_input_device_info()
    default_device_index = None if default_device_info is None else default_device_info['index']

    parser = argparse.ArgumentParser(
        description="Async Riva Voice Assistant with Wake Word Detection and Piper TTS",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--input-device", type=int, default=default_device_index, help="Audio Input Device ID")
    parser.add_argument("--list-devices", action="store_true", help="List available audio devices")
    parser.add_argument("--camera-type", default="auto", choices=['auto', 'picamera', 'opencv'], help="Camera backend")
    parser.add_argument("--sample-rate-hz", type=int, default=16000, help="Microphone sample rate")
    parser.add_argument("--file-streaming-chunk", type=int, default=1600, help="Riva chunk size")
    parser.add_argument("--server-url", type=str, default=SERVER_URL, help="Backend server URL")
    parser.add_argument("--piper-url", type=str, default=PIPER_TTS_URL, help="Piper TTS WebSocket server URL")
    parser.add_argument("--debug-export-images", action="store_true",
                        help="Enable image export to local directory for debugging")
    parser.add_argument("--export-dir", type=str, default=None,
                        help="Directory to export captured images (default: /tmp/voice_assistant_images)")
    
    # Wake word arguments
    parser.add_argument("--wake-phrases", nargs="+", 
                        default=['xin chào', 'xin chao', 'chào', 'chao'],
                        help="Wake phrases to activate assistant")
    parser.add_argument("--wake-threshold", type=float, default=0.75,
                        help="Wake word similarity threshold (0.0-1.0)")
    
    # Overlap manager arguments
    parser.add_argument("--overlap-fuzzy-threshold", type=float, default=0.70,
                        help="Minimum match ratio to consider as overlap (0.0-1.0)")
    parser.add_argument("--overlap-expiry-seconds", type=float, default=15.0,
                        help="How long to remember last command for overlap removal")
    parser.add_argument("--overlap-time-threshold", type=float, default=2.0,
                        help="Time threshold for time-based overlap detection (seconds)")
    parser.add_argument("--overlap-min-new-chars", type=int, default=3,
                        help="Minimum characters needed to consider as new content")
    parser.add_argument("--intent-timeout", type=int, default=10,
                        help="Server intent detection timeout in seconds")

    parser = add_connection_argparse_parameters(parser)
    parser = add_asr_config_argparse_parameters(parser, profanity_filter=True)

    args = parser.parse_args()

    if args.list_devices:
        print("\n=== Available Audio Input Devices ===")
        print(sd.query_devices())
        return

    logger.info("=== Starting System with Wake Word Detection ===")
    logger.info(f"Wake phrases: {args.wake_phrases}")
    logger.info(f"Wake threshold: {args.wake_threshold}")
    logger.info(f"Mode: One-shot activation (auto-deactivate after command)")

    image_exporter = ImageExporter(
        export_dir=args.export_dir,
        enabled=args.debug_export_images
    )
    if args.debug_export_images:
        logger.info(f"[Main] Debug image export enabled: {image_exporter.export_dir}")

    cam_type = args.camera_type
    if cam_type == 'auto':
        cam_type = 'picamera' if PICAM2_AVAILABLE else 'opencv'
    
    logger.info(f"Selected Camera: {cam_type}")
    if cam_type == 'picamera':
        camera_mgr = PiCameraManager(CAMERA_CONFIG)
    else:
        camera_mgr = OpenCVCameraManager(CAMERA_CONFIG)

    if not camera_mgr.initialize():
        logger.warning("Camera init failed. System will run in Audio-Only mode.")
        camera_mgr = None

    # Initialize wake word detector
    wake_word_detector = WakeWordDetector(
        wake_phrases=args.wake_phrases,
        similarity_threshold=args.wake_threshold,
        logger_instance=logger
    )

    # Initialize classifiers
    server_classifier = ServerIntentClassifier(
        server_url=args.server_url,
        timeout=args.intent_timeout,
        logger_instance=logger
    )
    fast_classifier = FastIntentClassifier()
    logger.info(f"Intent Classification: Server-side (primary) + Fast (fallback)")

    # Initialize server client
    server_client = ServerClient(url=args.server_url, timeout=30)
    logger.info(f"Server Client initialized: {args.server_url}")

    # Initialize Piper TTS client
    tts_client = PiperTTSClient(server_url=args.piper_url)
    logger.info(f"Piper TTS Client initialized at {args.piper_url}")

    tts_thread = TTSWorker(tts_client)
    tts_thread.start()

    logic_thread = LogicProcessor(
        wake_word_detector,
        server_classifier, 
        fast_classifier, 
        server_client, 
        camera_mgr, 
        image_exporter, 
        consecutive_unknown_threshold=2
    )
    logic_thread.start()

    try:
        auth = riva.client.Auth(uri=args.server, use_ssl=args.use_ssl)
        asr_thread = AsyncASREngine(auth, args)
        asr_thread.start()
    except Exception as e:
        logger.critical(f"Riva Connection Failed: {e}")
        if camera_mgr: camera_mgr.cleanup()
        sys.exit(1)

    TTS_QUEUE.put(f"Hệ thống đã sẵn sàng. Hãy nói '{args.wake_phrases[0]}' để kích hoạt.")
    logger.info(f"=== System Running. Say '{args.wake_phrases[0]}' to activate. Press Ctrl+C to exit. ===")

    stats_print_interval = 60
    last_stats_print = time.time()
    
    try:
        while True:
            time.sleep(1)
            
            current_time = time.time()
            if current_time - last_stats_print >= stats_print_interval:
                logic_thread.print_stats()
                last_stats_print = current_time
                
    except KeyboardInterrupt:
        logger.info("Shutdown requested...")
        logger.info("=== Final Statistics ===")
        logic_thread.print_stats()
        
        state.stop()
        time.sleep(1.0)
        
        if camera_mgr:
            camera_mgr.cleanup()
        
        # Clean up TTS client
        tts_client.close()
            
        logger.info("Goodbye.")
        sys.exit(0)

if __name__ == "__main__":
    main()