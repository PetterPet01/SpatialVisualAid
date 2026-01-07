from typing import List, Dict, Any, Optional, Union, Tuple, Literal, TypedDict
from PIL import Image
from openai import OpenAI
import requests
import json
import base64
import mimetypes
from pathlib import Path
from io import BytesIO
import random # <-- Import the random module

Message = Dict[str, Union[str, List[Dict[str, Union[str, Dict[str, str]]]]]]


class Generator:
    """
    An optimized, universal client for the Unified AI Gateway server.
    Supports both text and multimodal (text and image) generation, intelligently
    leveraging the server's backend routing for Mistral and Gemini models.
    """
    def __init__(self,
                 base_url: str = "http://localhost:8501",
                 api_key: str = "dummy", # The gateway manages keys, so this can be a placeholder
                 model_name: str = "mistral-medium-latest",
                 temperature: float = 0.7,
                 max_new_tokens: int = 4096,
                 timeout: int = 120):
        """
        Initialize the Generator for the Unified AI Gateway.

        Args:
            base_url: URL of your AI Gateway server.
            api_key: API key (can be a dummy value for the gateway).
            model_name: Default model name to use for requests.
            temperature: Default temperature for generation.
            max_new_tokens: Default maximum number of new tokens to generate.
            timeout: Request timeout in seconds (increased for vision models).
        """
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.timeout = timeout

        # Initialize OpenAI client pointed at your server's endpoint
        self.client = OpenAI(
            base_url=f"{self.base_url}/v1",
            api_key=api_key,
            timeout=timeout
        )

        print(f"✅ Generator Initialized: Connected to Unified AI Gateway at {self.base_url}")
        self._test_connection()

    def _test_connection(self) -> None:
        """Tests the connection to vLLM server."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            if response.status_code == 200:
                print(f"✅ Connected to vLLM at {self.base_url}")
                print(f"🎯 Model: {self.model_name}")
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Could not connect to vLLM: {e}")
        def get_available_models(self) -> List[str]:
            """Gets the list of available models from the server."""
            try:
                response = self.client.models.list()
                return [model.id for model in response.data]
            except Exception as e:
                print(f"⚠️ Could not fetch models from the gateway: {e}")
                return []

    def _encode_image_from_path(self, image_path: str) -> str:
        """Loads an image file, encodes it to Base64, and formats it as a data URI."""
        try:
            path = Path(image_path)
            if not path.exists():
                raise FileNotFoundError(f"Image file not found at {image_path}")

            mime_type, _ = mimetypes.guess_type(path)
            if not mime_type or not mime_type.startswith('image'):
                ext_map = {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png', '.gif': 'image/gif', '.webp': 'image/webp'}
                mime_type = ext_map.get(path.suffix.lower())
                if not mime_type:
                    raise ValueError(f"Unsupported or unknown image format for file: {path.name}")

            with open(path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

            return f"data:{mime_type};base64,{encoded_string}"
        except Exception as e:
            raise IOError(f"Error processing image file {image_path}: {e}") from e

    def _encode_image_from_pil(self, image: Image.Image) -> str:
        """Encodes a PIL.Image object to a Base64 data URI."""
        try:
            buffered = BytesIO()
            # Save as PNG to preserve quality; it's a safe, widely supported format.
            image.save(buffered, format="PNG")
            base64_bytes = base64.b64encode(buffered.getvalue())
            return f"data:image/png;base64,{base64_bytes.decode('utf-8')}"
        except Exception as e:
            raise IOError(f"Error processing PIL image object: {e}") from e

    def _prepare_api_messages(self,
                            prompt: Optional[str],
                            messages: Optional[List[Message]],
                            images: Optional[List[Union[str, Image.Image]]]) -> Tuple[List[Message], List[Message], bool]:
        """Prepares messages for the API server, handling text and various image input types."""
        if not prompt and not messages:
            raise ValueError("Either 'prompt' or 'messages' must be provided.")

        full_history = list(messages) if messages else []

        if not full_history:
            full_history.append({"role": "user", "content": prompt or ""})

        if not images:
            return full_history, full_history, False

        # Images can only be attached to the last user message
        last_message = full_history[-1]
        if last_message["role"] != "user":
            raise ValueError("Images can only be added to the most recent 'user' message in the history.")

        # Ensure content is a string before modification
        text_content = last_message.get("content", "")
        if not isinstance(text_content, str):
            raise TypeError("The last message content must be a string when providing new images.")

        has_images = False
        final_user_content = [{"type": "text", "text": text_content}]

        for i, img_input in enumerate(images):
            try:
                if isinstance(img_input, str):
                    base64_image_uri = self._encode_image_from_path(img_input)
                elif isinstance(img_input, Image.Image):
                    base64_image_uri = self._encode_image_from_pil(img_input)
                else:
                    raise TypeError(f"Image input must be a file path (str) or a PIL.Image.Image object, but got {type(img_input)}")

                final_user_content.append({
                    "type": "image_url",
                    "image_url": {"url": base64_image_uri}
                })
                has_images = True
            except (IOError, FileNotFoundError, ValueError, TypeError) as e:
                error_msg = f"[Error processing image {i+1}: {e}]"
                # Prepend the error to the text content for the LLM to see
                final_user_content[0]["text"] = f"{error_msg}\n{final_user_content[0]['text']}"
                print(f"⚠️  {error_msg}")

        # Replace the last message with the new multimodal content
        api_messages = full_history[:-1]
        api_messages.append({"role": "user", "content": final_user_content})

        return api_messages, full_history, has_images

    def _handle_api_error(self, error: Exception) -> str:
        """Provides user-friendly interpretations of potential API errors."""
        error_str = str(error)
        if "Connection refused" in error_str:
            return f"Connection Failed. Is the AI Gateway server running at {self.base_url}?"
        if "All Gemini API keys have been exhausted" in error_str:
            return "Server reported that all Gemini keys are exhausted or have failed. Please check the server logs."
        if "Mistral request failed after" in error_str:
            return "Server reported that all retries for Mistral failed. The Mistral API might be down or all keys are invalid."
        return f"An unhandled error occurred: {error_str}"

    def generate(self,
                 prompt: Optional[str] = None,
                 messages: Optional[List[Message]] = None,
                 images: Optional[List[Union[str, Image.Image]]] = None,
                 model_name: Optional[str] = None,
                 temperature: Optional[float] = None,
                 max_new_tokens: Optional[int] = None,
                 **kwargs) -> Tuple[str, List[Message]]:
        """
        Generates a response using the Unified AI Gateway.

        The gateway server automatically:
        - Routes requests to Mistral or Gemini based on the model name.
        - Manages API key rotation, rate limiting, and load balancing.
        - Provides real-time metrics and monitoring.

        Args:
            prompt: A single string prompt. Used if 'messages' is not provided.
            messages: A list of message dictionaries (OpenAI format).
            images: A list of image inputs. Items can be file paths (str) or PIL.Image.Image objects.
            model_name: The model to use (e.g., "gpt-4o", "mistral-medium-latest").
            temperature: The temperature for this request.
            max_new_tokens: The max tokens for this request.
            **kwargs: Additional parameters to pass to the API.

        Returns:
            A tuple containing:
            - The generated text response (str).
            - The complete conversation history including the new response (list).
        """
        try:
            api_messages, conversation_history, has_images = self._prepare_api_messages(
                prompt, messages, images
            )

            selected_model = model_name if model_name is not None else self.model_name

            if has_images:
                num_images = len([part for part in api_messages[-1]['content'] if part.get('type') == 'image_url'])
                print(f"📷 Sending multimodal request to model '{selected_model}' with {num_images} image(s)")

            api_params = {
                "model": selected_model,
                "messages": api_messages,
                "temperature": temperature if temperature is not None else self.temperature,
                "max_tokens": max_new_tokens if max_new_tokens is not None else self.max_new_tokens,
                **kwargs
            }

            response = self.client.chat.completions.create(**api_params)
            result = response.choices[0].message.content or ""

            if response.model and response.model != selected_model:
                print(f"🔄 Server auto-routed request to model: {response.model} (requested: {selected_model})")

        except Exception as e:
            error_message = self._handle_api_error(e)
            print(f"❌ Generation failed: {error_message}")
            raise RuntimeError(error_message) from e

        conversation_history.append({"role": "assistant", "content": result})
        return result, conversation_history

    def ocr(self,
            image: Union[str, Image.Image],
            model_name: str = "mistral-ocr-latest",
            include_image_base64: bool = False,
            **kwargs) -> Dict[str, Any]:
        """
        Performs Optical Character Recognition (OCR) on an image using Mistral's OCR models.

        This method sends a request to the dedicated `/v1/ocr` endpoint on the Unified AI Gateway.

        Args:
            image: The image to process. Can be a file path (str) or a PIL.Image.Image object.
            model_name: The OCR model to use (e.g., "mistral-ocr-latest").
            include_image_base64: Whether to include the base64-encoded image in the response.
            **kwargs: Additional parameters to pass to the API.

        Returns:
            A dictionary containing the structured OCR response from the server.

        Raises:
            RuntimeError: If the OCR request fails after all retries on the server side
                          or if there is a connection issue.
        """
        print(f"📄 Sending dedicated OCR request for model '{model_name}'...")
        try:
            if isinstance(image, str):
                base64_image_uri = self._encode_image_from_path(image)
            elif isinstance(image, Image.Image):
                base64_image_uri = self._encode_image_from_pil(image)
            else:
                raise TypeError(f"Image input must be a file path (str) or a PIL.Image.Image object, but got {type(image)}")

            ocr_url = f"{self.base_url}/v1/ocr"
            
            payload = {
                "model": model_name,
                "image": base64_image_uri,
                "include_image_base64": include_image_base64,
                **kwargs
            }

            response = requests.post(ocr_url, json=payload, timeout=self.timeout)
            response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)

            result = response.json()
            # The server might return 200 OK but with an error message in the JSON
            if "error" in result and result.get("status") == "error":
                 error_message = result.get("error_message", "Unknown OCR error from server")
                 raise Exception(error_message)

            return result

        except requests.exceptions.HTTPError as http_err:
            try:
                # Try to parse the JSON error from the response body
                error_details = http_err.response.json()
                error_message = error_details.get("error", {}).get("message", http_err.response.text)
            except json.JSONDecodeError:
                error_message = str(http_err)
            
            print(f"❌ OCR failed: {error_message}")
            raise RuntimeError(f"OCR request failed: {error_message}") from http_err
        
        except Exception as e:
            error_message = self._handle_api_error(e)
            print(f"❌ OCR failed: {error_message}")
            raise RuntimeError(error_message) from e

    def ocr_with_gemma(self,
                       image: Union[str, Image.Image],
                       model_name: Union[str, List[str]] = 'gemma-3-27b-it',
                       **kwargs) -> str:
        """
        Performs a two-step, refined OCR on an image using a general-purpose vision
        model followed by a text model for cleanup.

        Step 1: Uses a powerful vision model to extract all text from the image.
        Step 2: Feeds the raw text into `mistral-small-latest` to clean it up.

        This method provides a highly robust OCR result, suitable for challenging images.

        Args:
            image: The image to process. Can be a file path (str) or a PIL.Image.Image object.
            model_name: The vision model to use for the initial extraction. Can be a
                        single model name (str) or a list of model names (List[str]).
                        If a list is provided, one will be chosen at random for each call.
                        (e.g., ["gemma-2-27b-it", "gemma-2-9b-it"])
            **kwargs: Additional parameters to pass to the initial `generate` call
                      (e.g., temperature).

        Returns:
            The cleaned and refined text as a single string.

        Raises:
            RuntimeError: If the initial OCR (Step 1) fails. If the refinement (Step 2)
                          fails, a warning is logged and the raw text is returned.
            ValueError: If an empty list is provided for `model_name`.
        """
        # --- Select the model for Step 1 ---
        selected_model: str
        if isinstance(model_name, list):
            if not model_name:
                raise ValueError("The 'model_name' list cannot be empty.")
            selected_model = random.choice(model_name)
            print(f"🔩 Randomly selected '{selected_model}' from the provided model list for Step 1.")
        elif isinstance(model_name, str):
            selected_model = model_name
        else:
            raise TypeError(f"The 'model_name' parameter must be a string or a list of strings, but got {type(model_name)}.")

        # --- Step 1: Initial OCR with the selected vision model ---
        initial_ocr_prompt = "Extract all the text in this image and output the OCR result"
        print(f"📄 Step 1/2: Performing initial OCR with '{selected_model}'...")

        try:
            raw_ocr_text, _ = self.generate(
                prompt=initial_ocr_prompt,
                images=[image],
                model_name=selected_model,
                **kwargs
            )
            print(f"📝 Raw OCR result received from {selected_model}.")
        except RuntimeError as e:
            print(f"❌ Initial OCR (Step 1) with {selected_model} failed.")
            raise e

        # --- Step 2: Refine the raw OCR output with Mistral as a safeguard ---
        refinement_model = "mistral-small-latest"
        refinement_prompt = f"""
The following text is a raw output from an OCR process. It might contain noise, errors, or conversational filler like "Here is the text...".

Your task is to clean this text.
1. Remove any conversational introductions or summaries.
2. Correct obvious OCR errors if possible, but prioritize removing gibberish.
3. Preserve the original structure and line breaks of the core text.
4. Output *only* the cleaned text. Do not add any extra comments or explanations.

Raw OCR Text:
---
{raw_ocr_text}
---
Cleaned Text:
"""
        print(f"🧹 Step 2/2: Refining OCR output with '{refinement_model}'...")

        try:
            refined_text, _ = self.generate(
                prompt=refinement_prompt,
                model_name=refinement_model,
                temperature=0.1
            )
            print("✅ Refinement complete.")
            return refined_text.strip()
        except RuntimeError as e:
            print(f"⚠️ Warning: Refinement (Step 2) with {refinement_model} failed: {e}")
            print("Returning the raw, unrefined OCR text as a fallback.")
            return raw_ocr_text.strip()

    def get_server_metrics(self) -> Optional[Dict]:
        """Gets real-time metrics from the gateway's dashboard."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            response.raise_for_status()
            return response.json().get('metrics', {})
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Could not fetch server metrics: {e}")
        return None

    def print_server_status(self) -> None:
        """Prints a comprehensive status report from the AI Gateway."""
        metrics = self.get_server_metrics()
        if not metrics:
            print("\n❌ Could not retrieve server status. Is the server online?")
            return

        print("\n" + "="*50)
        print("🚀 UNIFIED AI GATEWAY STATUS")
        print("="*50)
        print(f"📊 Total Requests: {metrics.get('total_requests', 0):,}")
        print(f"✅ Success Rate:   {metrics.get('success_rate', 0):.1f}%")
        print(f"⚡ Active Requests:  {metrics.get('active_count', 0)}")
        print(f"🔑 Active Keys - Mistral: {metrics.get('mistral_keys_available', 'N/A')} | Gemini: {metrics.get('gemini_keys_available', 'N/A')}")
        print(f"⏱️ Avg Response:   {metrics.get('avg_response_time', 0)*1000:.0f} ms")
        print(f"📈 Throughput:     {metrics.get('requests_per_minute', 0):.1f} req/min")
        print(f"⏰ Uptime:         {metrics.get('uptime_formatted', 'Unknown')}")
        print("="*50)