import requests
import json
import base64
import io
from urllib.parse import urlsplit
import numpy as np
from PIL import Image

class OpenAILLMNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "User prompt..."
                }),
                "endpoint": ("STRING", {
                    "multiline": False,
                    "default": "https://api.openai.com/v1/chat/completions",
                    "placeholder": "OpenAI-compatible endpoint URL"
                }),
                "api_token": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "placeholder": "Your API token"
                }),
            },
            "optional": {
                "image": ("IMAGE",),
                "model": ("STRING", {
                    "multiline": False,
                    "default": "gpt-4-vision-preview",
                    "placeholder": "Model name"
                }),
                "max_tokens": ("INT", {
                    "default": 150,
                    "min": 1,
                    "max": 4096,
                    "step": 1
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1
                }),
                "image_detail": (["low", "high", "auto"], {
                    "default": "auto"
                }),
                # Added last so existing saved workflows keep their widget order
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "System prompt, e.g. You are a helpful assistant. (leave empty to send none)"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_text"
    CATEGORY = "LLM"
    
    def _encode_image_to_base64(self, image_array):
        """Convert one ComfyUI image ([height, width, channels], values 0-1) to a PNG data URL"""
        try:
            # ComfyUI passes torch tensors; convert to a numpy array first
            if hasattr(image_array, "detach"):
                image_array = image_array.detach().cpu().float().numpy()
            else:
                image_array = np.asarray(image_array)
            
            # Float images are 0-1 (and may overshoot slightly); integer images are 0-255
            if np.issubdtype(image_array.dtype, np.floating):
                image_array = np.clip(image_array, 0.0, 1.0) * 255.0
            image_array = np.clip(image_array, 0, 255).astype(np.uint8)
            
            # Drop a trailing single channel so PIL treats it as grayscale
            if image_array.ndim == 3 and image_array.shape[2] == 1:
                image_array = image_array[:, :, 0]
            
            # PIL infers L / RGB / RGBA from the array shape
            pil_image = Image.fromarray(image_array)
            
            buffered = io.BytesIO()
            pil_image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            return f"data:image/png;base64,{img_base64}"
            
        except Exception as e:
            raise RuntimeError(f"Failed to encode image: {e}") from e
    
    @staticmethod
    def _iter_images(image):
        """Yield each image of a ComfyUI IMAGE input ([batch, H, W, C] or a single [H, W, C])"""
        if len(image.shape) == 4:
            for i in range(image.shape[0]):
                yield image[i]
        else:
            yield image
    
    @staticmethod
    def _api_error_detail(response):
        """Best-effort extraction of the provider's error message from a failed response"""
        try:
            body = response.json()
        except ValueError:
            return response.text[:500] or response.reason
        err = body.get("error", body) if isinstance(body, dict) else body
        if isinstance(err, dict):
            detail = err.get("message") or json.dumps(err)
            code = err.get("code") or err.get("type")
            return f"[{code}] {detail}" if code else detail
        return str(err)
    
    def generate_text(self, prompt, endpoint, api_token, model="gpt-4-vision-preview", max_tokens=150, temperature=0.7, image=None, image_detail="auto", system_prompt=""):
        # Errors are raised (not returned as text) so ComfyUI stops the run and
        # shows them, instead of passing an error message downstream as a prompt.
        headers = {
            "Authorization": f"Bearer {api_token.strip()}",
            "Content-Type": "application/json"
        }
        
        # Construct message content
        if image is not None:
            # Text plus every image in the batch
            message_content = [{"type": "text", "text": prompt}]
            for img in self._iter_images(image):
                message_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": self._encode_image_to_base64(img),
                        "detail": image_detail
                    }
                })
        else:
            # Text-only message
            message_content = prompt
        
        messages = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": message_content})
        
        data = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        # Report the host only: some providers put the API key in the URL query string
        host = urlsplit(endpoint).netloc or "endpoint"
        
        try:
            response = requests.post(endpoint, headers=headers, json=data, timeout=30)
        except requests.exceptions.RequestException as e:
            msg = str(e)
            query = urlsplit(endpoint).query
            if query:
                msg = msg.replace(query, "<redacted>")
            raise RuntimeError(f"Request to {host} failed: {type(e).__name__}: {msg}") from None
        
        if not response.ok:
            raise RuntimeError(f"HTTP {response.status_code} from {host}: {self._api_error_detail(response)}")
        
        try:
            result = response.json()
        except ValueError:
            raise RuntimeError(f"Invalid JSON from {host}: {response.text[:500]}") from None
        
        choices = result.get("choices") if isinstance(result, dict) else None
        if not choices:
            raise RuntimeError(f"No choices in response from {host}: {json.dumps(result)[:500]}")
        
        choice = choices[0]
        content = (choice.get("message") or {}).get("content")
        if not content:
            reason = choice.get("finish_reason")
            hint = " (try raising max_tokens)" if reason == "length" else ""
            raise RuntimeError(f"Empty response content from {host} (finish_reason: {reason}){hint}")
        return (content,)
