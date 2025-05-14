import json
import logging
import asyncio
from typing import Dict, Any, Optional
import google.generativeai as genai
from litellm import completion
from llm_base import BaseLLM

logger = logging.getLogger("web_automation")

class GeminiLLM(BaseLLM):
    def __init__(self, model):
        self.model = model

    async def generate_content(self, prompt: str, generation_config: Dict[str, Any] = None) -> Any:
        response = await asyncio.to_thread(
            self.model.generate_content,
            prompt,
            generation_config=generation_config
        )
        return response

    @property
    def name(self) -> str:
        return "gemini"

class OllamaLLM(BaseLLM):
    def __init__(self, model_name: str = "qwen3:32b"):
        self.model_name = model_name

    async def generate_content(self, prompt: str, generation_config: Dict[str, Any] = None) -> Any:
        try:
            response = await completion(
                model=f"ollama/{self.model_name}",
                messages=[{"role": "user", "content": prompt}],
                temperature=generation_config.get("temperature", 0.7) if generation_config else 0.7
            )
            
            # Create a response object that mimics Gemini's interface
            class Response:
                def __init__(self, text):
                    self.text = text

            return Response(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            raise

    @property
    def name(self) -> str:
        return f"ollama/{self.model_name}"
