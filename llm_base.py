from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseLLM(ABC):
    @abstractmethod
    async def generate_content(self, prompt: str, generation_config: Dict[str, Any] = None) -> Any:
        """Generate content from the LLM"""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the LLM implementation"""
        pass
