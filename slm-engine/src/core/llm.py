import os
import logging
from typing import Dict, Any, Optional, Tuple
from llama_index.llms.ollama import Ollama

logger = logging.getLogger(__name__)

class LLMConfig:
    """Centralized LLM configuration and management class."""
    
    def __init__(self):
        self.settings = {
            "ollama_host": os.getenv("OLLAMA_HOST", "http://localhost:9292/"),
            "ollama_model": os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
            "additional_kwargs": {
                "num_predict": 8192,
                "temperature": 0.7,
            },
            "prompt_routing": 0,
            "enrich_schema": True
        }
        self.llm = None
        self._initialize_llm()

    def _initialize_llm(self) -> None:
        """Initialize the LLM client with current settings."""
        logger.info("Initializing LLM client with settings:")
        logger.info(f"Host: {self.settings['ollama_host']}")
        logger.info(f"Model: {self.settings['ollama_model']}")
        logger.info(f"Additional kwargs: {self.settings['additional_kwargs']}")
        logger.info(f"Prompt routing: {self.settings['prompt_routing']}")
        logger.info(f"Enrich schema: {self.settings['enrich_schema']}")
        
        self.llm = Ollama(
            model=self.settings["ollama_model"],
            base_url=self.settings["ollama_host"],
            request_timeout=300.0,
            keep_alive=30*60,
            additional_kwargs=self.settings["additional_kwargs"]
        )
        logger.info("LLM client initialized successfully")

    def update_settings(
        self,
        host: Optional[str] = None,
        model: Optional[str] = None,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        prompt_routing: Optional[int] = None,
        enrich_schema: Optional[bool] = None
    ) -> None:
        """Update LLM settings and reinitialize the client."""
        if host is not None:
            self.settings["ollama_host"] = host
        if model is not None:
            self.settings["ollama_model"] = model
        if additional_kwargs is not None:
            self.settings["additional_kwargs"] = additional_kwargs
        if prompt_routing is not None:
            self.settings["prompt_routing"] = int(prompt_routing)
        if enrich_schema is not None:
            if isinstance(enrich_schema, str):
                self.settings["enrich_schema"] = enrich_schema.lower() in ["true", "1", "yes", "y"]
            else:
                self.settings["enrich_schema"] = bool(enrich_schema)

        self._initialize_llm()

    def get_settings(self) -> Dict[str, Any]:
        """Get current LLM settings."""
        return self.settings.copy()

    def get_llm(self) -> Ollama:
        """Get the current LLM instance."""
        return self.llm

# Create a singleton instance
llm_config = LLMConfig()
