import os
import logging
from typing import Dict, Any, Optional, Union
from abc import ABC, abstractmethod
from llama_index.llms.ollama import Ollama
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.llms import LLM

logger = logging.getLogger(__name__)

class BaseLLMConfig(ABC):
    """Abstract base class for LLM configuration."""
    
    def __init__(self):
        self.settings = self._get_default_settings()
        self.llm = None
        self._initialize_llm()

    @abstractmethod
    def _get_default_settings(self) -> Dict[str, Any]:
        """Get default settings for the specific LLM provider."""
        pass

    @abstractmethod
    def _initialize_llm(self) -> None:
        """Initialize the LLM client with current settings."""
        pass

    @abstractmethod
    def update_settings(self, **kwargs) -> None:
        """Update LLM settings and reinitialize the client."""
        pass

    def get_settings(self) -> Dict[str, Any]:
        """Get current LLM settings."""
        return self.settings.copy()

    def get_llm(self) -> LLM:
        """Get the current LLM instance."""
        return self.llm

class OllamaConfig(BaseLLMConfig):
    """Ollama-specific LLM configuration."""
    
    def _get_default_settings(self) -> Dict[str, Any]:
        return {
            "ollama_host": os.getenv("OLLAMA_HOST", "http://localhost:9292/"),
            "ollama_model": os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
            "additional_kwargs": {
                "num_predict": 8192,
                "temperature": 0.7,
            },
            "prompt_routing": 0,
            "enrich_schema": True
        }

    def _initialize_llm(self) -> None:
        """Initialize the Ollama LLM client with current settings."""
        logger.info("Initializing Ollama LLM client with settings:")
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
        logger.info("Ollama LLM client initialized successfully")

    def update_settings(
        self,
        host: Optional[str] = None,
        model: Optional[str] = None,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        prompt_routing: Optional[int] = None,
        enrich_schema: Optional[bool] = None
    ) -> None:
        """Update Ollama LLM settings and reinitialize the client."""
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

class GoogleGenAIConfig(BaseLLMConfig):
    """Google Gemini-specific LLM configuration."""
    
    def _get_default_settings(self) -> Dict[str, Any]:
        return {
            "model": os.getenv("GOOGLE_MODEL", "gemini-2.0-flash"),
            "api_key": os.getenv("GOOGLE_API_KEY", ""),
            "temperature": float(os.getenv("GOOGLE_TEMPERATURE", "0.7")),
            "max_tokens": int(os.getenv("GOOGLE_MAX_TOKENS", "8192")),
            "prompt_routing": 0,
            "enrich_schema": True
        }

    def _initialize_llm(self) -> None:
        """Initialize the Google Gemini LLM client with current settings."""
        logger.info("Initializing Google Gemini LLM client with settings:")
        logger.info(f"Model: {self.settings['model']}")
        logger.info(f"Temperature: {self.settings['temperature']}")
        logger.info(f"Max tokens: {self.settings['max_tokens']}")
        logger.info(f"Prompt routing: {self.settings['prompt_routing']}")
        logger.info(f"Enrich schema: {self.settings['enrich_schema']}")
        
        self.llm = GoogleGenAI(
            model=self.settings["model"],
            api_key=self.settings["api_key"],
            temperature=self.settings["temperature"],
            max_tokens=self.settings["max_tokens"]
        )
        logger.info("Google Gemini LLM client initialized successfully")

    def update_settings(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        prompt_routing: Optional[int] = None,
        enrich_schema: Optional[bool] = None
    ) -> None:
        """Update Google Gemini LLM settings and reinitialize the client."""
        if model is not None:
            self.settings["model"] = model
        if api_key is not None:
            self.settings["api_key"] = api_key
        if temperature is not None:
            self.settings["temperature"] = float(temperature)
        if max_tokens is not None:
            self.settings["max_tokens"] = int(max_tokens)
        if prompt_routing is not None:
            self.settings["prompt_routing"] = int(prompt_routing)
        if enrich_schema is not None:
            if isinstance(enrich_schema, str):
                self.settings["enrich_schema"] = enrich_schema.lower() in ["true", "1", "yes", "y"]
            else:
                self.settings["enrich_schema"] = bool(enrich_schema)

        self._initialize_llm()

class LLMFactory:
    """Factory class for creating LLM configurations."""
    
    @staticmethod
    def create_llm_config(provider: str = "ollama") -> BaseLLMConfig:
        """
        Create an LLM configuration instance for the specified provider.
        
        Args:
            provider (str): The LLM provider to use ("ollama" or "google")
            
        Returns:
            BaseLLMConfig: An instance of the appropriate LLM configuration class
        """
        provider = provider.lower()
        if provider == "ollama":
            return OllamaConfig()
        elif provider == "google":
            return GoogleGenAIConfig()
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

# Create a default instance using the factory
llm_config = LLMFactory.create_llm_config(os.getenv("LLM_PROVIDER", "ollama"))
