import atexit
import logging
from langfuse import Langfuse
from langfuse.llama_index import LlamaIndexInstrumentor
from config.app_config import app_config

logger = logging.getLogger(__name__)

class ObservabilityService:
    def __init__(self):
        self.langfuse = None
        self.llama_index_instrumentor = None
        self._initialize()
    
    def _initialize(self):
        """Initialize Langfuse clients for observability"""
        try:
            # Initialize Langfuse client
            self.langfuse = Langfuse(
                public_key=app_config.LANGFUSE_PUBLIC_KEY,
                secret_key=app_config.LANGFUSE_SECRET_KEY,
                host=app_config.LANGFUSE_HOST
            )
            logger.info("Langfuse client initialized")
            
            # Initialize Langfuse LlamaIndex instrumentor
            self.llama_index_instrumentor = LlamaIndexInstrumentor(
                public_key=app_config.LANGFUSE_PUBLIC_KEY,
                secret_key=app_config.LANGFUSE_SECRET_KEY,
                host=app_config.LANGFUSE_HOST
            )
            self.llama_index_instrumentor.start()
            logger.info("Langfuse LlamaIndex instrumentor started")
            
            # Register shutdown handler
            atexit.register(self.flush_on_shutdown)
        except Exception as e:
            logger.error(f"Failed to initialize Langfuse: {str(e)}")
            # Set up dummy clients that don't do anything if initialization fails
            self._setup_dummy_clients()
    
    def _setup_dummy_clients(self):
        """Set up dummy clients if Langfuse initialization fails"""
        class DummyLangfuse:
            def trace(self, *args, **kwargs):
                return DummyTrace()
            
            def span(self, *args, **kwargs):
                return DummySpan()
            
            def flush(self):
                pass
        
        class DummyTrace:
            def __init__(self):
                self.id = "dummy-trace-id"
            
            def update(self, *args, **kwargs):
                pass
            
            def span(self, *args, **kwargs):
                return DummySpan()
        
        class DummySpan:
            def update(self, *args, **kwargs):
                pass
        
        class DummyInstrumentor:
            def observe(self, *args, **kwargs):
                class DummyContextManager:
                    def __enter__(self):
                        pass
                    
                    def __exit__(self, exc_type, exc_val, exc_tb):
                        pass
                
                return DummyContextManager()
            
            def flush(self):
                pass
        
        self.langfuse = DummyLangfuse()
        self.llama_index_instrumentor = DummyInstrumentor()
        logger.warning("Using dummy Langfuse clients due to initialization failure")
    
    def flush_on_shutdown(self):
        """Flush Langfuse events before shutdown"""
        logger.info("Flushing Langfuse events before shutdown")
        if self.langfuse:
            self.langfuse.flush()
        if self.llama_index_instrumentor:
            self.llama_index_instrumentor.flush()

# Create a singleton instance
observability_service = ObservabilityService() 