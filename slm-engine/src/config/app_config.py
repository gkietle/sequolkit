import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from core.llm import llm_config, LLMFactory
from core.templates import text2sql_prompt_routing
from typing import Tuple

# Configure logging with a more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
def load_environment():
    env_path = Path('.env')
    load_dotenv(dotenv_path=env_path)
    logger.info(f"Looking for .env file at: {env_path.absolute()}")
    logger.info(f"Environment: {os.getenv('ENV')}")

# Application settings
class AppConfig:
    def __init__(self):
        # Load environment variables
        load_environment()
        
        # Service configuration
        self.SERVICE_PORT = int(os.getenv("SERVICE_PORT", 5000))
        
        # Default values for settings
        self.PROMPT_ROUTING = int(os.getenv("PROMPT_ROUTING", 1))
        self.ENRICH_SCHEMA = True
        self.PRIVACY_MODE = os.getenv("PRIVACY_MODE", "False").lower() in ["true", "1", "yes", "y"]
        
        # Langfuse configuration
        self.LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
        self.LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
        self.LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        
        # Log all configuration values
        logger.info("Configuration loaded:")
        logger.info(f"PROMPT_ROUTING: {self.PROMPT_ROUTING}")
        logger.info(f"ENRICH_SCHEMA: {self.ENRICH_SCHEMA}")
        logger.info(f"PRIVACY_MODE: {self.PRIVACY_MODE}")
    
    def print_banner(self, banner_file='banner.txt'):
        """Print a banner from a file when the application starts if it exists"""
        try:
            # Check if the banner file exists
            if os.path.exists(banner_file):
                # Read the banner from the file
                with open(banner_file, 'r') as f:
                    banner = f.read()
                logger.info("\n" + banner)
        except Exception:
            # Silently ignore any errors reading the banner file
            pass
            
        # Always log server started message
        logger.info("Server started successfully!")

# Create a singleton instance
app_config = AppConfig()

# Initialize workflows
def initialize_workflows():
    TEXT_TO_SQL_PROMPT_SKELETON = text2sql_prompt_routing(llm_config.settings["prompt_routing"])
    
    # Import here to avoid circular imports
    from core.workflow import SQLAgentWorkflow, SchemaEnrichmentWorkflow, BaselineWorkflow, QuestionSuggestionWorkflow
    
    workflow = SQLAgentWorkflow(
        text2sql_prompt=TEXT_TO_SQL_PROMPT_SKELETON,
        llm=llm_config.get_llm(),
        verbose=True
    )
    logger.info("SQL Agent Workflow initialized successfully")

    schema_workflow = SchemaEnrichmentWorkflow(
        llm=llm_config.get_llm(),
        verbose=True
    )
    logger.info("Schema Enrichment Workflow initialized successfully")

    baseline_workflow = BaselineWorkflow(
        llm=llm_config.get_llm(),
        text2sql_prompt=TEXT_TO_SQL_PROMPT_SKELETON,
        verbose=True
    )
    logger.info("Baseline Workflow initialized successfully")
    
    question_workflow = QuestionSuggestionWorkflow(
        llm=llm_config.get_llm(),
        verbose=True
    )
    logger.info("Question Suggestion Workflow initialized successfully")
    
    return workflow, schema_workflow, baseline_workflow, question_workflow

def preload_llm() -> Tuple[bool, str]:
    try:
        logger.info("Pre-loading LLM...")
        is_healthy, response = llm_config.get_health_check()
        if is_healthy:
            logger.info("LLM pre-loaded successfully")
            return True, response
        else:
            logger.error(f"Failed to pre-load LLM: {response}")
            return False, response
    except Exception as e:
        logger.error(f"Error pre-loading LLM: {str(e)}")
        return False, str(e)
