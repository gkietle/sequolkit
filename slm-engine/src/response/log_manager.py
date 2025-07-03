import logging
from datetime import datetime
import json

# Configure logging
logger = logging.getLogger(__name__)

def log_step_start(step_name, **kwargs):
    """Log the start of a workflow step with consistent formatting."""
    logger.info(f"\033[94m[{step_name}] Starting {step_name.lower()}\033[0m")
    for key, value in kwargs.items():
        if isinstance(value, str) and len(value) > 200:
            value = value[:200] + "..." # Truncate long strings
        logger.info(f"\033[94m[{step_name}] {key}: {value}\033[0m")
    return datetime.now() # Return start time for duration calculation

def log_step_end(step_name, start_time, **kwargs):
    """Log the end of a workflow step with timing information."""
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    logger.info(f"\033[94m[{step_name}] Step completed in {duration:.2f} seconds\033[0m")
    for key, value in kwargs.items():
        logger.info(f"\033[94m[{step_name}] {key}: {value}\033[0m")

def log_success(step_name, message):
    """Log a success message."""
    logger.info(f"\033[92m[{step_name}] {message}\033[0m")

def log_error(step_name, message):
    """Log an error message."""
    logger.error(f"\033[91m[{step_name}] {message}\033[0m")

def log_warning(step_name, message):
    """Log a warning message."""
    logger.warning(f"\033[94m[{step_name}] {message}\033[0m")

def log_llm_operation(step_name, operation_name, start_time, response=None):
    """Log an LLM operation with timing."""
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    logger.info(f"\033[94m[{step_name}] {operation_name} time: {duration:.2f} seconds\033[0m")
    if response:
        logger.info(f"\033[94m[{step_name}] {operation_name} response: {str(response)}\033[0m")

def log_summary(title, items):
    """Log a summary section with consistent formatting."""
    logger.info("\033[94m" + "="*50 + "\033[0m")
    logger.info(f"\033[94m[{title}]\033[0m")
    for key, value in items.items():
        logger.info(f"\033[94m[{title}] {key}: {value}\033[0m")
    logger.info("\033[94m" + "="*50 + "\033[0m")

def log_prompt(prompt_text, step_name):
    """Log the formatted prompt with detailed formatting.
    
    Args:
        prompt_text (str): The prompt text to log
        step_name (str): The name of the current workflow step
    """
    logger.info(f"\n\033[95m===== {step_name} PROMPT =====\033[0m")
    logger.info(f"{prompt_text}")
    logger.info(f"\033[95m===== END {step_name} PROMPT =====\033[0m\n")
