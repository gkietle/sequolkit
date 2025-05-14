import os
import json
import requests
import base64
import time
import logging
import shutil
import signal
import sys
from typing import List, Dict, Tuple

# Global flag for graceful shutdown
terminate_requested = False

# Define directory structure first so we can set up logging correctly
base_dir = "./schema"
enriched_schema_dir = os.path.join(base_dir, "enriched_schema")
logs_dir = os.path.join(base_dir, "logs")

# Set up signal handler for graceful termination
def signal_handler(sig, frame):
    global terminate_requested
    if not terminate_requested:
        print("\nTermination requested. Finishing current task and exiting...\n")
        terminate_requested = True
    else:
        print("\nForced exit. Some data may be lost.\n")
        sys.exit(1)

# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)

# Remove and recreate directories at startup - DISABLED FOR RESUMING
def clean_directories():
    """Remove and recreate the directory structure."""
    # Check if base directory exists
    if os.path.exists(base_dir):
        # Remove only subdirectories, keep the main log file
        if os.path.exists(enriched_schema_dir):
            shutil.rmtree(enriched_schema_dir)
        if os.path.exists(logs_dir):
            shutil.rmtree(logs_dir)
    else:
        # Create base directory if it doesn't exist
        os.makedirs(base_dir, exist_ok=True)
    
    # Create subdirectories
    os.makedirs(enriched_schema_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

# Skip cleaning directories for resume
# clean_directories()

# Ensure directories exist (without cleaning)
os.makedirs(base_dir, exist_ok=True)
os.makedirs(enriched_schema_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(base_dir, "schema_enrichment.log"), encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def find_sqlite_files(base_dir: str) -> List[Dict[str, str]]:
    """
    Find all SQLite files in the subdirectories of the base directory.
    """
    sqlite_files = []
    
    logger.info(f"Scanning directory {base_dir} for SQLite files...")
    
    # Loop through all subdirectories
    for folder_name in os.listdir(base_dir):
        # Check for termination request
        if terminate_requested:
            logger.info("Termination requested during scan. Stopping scan process.")
            break
            
        folder_path = os.path.join(base_dir, folder_name)
        
        # Only process directories
        if os.path.isdir(folder_path):
            # Find .sqlite file in the directory
            for file_name in os.listdir(folder_path):
                if file_name.endswith('.sqlite'):
                    file_path = os.path.join(folder_path, file_name)
                    file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
                    
                    sqlite_files.append({
                        'db_id': folder_name,
                        'file_path': file_path,
                        'file_size': f"{file_size:.2f} MB"
                    })
                    break  # Only get the first sqlite file in each directory
    
    # Don't sort by size as requested
    return sqlite_files

def encode_sqlite_file(file_path: str) -> str:
    """
    Read SQLite file and encode to Base64.
    """
    try:
        with open(file_path, "rb") as file:
            sqlite_binary = file.read()
            return base64.b64encode(sqlite_binary).decode('utf-8')
    except Exception as e:
        logger.error(f"Error encoding file {file_path}: {str(e)}")
        raise

def process_database(file_info: Dict[str, str], api_url: str, schema_dir: str, logs_dir: str, timeout: int = 3000, retries: int = 3):
    """
    Process a database and save the results.
    
    Args:
        file_info: Information about the SQLite file
        api_url: URL of the schema-enrichment API
        schema_dir: Directory to save enriched schema JSON files
        logs_dir: Directory to save log files
        timeout: Maximum wait time in seconds
        retries: Number of retry attempts on failure
        
    Returns:
        True if successful, False if failed
    """
    global terminate_requested
    
    db_id = file_info['db_id']
    file_path = file_info['file_path']
    file_size = file_info.get('file_size', 'unknown size')
    
    start_time = time.time()
    logger.info(f"Starting processing: {db_id} ({file_size})")
    
    # Check if already processed
    output_file = os.path.join(schema_dir, f"{db_id}.json")
    if os.path.exists(output_file):
        logger.info(f"Results already exist for {db_id}, skipping")
        return True, {}
    
    # Write processing start information
    log_file = os.path.join(logs_dir, f"{db_id}.log")
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"Processing started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Database: {db_id}\n")
        f.write(f"File path: {file_path}\n")
        f.write(f"File size: {file_size}\n")
        f.write("-" * 50 + "\n")
    
    for attempt in range(retries + 1):
        # Check for termination request
        if terminate_requested:
            logger.info(f"Termination requested during processing of {db_id}. Stopping.")
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"Processing interrupted by user at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            return False, {}
            
        try:
            if attempt > 0:
                logger.warning(f"Retry {attempt}/{retries} for {db_id}")
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(f"Retry {attempt}/{retries}\n")
                
            # Encode SQLite file
            sqlite_base64 = encode_sqlite_file(file_path)
            
            # Create payload
            payload = {
                "connection_payload": {
                    "file": sqlite_base64,
                    "dbType": "sqlite"
                }
            }
            
            # Calculate timeout based on file size
            adaptive_timeout = min(timeout, max(300, int(float(file_size.split()[0]) * 100)))
            
            # Log
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"Sending request with timeout {adaptive_timeout}s\n")
            
            # Send request to API with smaller timeout chunks to allow interruption
            logger.info(f"Sending API request with timeout {adaptive_timeout}s for {db_id}")
            
            try:
                # Set a shorter timeout for better interrupt handling
                chunk_timeout = 1800
                response = requests.post(api_url, json=payload, timeout=chunk_timeout)
            except requests.exceptions.Timeout:
                # If we hit the chunk timeout but termination was requested, exit gracefully
                if terminate_requested:
                    logger.info(f"Termination requested during API call for {db_id}. Stopping.")
                    with open(log_file, 'a', encoding='utf-8') as f:
                        f.write(f"Processing interrupted by user at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    return False, {}
                # Otherwise re-raise the timeout for normal handling
                raise
            
            # Check if request was successful
            if response.status_code == 200:
                result = response.json()
                
                # Save result to JSON file
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                
                elapsed_time = time.time() - start_time
                
                # Log result
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(f"Processing successful\n")
                    f.write(f"Time: {elapsed_time:.2f}s\n")
                    f.write(f"Results saved to: {output_file}\n")
                
                # Use plain text instead of emoji
                logger.info(f"SUCCESS: {db_id} (time: {elapsed_time:.2f}s)")
                return True, result
            else:
                error_msg = f"HTTP Error {response.status_code}: {response.text[:200]}..."
                logger.error(f"FAILED: {error_msg}")
                
                # Log error
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(f"Error: {error_msg}\n")
                
                if attempt == retries:
                    return False, {}
                    
        except requests.exceptions.Timeout:
            logger.error(f"TIMEOUT: Processing {db_id} (attempt {attempt+1}/{retries+1})")
            
            # Log error
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"Error: Timeout\n")
                
            if attempt == retries:
                return False, {}
                
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            logger.error(f"ERROR: Exception while processing {db_id}: {error_msg}")
            
            # Log error
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"Error: {error_msg}\n")
            
            if attempt == retries:
                return False, {}
        
        # Wait before retrying, but check for termination periodically
        if attempt < retries:
            # Increase wait time exponentially
            backoff_time = 2 ** attempt * 5
            logger.info(f"Waiting {backoff_time}s before retry...")
            
            # Split the wait time into smaller chunks to allow interruption
            chunk_size = 1  # 1 second chunks
            for _ in range(int(backoff_time / chunk_size)):
                if terminate_requested:
                    logger.info(f"Termination requested during backoff for {db_id}. Stopping.")
                    return False, {}
                time.sleep(chunk_size)
    
    # Should never reach here, but just in case
    return False, {}

def main():
    global terminate_requested
    
    # Configuration
    spider_database_path = "../../database"
    api_url = "http://localhost:9393/schema-enrichment"
    max_retries = 3
    base_timeout = 3000  # Base timeout in seconds
    
    # Find all SQLite files
    sqlite_files = find_sqlite_files(spider_database_path)
    logger.info(f"Found {len(sqlite_files)} SQLite databases")
    
    # Display useful information
    total_size_mb = sum(float(info['file_size'].split()[0]) for info in sqlite_files)
    logger.info(f"Total size: {total_size_mb:.2f} MB")
    
    # Statistics
    total = len(sqlite_files)
    success = 0
    failed = 0
    
    # Save list of databases to process
    with open(os.path.join(base_dir, "database_list.txt"), 'w', encoding='utf-8') as f:
        for i, file_info in enumerate(sqlite_files):
            f.write(f"{i+1}. {file_info['db_id']} ({file_info['file_size']})\n")
    
    # Find the index to resume from
    resume_index = 0
    # for i, file_info in enumerate(sqlite_files):
    #     if file_info['db_id'] == 'soccer_1':
    #         # Soccer_1 already finished processing successfully but we got an error when trying to extract stats
    #         # So we resume from the next database
    #         resume_index = i + 1
    #         logger.info(f"Resuming from index {resume_index}, after database '{file_info['db_id']}'")
    #         break
    
    # Process each database from the resume point
    for i, file_info in enumerate(sqlite_files):
        # Skip databases before the resume point
        if i < resume_index:
            logger.info(f"Skipping already processed database [{i+1}/{total}]: {file_info['db_id']}")
            
            # Check if this database was successfully processed
            output_file = os.path.join(enriched_schema_dir, f"{file_info['db_id']}.json")
            if os.path.exists(output_file):
                success += 1
            continue
        
        # Check for termination request
        if terminate_requested:
            logger.info("Termination requested. Stopping processing.")
            break
            
        db_id = file_info['db_id']
        file_size = file_info['file_size']
        
        logger.info(f"\n[{i+1}/{total}] Processing: {db_id} ({file_size})")
        
        # Process database
        status, result = process_database(
            file_info, 
            api_url, 
            enriched_schema_dir,
            logs_dir,
            timeout=base_timeout,
            retries=max_retries
        )
        detail_results = ""
        if status:
            success += 1
            detail_results = "success"
            try:
                # More robust handling of result structure
                if isinstance(result, dict):
                    if "data" in result and isinstance(result["data"], dict):
                        if "_workflow_logs" in result["data"] and isinstance(result["data"]["_workflow_logs"], dict):
                            logs = result["data"]["_workflow_logs"]
                            enriched_tables = logs.get("enriched_tables", 0)
                            enriched_columns = logs.get("enriched_columns", 0)
                            total_tables = logs.get("total_tables", 0)
                            total_columns = logs.get("total_columns", 0)
                            detail_results = f"result: {enriched_tables}/{total_tables} tables, {enriched_columns}/{total_columns} columns"
            except Exception as e:
                logger.warning(f"Could not extract detailed statistics from result: {str(e)}")
        else:
            failed += 1
        
        # Display current progress
        logger.info(f"Progress: {i+1}/{total} | Success: {success} | Failed: {failed} | Details: {detail_results}")
        
        # If termination was requested, stop processing more databases
        if terminate_requested:
            logger.info("Termination requested. Stopping processing.")
            break
            
        # Wait between requests to avoid overload
        if i < total - 1 and status:  # Only wait if successful
            # Use small chunks to allow interruption
            for _ in range(5):  # 0.2s * 5 = 1s
                if terminate_requested:
                    break
                time.sleep(0.2)
    
    # Create simple summary file
    with open(os.path.join(base_dir, "summary.txt"), 'w', encoding='utf-8') as f:
        completion_status = "Completed" if not terminate_requested else "Interrupted by user"
        f.write(f"Status: {completion_status}\n")
        f.write(f"Completion time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        processed = success + failed
        f.write(f"Processed: {processed}/{total}\n")
        f.write(f"Success: {success}\n")
        f.write(f"Failed: {failed}\n")
        if processed > 0:
            f.write(f"Success rate: {(success/processed)*100:.2f}%\n")
    
    # Print summary
    logger.info(f"\n===== FINAL RESULTS =====")
    logger.info(f"Status: {'Completed' if not terminate_requested else 'Interrupted by user'}")
    logger.info(f"Processed: {success + failed}/{total}")
    logger.info(f"Success: {success}")
    logger.info(f"Failed: {failed}")
    if success + failed > 0:
        logger.info(f"Success rate: {(success/(success + failed))*100:.2f}%")

if __name__ == "__main__":
    try:
        start_time = time.time()
        main()
        total_time = time.time() - start_time
        logger.info(f"Total processing time: {total_time/60:.2f} minutes")
    except Exception as e:
        logger.critical(f"Critical error: {str(e)}", exc_info=True)