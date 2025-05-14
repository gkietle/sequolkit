import requests
import os
import json
import base64
import argparse
import sys
import time
import tqdm
import datetime
import subprocess
import glob
import shutil
import pickle
import traceback

def set_model_settings(api_url_base, model_name, enrich_schema, prompt_routing):
    """
    Set the model settings via API before running inference
    
    Args:
        api_url_base: Base URL of the API (without the endpoint)
        model_name: Name of the model to use (e.g., "qwen2.5-coder:14b", "phi4")
        enrich_schema: Whether to use schema enrichment
        prompt_routing: Prompt routing value (integer)
        
    Returns:
        tuple: (success, error_message)
    """
    settings_endpoint = f"{api_url_base}/settings"
    
    # Prepare settings payload
    settings_payload = {
        "ollama_model": model_name,
        "enrich_schema": enrich_schema,
        "prompt_routing": prompt_routing
    }
    
    try:
        # Make POST request to the settings endpoint
        response = requests.post(settings_endpoint, json=settings_payload, timeout=10)
        
        # Check response
        if response.status_code == 200:
            result = response.json()
            if result.get("code") == 0:
                print(f"Successfully set model to {model_name} with enrich_schema={enrich_schema}, prompt_routing={prompt_routing}")
                return True, None
            else:
                error_msg = f"Failed to set model settings: {result.get('code')}, {result.get('message')}"
                print(error_msg)
                return False, error_msg
        else:
            error_msg = f"Failed to set model settings: {response.status_code}, {response.text}"
            print(error_msg)
            return False, error_msg
    except requests.exceptions.Timeout:
        error_msg = "Request timed out while setting model settings"
        print(error_msg)
        return False, error_msg
    except requests.exceptions.ConnectionError:
        error_msg = f"Connection error: Could not connect to API server at {settings_endpoint}."
        print(error_msg)
        return False, error_msg
    except Exception as e:
        error_msg = f"Error setting model settings: {str(e)}"
        print(error_msg)
        return False, error_msg

def get_sqlite_base64(sqlite_path):
    """Read and encode SQLite file as Base64 string"""
    try:
        # Read SQLite file in binary mode
        with open(sqlite_path, 'rb') as f:
            sqlite_binary = f.read()
        
        # Encode binary data as Base64 string
        sqlite_base64 = base64.b64encode(sqlite_binary).decode('utf-8')
        return sqlite_base64, None
    except FileNotFoundError:
        return None, f"SQLite file not found at {sqlite_path}"
    except Exception as e:
        return None, f"Error processing SQLite file: {str(e)}"

def ensure_nltk_resources():
    """
    Check and download required NLTK resources for the evaluation script.
    """
    try:
        import nltk
        resources = ['punkt']
        
        for resource in resources:
            try:
                nltk.data.find(f'tokenizers/{resource}')
                print(f"NLTK resource '{resource}' is already available.")
            except LookupError:
                print(f"Downloading NLTK resource '{resource}'...")
                nltk.download(resource)
                print(f"NLTK resource '{resource}' has been downloaded.")
        
        return True
    except Exception as e:
        print(f"Error ensuring NLTK resources: {str(e)}")
        print("Please manually install NLTK and required resources:")
        print("  pip install nltk")
        print("  python -m nltk.downloader punkt")
        return False

def get_schema_enrich_info(db_id, schema_dir="schema/schema_qwen25"):
    """Get schema enrich information for a given database
    
    Args:
        db_id: Database ID
        schema_dir: Directory containing schema JSON files (default: schema/schema_qwen25)
    """
    try:
        # Read from specified schema directory
        schema_file = os.path.join(schema_dir, f"{db_id}.json")
        with open(schema_file, 'r') as f:
            schema_info = json.load(f)
        if schema_info["code"] != 0:
            return None
        return {
            "database_description": schema_info["data"]["database_description"],
            "enriched_schema": schema_info["data"]["enriched_schema"]
        }
    except FileNotFoundError:
        print(f"Schema file not found: {schema_file}")
        return None
    except requests.exceptions.Timeout:
        return None
    except Exception as e:
        print(f"Error loading schema for {db_id}: {str(e)}")
        return None

def generate_sql_for_question(query, db_id, sqlite_base64, schema_dir="schema/schema_qwen25", api_url="http://localhost:8383/query", timeout=30):
    """Generate SQL for a given question using the API"""
    try:        
        # Create connection payload for SQLite
        sqlite_connection_payload = {
            "file": sqlite_base64,
            "dbType": "sqlite",
            "schema_enrich_info": get_schema_enrich_info(db_id, schema_dir)
        }
        
        # Prepare request payload
        payload = {
            "query": query,
            "connection_payload": sqlite_connection_payload
        }

        # Make POST request to the endpoint with shorter timeout
        response = requests.post(api_url, json=payload, timeout=timeout)

        # Check response
        if response.status_code == 200:
            result = response.json()
            return result.get("data"), None
        else:
            return None, f"API Error: {response.status_code}, {response.text}"
    except requests.exceptions.Timeout:
        return None, f"Request timed out after {timeout} seconds. API server may be unavailable."
    except requests.exceptions.ConnectionError:
        return None, f"Connection error: Could not connect to API server at {api_url}. Server may be down."
    except requests.exceptions.RequestException as e:
        return None, f"Request failed: {str(e)}"
    except KeyboardInterrupt:
        print("\nInterrupt received during API call. Exiting...")
        raise
    except Exception as e:
        return None, f"Error: {str(e)}"

def generate_sql_for_question_with_retry(query, db_id, sqlite_base64, schema_dir="schema/schema_qwen25", api_url="http://localhost:8383/query", timeout=120, max_retries=3, retry_delay=5):
    """Generate SQL for a given question using the API with retry logic"""
    retries = 0
    last_error = None
    
    while retries < max_retries:
        sql, error = generate_sql_for_question(query, db_id, sqlite_base64, schema_dir, api_url, timeout)
        
        # If we got a valid SQL response, return it immediately
        if sql is not None and sql.strip() != "" and sql.strip() != "SELECT 0;":
            if retries > 0:
                print(f"Successfully generated SQL on retry {retries}")
            return sql, None
        
        # Increment retry counter
        retries += 1
        last_error = error if error else "Empty or placeholder SQL response"
        
        # If we've reached max retries, break out
        if retries >= max_retries:
            break
            
        print(f"Retry {retries}/{max_retries} for question: {query}")
        print(f"Previous attempt error: {last_error}")
        
        # Wait before retrying
        time.sleep(retry_delay)
    
    # If we get here, all retries failed
    return None, f"Failed after {max_retries} attempts. Last error: {last_error}"

def find_sqlite_file(database_dir, db_id):
    """
    Find the SQLite file for a db_id by searching in subdirectories.
    Returns the full path to the SQLite file if found, None otherwise.
    """
    # First try to find a directory with the exact db_id name
    db_dir = os.path.join(database_dir, db_id)
    if os.path.isdir(db_dir):
        sqlite_path = os.path.join(db_dir, f"{db_id}.sqlite")
        if os.path.isfile(sqlite_path):
            return sqlite_path
    
    # If not found, search all subdirectories
    for root, dirs, files in os.walk(database_dir):
        sqlite_filename = f"{db_id}.sqlite"
        if sqlite_filename in files:
            return os.path.join(root, sqlite_filename)
    
    return None

def create_log_directory(model_name, enrich_schema, prompt_routing, is_baseline=False):
    """Create and return path to a timestamped log directory with model info"""
    logs_dir = "logs"
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create a timestamp for the run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Format the directory name to include model and schema enrichment info
    model_name_clean = model_name.replace(":", "-")  # Replace any colons with hyphens for safe file naming
    enrich_text = "with_enrich" if enrich_schema else "no_enrich"
    baseline_text = "_baseline" if is_baseline else ""
    
    log_dir_name = f"{timestamp}_{model_name_clean}_{enrich_text}_prompt{prompt_routing}{baseline_text}"
    log_dir_path = os.path.join(logs_dir, log_dir_name)
    
    os.makedirs(log_dir_path, exist_ok=True)
    print(f"Created log directory: {log_dir_path}")
    
    return log_dir_path

def find_latest_log_directory():
    """Find the most recent timestamped log directory in the logs folder"""
    logs_dir = "logs"
    if not os.path.exists(logs_dir):
        return None
    
    # Get all subdirectories in the logs folder
    subdirs = [os.path.join(logs_dir, d) for d in os.listdir(logs_dir) 
               if os.path.isdir(os.path.join(logs_dir, d))]
    
    if not subdirs:
        return None
    
    # Sort by creation time, most recent first
    subdirs.sort(key=lambda x: os.path.getctime(x), reverse=True)
    
    # Return the most recent directory
    print(f"Found latest log directory: {subdirs[0]}")
    return subdirs[0]

def validate_sql_files(gold_file, pred_file):
    """
    Validate the format of gold and predicted SQL files before evaluation.
    Each line in the gold file should have exactly one tab character.
    
    Args:
        gold_file: Path to the gold SQL file
        pred_file: Path to the predicted SQL file
        
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        print(f"Validating files: {gold_file} and {pred_file}")
        
        # Check if files exist
        if not os.path.exists(gold_file):
            print(f"Error: Gold file {gold_file} does not exist")
            return False
            
        if not os.path.exists(pred_file):
            print(f"Error: Prediction file {pred_file} does not exist")
            return False
            
        # Check gold file format
        with open(gold_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:  # Skip empty lines
                continue
                
            parts = line.split('\t')
            if len(parts) != 2:
                print(f"Error: Line {i+1} in gold file has {len(parts)} parts instead of 2")
                print(f"Line content: {line}")
                return False
                
        # Check prediction file format (should be one SQL per line)
        with open(pred_file, 'r', encoding='utf-8') as f:
            pred_lines = f.readlines()
            
        # Check if counts match
        if len([l for l in lines if l.strip()]) != len([l for l in pred_lines if l.strip()]):
            print(f"Warning: Gold file has {len([l for l in lines if l.strip()])} non-empty lines, "
                  f"but prediction file has {len([l for l in pred_lines if l.strip()])} non-empty lines")
            
        return True
            
    except Exception as e:
        print(f"Error validating files: {str(e)}")
        traceback.print_exc()
        return False

def run_spider_test_pipeline(json_file_path, database_dir, output_dir, schema_dir="schema/schema_qwen25", api_url="http://localhost:8383/query", 
                           start_idx=0, end_idx=None, batch_size=None, delay=1, max_retries=3, retry_delay=2,
                           checkpoint_interval=5, resume=True):
    """
    Process all questions in a JSON file and generate SQL queries
    
    Args:
        json_file_path: Path to the JSON file with questions
        database_dir: Directory containing subdirectories with SQLite files
        output_dir: Directory to store the output files
        schema_dir: Directory containing schema JSON files
        api_url: API endpoint URL
        start_idx: Start index in JSON file (for resuming)
        end_idx: End index in JSON file (optional)
        batch_size: Number of questions to process (optional)
        delay: Delay between API calls in seconds (to avoid overwhelming the server)
        max_retries: Maximum number of retries for SQL generation
        retry_delay: Delay between retry attempts in seconds
        checkpoint_interval: Save checkpoint every N questions
        resume: Whether to try resuming from a saved checkpoint
        
    Returns:
        tuple: (success_count, failure_count, total_questions)
    """
    # Create directory for this JSON file's results
    json_file_name = os.path.basename(json_file_path).replace('.json', '')
    json_output_dir = os.path.join(output_dir, json_file_name)
    os.makedirs(json_output_dir, exist_ok=True)
    
    # Setup checkpoint directory
    checkpoint_dir = get_checkpoint_dir(output_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Define output files
    output_file = os.path.join(json_output_dir, "predict.txt")
    gold_file = os.path.join(json_output_dir, "gold.txt")
    log_file = os.path.join(json_output_dir, "run_log.txt")
    exception_file = os.path.join(json_output_dir, "exceptions.txt")
    
    # Check for checkpoint if resume is True
    resume_from_idx = start_idx
    db_cache = {}
    
    if resume:
        # First try to load checkpoint
        checkpoint_idx, checkpoint_db_cache = load_checkpoint(checkpoint_dir, json_file_path)
        
        # If there's a checkpoint, use it
        if checkpoint_idx is not None:
            resume_from_idx = checkpoint_idx + 1  # Resume from the next item after the checkpoint
            print(f"Resuming from checkpoint: index {resume_from_idx}")
            
            if checkpoint_db_cache:
                db_cache = checkpoint_db_cache
                print(f"Loaded {len(db_cache)} databases from checkpoint cache")
        
        # If no checkpoint but files exist, check existing results
        elif os.path.exists(output_file) and os.path.exists(gold_file):
            # Check how many items we've already processed by reading the files
            existing_count = verify_existing_results(output_dir, json_file_name, None)
            if existing_count > 0:
                resume_from_idx = start_idx + existing_count
                print(f"No checkpoint found, but detected {existing_count} processed items in output files")
                print(f"Resuming from index {resume_from_idx} based on existing output files")
    
    # If we're resuming, use the calculated index; otherwise use the provided start_idx
    start_idx = max(resume_from_idx, start_idx)
    
    # Set up logging to file (append if resuming)
    log_mode = 'a' if resume and os.path.exists(log_file) else 'w'
    exception_mode = 'a' if resume and os.path.exists(exception_file) else 'w'
    
    with open(log_file, log_mode, encoding='utf-8') as f:
        f.write(f"\n{'='*50}\n")
        f.write(f"Spider Test Pipeline Run Log - {'RESUMED' if resume and resume_from_idx > 0 else 'NEW'} RUN\n")
        f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"JSON file: {json_file_path}\n")
        f.write(f"Database directory: {database_dir}\n")
        f.write(f"API URL: {api_url}\n")
        f.write(f"Start index: {start_idx} {'(Resumed from checkpoint/existing files)' if resume and start_idx > 0 else ''}\n")
        f.write(f"End index: {end_idx}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Delay: {delay}\n")
        f.write(f"Max retries: {max_retries}\n")
        f.write(f"Retry delay: {retry_delay}\n")
        f.write(f"Checkpoint interval: {checkpoint_interval}\n\n")
    
    # Load JSON data
    try:
        print(f"Loading data from {json_file_path}...")
        with open(json_file_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"Successfully loaded data from {json_file_path}\n")
    except Exception as e:
        error_msg = f"Error loading JSON file {json_file_path}: {str(e)}"
        print(error_msg)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"{error_msg}\n")
        return 0, 0, 0
    
    # Calculate indices
    total_questions = len(json_data)
    print(f"Found {total_questions} questions in {json_file_path}")
    
    if end_idx is None:
        end_idx = total_questions
    else:
        end_idx = min(end_idx, total_questions)
    
    if batch_size is not None:
        end_idx = min(start_idx + batch_size, end_idx)
    
    # Prepare to process the data
    questions_to_process = json_data[start_idx:end_idx]
    log_message = f"Processing questions from index {start_idx} to {end_idx-1} ({len(questions_to_process)} questions)"
    print(log_message)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"{log_message}\n")
    
    # Create or append to output files
    file_mode = 'a' if resume and resume_from_idx > 0 else 'w'
    
    # Make sure the files exist even if we don't write to them
    if not os.path.exists(output_file):
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("")  # Create empty file
    
    if not os.path.exists(gold_file):
        with open(gold_file, 'w', encoding='utf-8') as f:
            f.write("")  # Create empty file
            
    # Create backup copies if we're resuming from existing files
    if resume and resume_from_idx > 0:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = os.path.join(json_output_dir, "backups")
        os.makedirs(backup_dir, exist_ok=True)
        
        for file_path in [output_file, gold_file]:
            if os.path.exists(file_path):
                backup_file = os.path.join(backup_dir, f"{os.path.basename(file_path)}.{timestamp}")
                shutil.copy2(file_path, backup_file)
                print(f"Created backup of {os.path.basename(file_path)} at {backup_file}")
    
    # Results storage for summary
    results = []
    
    # Process each question
    success_count = 0
    failure_count = 0
    skipped_count = 0
    api_unavailable_count = 0
    
    # Track progress for checkpoint saving
    last_checkpoint_index = start_idx - 1
    
    # Create progress bar
    pbar = tqdm.tqdm(total=len(questions_to_process), desc="Processing questions", unit="question")
    
    for i, item in enumerate(questions_to_process):
        current_index = start_idx + i
        db_id = item.get("db_id")
        question = item.get("question")
        gold_query = item.get("query", "")  # Original SQL query (for reference only)
        
        if not db_id or not question:
            log_message = f"Missing db_id or question at index {current_index}"
            print(log_message)
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"{log_message}\n")
                
            failure_count += 1
            skipped_count += 1
            results.append({"sql": "", "db_id": db_id, "error": "Missing data"})
            
            # Log the exception instead of writing to SQL files
            with open(exception_file, exception_mode, encoding='utf-8') as f:
                f.write(f"INDEX: {current_index}\n")
                f.write(f"DB_ID: {db_id}\n")
                f.write(f"QUESTION: {question}\n")
                f.write(f"ERROR: Missing db_id or question\n")
                f.write(f"GOLD_QUERY: {gold_query}\n")
                f.write("-" * 80 + "\n\n")
            
            pbar.update(1)
            continue
        
        log_message = f"\n[{i+1}/{len(questions_to_process)}] Processing: DB: {db_id}, Question: {question}"
        print(log_message)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"{log_message}\n")
        
        # Get or create the base64 encoding of the SQLite file
        if db_id in db_cache:
            sqlite_base64 = db_cache[db_id]
        else:
            # Find the SQLite file in the database directory structure
            sqlite_path = find_sqlite_file(database_dir, db_id)
            
            if not sqlite_path:
                error = f"SQLite file for database '{db_id}' not found in {database_dir} or its subdirectories"
                print(error)
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(f"{error}\n")
                    
                failure_count += 1
                skipped_count += 1
                results.append({"sql": "", "db_id": db_id, "error": error})
                
                # Log the exception instead of writing to SQL files
                with open(exception_file, exception_mode, encoding='utf-8') as f:
                    f.write(f"INDEX: {current_index}\n")
                    f.write(f"DB_ID: {db_id}\n")
                    f.write(f"QUESTION: {question}\n")
                    f.write(f"ERROR: {error}\n")
                    f.write(f"GOLD_QUERY: {gold_query}\n")
                    f.write("-" * 80 + "\n\n")
                
                pbar.update(1)
                continue
            
            log_message = f"Reading database from {sqlite_path}"
            print(log_message)
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"{log_message}\n")
                
            sqlite_base64, error = get_sqlite_base64(sqlite_path)
            
            if error:
                log_message = f"Error with database {db_id}: {error}"
                print(log_message)
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(f"{log_message}\n")
                    
                failure_count += 1
                skipped_count += 1
                results.append({"sql": "", "db_id": db_id, "error": error})
                
                # Log the exception instead of writing to SQL files
                with open(exception_file, exception_mode, encoding='utf-8') as f:
                    f.write(f"INDEX: {current_index}\n")
                    f.write(f"DB_ID: {db_id}\n")
                    f.write(f"QUESTION: {question}\n")
                    f.write(f"ERROR: {error}\n")
                    f.write(f"GOLD_QUERY: {gold_query}\n")
                    f.write("-" * 80 + "\n\n")
                
                pbar.update(1)
                continue
            
            # Cache the encoding
            db_cache[db_id] = sqlite_base64
        
        # Generate SQL with retry logic
        log_message = f"Generating SQL for question: {question} (with up to {max_retries} retries)"
        print(log_message)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"{log_message}\n")
            
        sql, error = generate_sql_for_question_with_retry(
            question, 
            db_id,
            sqlite_base64, 
            schema_dir,
            api_url, 
            timeout=300,
            max_retries=max_retries,
            retry_delay=retry_delay
        )
        
        if error:
            log_message = f"Error generating SQL: {error}"
            print(log_message)
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"{log_message}\n")
                
            failure_count += 1
            results.append({"sql": "", "db_id": db_id, "error": error})
            
            # Log the exception instead of writing to SQL files
            with open(exception_file, exception_mode, encoding='utf-8') as f:
                f.write(f"INDEX: {current_index}\n")
                f.write(f"DB_ID: {db_id}\n")
                f.write(f"QUESTION: {question}\n")
                f.write(f"ERROR: {error}\n")
                f.write(f"GOLD_QUERY: {gold_query}\n")
                f.write("-" * 80 + "\n\n")
            
            # If error contains "API server may be unavailable" then stop the inference
            # if "API server may be unavailable" in error:
            #     api_unavailable_count += 1
            #     print("Stopping inference due to API server unavailability")
            #     # Save checkpoint before stopping
            #     save_checkpoint(checkpoint_dir, json_file_path, current_index, db_cache)
            #     break
        elif sql is None or sql.strip() == "SELECT 0;":
            # Handle empty or placeholder SQL as an exception
            log_message = f"Generated SQL is None or a placeholder: {sql}"
            print(log_message)
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"{log_message}\n")
                
            failure_count += 1
            results.append({"sql": "", "db_id": db_id, "error": "Empty or placeholder SQL"})
            
            # Log the exception
            with open(exception_file, exception_mode, encoding='utf-8') as f:
                f.write(f"INDEX: {current_index}\n")
                f.write(f"DB_ID: {db_id}\n")
                f.write(f"QUESTION: {question}\n")
                f.write(f"ERROR: Empty or placeholder SQL\n")
                f.write(f"GENERATED_SQL: {sql}\n")
                f.write(f"GOLD_QUERY: {gold_query}\n")
                f.write("-" * 80 + "\n\n")
        else:
            # Success with valid SQL!
            log_message = ""
            if "SELECT" in sql.upper():
                log_message = f"\033[92mGenerated valid SQL: {sql}\033[0m"
            else:
                log_message = f"\033[93mGenerated valid SQL: {sql}\033[0m"
            print(log_message)
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"{log_message}\n")
                
            success_count += 1
            results.append({"sql": sql, "db_id": db_id, "error": None})
            
            # Clean up SQL before writing to file
            sql = sql.replace("\n", " ").strip(";") + " ;"
            
            # Write SQL to output file
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(f"{sql}\n")
            
            # Clean up gold query before writing to file
            gold_query = gold_query.replace("\n", " ").replace("\t", " ").strip(";") + " ;"
            
            # Write gold query and db_id to gold file
            with open(gold_file, 'a', encoding='utf-8') as f:
                f.write(f"{gold_query}\t{db_id}\n")
            
            log_message = f"SQL written to {output_file}"
            print(log_message)
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"{log_message}\n")
        
        # Check if we should save a checkpoint
        if checkpoint_interval > 0 and (i + 1) % checkpoint_interval == 0:
            save_checkpoint(checkpoint_dir, json_file_path, current_index, db_cache)
            last_checkpoint_index = current_index
        
        # Add delay to avoid overwhelming the server
        if i < len(questions_to_process) - 1 and delay > 0:
            time.sleep(delay)
        
        pbar.update(1)
    
    pbar.close()
    
    # Save final checkpoint if we've processed more items since the last checkpoint
    if checkpoint_interval > 0 and (start_idx + len(questions_to_process) - 1) > last_checkpoint_index:
        final_index = start_idx + len(questions_to_process) - 1
        save_checkpoint(checkpoint_dir, json_file_path, final_index, db_cache)
    
    # Print summary for this JSON file
    summary = [
        f"\n===== Test Summary for {json_file_name} =====",
        f"Total questions in file: {total_questions}",
        f"Questions to process: {len(questions_to_process)}",
        f"Questions processed: {success_count + failure_count}",
        f"Successful: {success_count}",
        f"Failed: {failure_count}",
        f"Skipped: {skipped_count}",
        f"API unavailable: {api_unavailable_count}",
        f"Results written to: {output_file}",
        f"Gold file written to: {gold_file}",
        f"=========================================="
    ]
    
    for line in summary:
        print(line)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"{line}\n")
    
    # After processing all questions
    if success_count > 0:
        # Validate the output files before suggesting evaluation
        if validate_sql_files(gold_file, output_file):
            print(f"Files validated successfully. Ready for evaluation.")
            
            # Generate command suggestion for evaluation
            eval_cmd = f"python evaluation.py --gold {gold_file} --pred {output_file} --db {database_dir} --table metadata/tables.json --etype all --plug_value --keep_distinct"
            print(f"\nTo evaluate the results for {json_file_name} with test-suite-sql-eval, use:")
            print(eval_cmd)
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"\nEvaluation command:\n{eval_cmd}\n")
            
            # Also suggest the exact matching evaluation
            match_eval_cmd = f"python evaluation.py --gold {gold_file} --pred {output_file} --db {database_dir} --table metadata/tables.json --etype match"
            print(f"\nOr for exact matching evaluation of {json_file_name}, use:")
            print(match_eval_cmd)
        else:
            print(f"⚠️ File validation failed. Please check the format before evaluation.")
    
    # Write a results summary file in JSON format
    results_summary = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_questions": total_questions,
        "questions_to_process": len(questions_to_process),
        "questions_processed": success_count + failure_count,
        "successful": success_count,
        "failed": failure_count,
        "skipped": skipped_count,
        "api_unavailable": api_unavailable_count,
        "success_rate": success_count / (success_count + failure_count) if (success_count + failure_count) > 0 else 0,
        "json_file": json_file_path,
        "database_dir": database_dir,
        "output_file": output_file,
        "gold_file": gold_file,
        "api_url": api_url,
        "max_retries": max_retries,
        "retry_delay": retry_delay,
        "indices": {
            "start": start_idx,
            "end": end_idx-1
        },
        "eval_commands": {
            "exec": eval_cmd,
            "match": match_eval_cmd
        }
    }
    
    results_file = os.path.join(json_output_dir, "results_summary.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2)
    
    return success_count, failure_count, len(questions_to_process)

def process_json_dir(json_dir, database_dir, output_dir, schema_dir="schema/schema_qwen25", api_url="http://localhost:8383/query", start_idx=0, end_idx=None, batch_size=None, delay=1, max_retries=3, retry_delay=5, checkpoint_interval=5, resume=True):
    """
    Process all JSON files in the given directory
    
    Args:
        json_dir: Directory containing JSON files to process
        database_dir: Directory containing subdirectories with SQLite files
        output_dir: Directory to store the output files
        schema_dir: Directory containing schema JSON files
        api_url: API endpoint URL
        start_idx: Start index in JSON files (for resuming)
        end_idx: End index in JSON files (optional)
        batch_size: Number of questions to process per file (optional)
        delay: Delay between API calls in seconds
        max_retries: Maximum number of retries for SQL generation
        retry_delay: Delay between retry attempts in seconds
        checkpoint_interval: Save checkpoint every N questions (0 to disable)
        resume: Whether to try to resume from last checkpoint
    """
    # Find all JSON files in the directory
    json_files = glob.glob(os.path.join(json_dir, "*.json"))
    
    if not json_files:
        print(f"No JSON files found in {json_dir}")
        return False
    
    print(f"Found {len(json_files)} JSON files to process:")
    for i, json_file in enumerate(json_files):
        print(f"  {i+1}. {os.path.basename(json_file)}")
    
    # Create checkpoint directory
    checkpoint_dir = get_checkpoint_dir(output_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create overall summary file
    summary_file = os.path.join(output_dir, "overall_summary.json")
    
    # Load existing summary if resuming and it exists
    if resume and os.path.exists(summary_file):
        try:
            with open(summary_file, 'r', encoding='utf-8') as f:
                overall_summary = json.load(f)
                print(f"Loaded existing summary file from {summary_file}")
        except Exception as e:
            print(f"Error loading existing summary: {e}")
            # Create a new summary
            overall_summary = {
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "json_dir": json_dir,
                "database_dir": database_dir,
                "output_dir": output_dir,
                "api_url": api_url,
                "max_retries": max_retries,
                "retry_delay": retry_delay,
                "files_processed": [],
            }
    else:
        overall_summary = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "json_dir": json_dir,
            "database_dir": database_dir,
            "output_dir": output_dir,
            "api_url": api_url,
            "max_retries": max_retries,
            "retry_delay": retry_delay,
            "files_processed": [],
        }
    
    # Track processed file names
    processed_files = {item["file"] for item in overall_summary.get("files_processed", [])}
    
    # Process each JSON file
    total_success = 0
    total_failure = 0
    total_questions = 0
    
    for i, json_file in enumerate(json_files):
        json_file_name = os.path.basename(json_file)
        
        # Skip if this file has been fully processed already
        # We consider a file fully processed if it's in the summary list and the checkpoint is at the end
        if json_file_name in processed_files:
            print(f"\n[{i+1}/{len(json_files)}] Skipping {json_file_name} - already fully processed")
            
            # Find the summary for this file and add its counts
            for file_summary in overall_summary["files_processed"]:
                if file_summary["file"] == json_file_name:
                    total_success += file_summary["successful"]
                    total_failure += file_summary["failed"]
                    total_questions += file_summary["total_questions"]
                    break
            
            continue
                
        print(f"\n[{i+1}/{len(json_files)}] Processing {json_file_name}...")
        
        file_success, file_failure, file_total = run_spider_test_pipeline(
            json_file_path=json_file,
            database_dir=database_dir,
            output_dir=output_dir,
            schema_dir=schema_dir,
            api_url=api_url,
            start_idx=start_idx,
            end_idx=end_idx,
            batch_size=batch_size,
            delay=delay,
            max_retries=max_retries,
            retry_delay=retry_delay,
            checkpoint_interval=checkpoint_interval,
            resume=resume
        )
        
        # Update overall summary
        file_exists = False
        
        # Check if file is already in summary, update if yes
        for j, file_summary in enumerate(overall_summary["files_processed"]):
            if file_summary["file"] == json_file_name:
                overall_summary["files_processed"][j] = {
                    "file": json_file_name,
                    "total_questions": file_total,
                    "successful": file_success,
                    "failed": file_failure,
                    "success_rate": file_success / file_total if file_total > 0 else 0
                }
                file_exists = True
                break
        
        # If not, add it
        if not file_exists:
            file_summary = {
                "file": json_file_name,
                "total_questions": file_total,
                "successful": file_success,
                "failed": file_failure,
                "success_rate": file_success / file_total if file_total > 0 else 0
            }
            overall_summary["files_processed"].append(file_summary)
        
        # Update overall counts
        total_success += file_success
        total_failure += file_failure
        total_questions += file_total
        
        # Update the summary file after each JSON file
        overall_summary["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        overall_summary["total_statistics"] = {
            "total_questions": total_questions,
            "total_successful": total_success,
            "total_failed": total_failure,
            "overall_success_rate": total_success / total_questions if total_questions > 0 else 0
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(overall_summary, f, indent=2)
    
    # Add overall statistics one final time
    overall_summary["total_statistics"] = {
        "total_questions": total_questions,
        "total_successful": total_success,
        "total_failed": total_failure,
        "overall_success_rate": total_success / total_questions if total_questions > 0 else 0
    }
    
    # Write overall summary
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(overall_summary, f, indent=2)
    
    print("\n===== Overall Summary =====")
    print(f"Total JSON files processed: {len(json_files)}")
    print(f"Total questions processed: {total_questions}")
    print(f"Total successful: {total_success}")
    print(f"Total failed: {total_failure}")
    print(f"Overall success rate: {total_success / total_questions:.2f}" if total_questions > 0 else "No questions processed")
    print(f"Overall summary written to: {summary_file}")
    print("===========================")
    
    return total_success > 0

def save_checkpoint(checkpoint_dir, json_file, current_idx, db_cache=None):
    """
    Save progress checkpoint to allow resuming from this point.
    
    Args:
        checkpoint_dir: Directory to save the checkpoint
        json_file: Path to the JSON file being processed
        current_idx: Current index in the JSON file
        db_cache: Optional cache of DB encodings to save
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create checkpoint data
    checkpoint_data = {
        "json_file": os.path.basename(json_file),
        "last_processed_idx": current_idx,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Save checkpoint data
    checkpoint_file = os.path.join(checkpoint_dir, f"{os.path.basename(json_file)}.checkpoint.json")
    with open(checkpoint_file, 'w', encoding='utf-8') as f:
        json.dump(checkpoint_data, f, indent=2)
    
    # Save DB cache if provided (can save significant time for large DBs)
    if db_cache:
        cache_file = os.path.join(checkpoint_dir, f"{os.path.basename(json_file)}.dbcache.pkl")
        with open(cache_file, 'wb') as f:
            pickle.dump(db_cache, f)
    
    print(f"Checkpoint saved at index {current_idx} for {os.path.basename(json_file)}")
    return checkpoint_file

def load_checkpoint(checkpoint_dir, json_file):
    """
    Load checkpoint to resume processing from last saved point.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        json_file: Path to the JSON file to check for checkpoints
        
    Returns:
        Tuple of (last_processed_idx, db_cache)
    """
    # Check if checkpoint exists
    checkpoint_file = os.path.join(checkpoint_dir, f"{os.path.basename(json_file)}.checkpoint.json")
    if not os.path.exists(checkpoint_file):
        return None, None
    
    # Load checkpoint data
    with open(checkpoint_file, 'r', encoding='utf-8') as f:
        checkpoint_data = json.load(f)
    
    # Verify this checkpoint is for the correct JSON file
    if checkpoint_data.get("json_file") != os.path.basename(json_file):
        print(f"Warning: Checkpoint file mismatch. Expected: {os.path.basename(json_file)}, Found: {checkpoint_data.get('json_file')}")
        return None, None
    
    # Get last processed index
    last_idx = checkpoint_data.get("last_processed_idx", 0)
    
    # Try to load DB cache if it exists
    db_cache = None
    cache_file = os.path.join(checkpoint_dir, f"{os.path.basename(json_file)}.dbcache.pkl")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                db_cache = pickle.load(f)
            print(f"Loaded DB cache with {len(db_cache)} databases")
        except Exception as e:
            print(f"Error loading DB cache: {e}")
    
    print(f"Loaded checkpoint for {os.path.basename(json_file)} at index {last_idx} from {checkpoint_file}")
    return last_idx, db_cache

def get_checkpoint_dir(output_dir):
    """Get the directory for storing checkpoints"""
    return os.path.join(output_dir, "checkpoints")

def verify_existing_results(output_dir, json_file_name, total_items):
    """
    Verify existing SQL results to determine where to resume from.
    
    Returns:
        Tuple of (predict_sql_line_count, gold_sql_line_count)
    """
    json_output_dir = os.path.join(output_dir, json_file_name)
    predict_file = os.path.join(json_output_dir, "predict.txt")
    gold_file = os.path.join(json_output_dir, "gold.txt")
    
    predict_count = 0
    gold_count = 0
    
    if os.path.exists(predict_file):
        with open(predict_file, 'r', encoding='utf-8') as f:
            predict_count = sum(1 for _ in f)
    
    if os.path.exists(gold_file):
        with open(gold_file, 'r', encoding='utf-8') as f:
            gold_count = sum(1 for _ in f)
    
    # Verify both files have same number of lines
    if predict_count != gold_count:
        print(f"Warning: predict.txt ({predict_count} items) and gold.txt ({gold_count} items) have different line counts")
        # Use the smaller number as the safe point to resume from
        return min(predict_count, gold_count)
    
    return predict_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spider Test Pipeline for SQL Generation")
    parser.add_argument("--json-dir", type=str, required=False,
                      help="Directory containing JSON files to process")
    parser.add_argument("--json-file", type=str, required=False,
                      help="Single JSON file to process")
    parser.add_argument("--database-dir", type=str, default="./database", 
                      help="Directory containing subdirectories with SQLite files")
    parser.add_argument("--schema-dir", type=str, default="schema/schema_qwen25",
                      help="Directory containing schema JSON files")
    parser.add_argument("--output-dir", type=str, default=None,
                      help="Directory to store output files (defaults to timestamped directory in 'logs')")
    parser.add_argument("--api-url", type=str, default="http://localhost:8383/query", 
                      help="API endpoint URL")
    parser.add_argument("--start-idx", type=int, default=0, 
                      help="Start index in JSON files")
    parser.add_argument("--end-idx", type=int, default=None, 
                      help="End index in JSON files")
    parser.add_argument("--batch-size", type=int, default=None, 
                      help="Number of questions to process per file")
    parser.add_argument("--delay", type=float, default=1.0, 
                      help="Delay between API calls in seconds")
    parser.add_argument("--max-retries", type=int, default=3,
                      help="Maximum number of retries for SQL generation when response is None")
    parser.add_argument("--retry-delay", type=float, default=5.0,
                      help="Delay between retry attempts in seconds")
    parser.add_argument("--ensure-nltk", action="store_true",
                      help="Check and download required NLTK resources for evaluation")
    parser.add_argument("--checkpoint-interval", type=int, default=5,
                      help="Save checkpoint every N questions (0 to disable)")
    parser.add_argument("--no-resume", action="store_true",
                      help="Don't try to resume from checkpoints")
    parser.add_argument("--resume", action="store_true",
                      help="Automatically resume from latest logs directory")
    parser.add_argument("--model", type=str, default="phi4",
                      help="Model to use (e.g., 'qwen2.5-coder:14b', 'phi4')")
    parser.add_argument("--enrich-schema", action="store_true", default=False,
                      help="Whether to use schema enrichment")
    parser.add_argument("--prompt-routing", type=int, default=1,
                      help="Prompt routing value (integer)")
    parser.add_argument("--baseline", action="store_true", default=False,
                      help="Use the baseline endpoint (/query-baseline) instead of the default endpoint (/query)")
    
    args = parser.parse_args()
    
    # Validate that either json-dir or json-file is provided
    if not args.json_dir and not args.json_file:
        parser.error("Either --json-dir or --json-file must be provided")
    if args.json_dir and args.json_file:
        parser.error("Cannot specify both --json-dir and --json-file")
    
    # Check for NLTK resources if requested
    if args.ensure_nltk:
        print("Checking and downloading required NLTK resources...")
        ensure_nltk_resources()
    
    # Extract the base API URL from the query endpoint for setting model settings
    api_base_url = args.api_url.rsplit('/', 1)[0]  # Remove the last part (/query) to get the base URL
    
    # Set the model and schema enrichment settings
    settings_success, settings_error = set_model_settings(api_base_url, args.model, args.enrich_schema, args.prompt_routing)
    if not settings_success:
        print(f"Failed to set model settings: {settings_error}")
        print("Exiting...")
        sys.exit(1)
    
    # Modify API URL if baseline flag is set
    if args.baseline:
        args.api_url = args.api_url.replace("/query", "/query-baseline")
        print(f"Using baseline endpoint: {args.api_url}")
    
    # Handle resuming from latest logs directory
    if args.resume:
        latest_log_dir = find_latest_log_directory()
        if latest_log_dir:
            print(f"Auto-resuming from latest logs directory: {latest_log_dir}")
            args.output_dir = latest_log_dir
        else:
            print("No previous logs directory found. Starting with a new logs directory.")
            args.output_dir = create_log_directory(args.model, args.enrich_schema, args.prompt_routing, args.baseline)
    # Create a timestamped log directory if output_dir not specified
    elif args.output_dir is None:
        args.output_dir = create_log_directory(args.model, args.enrich_schema, args.prompt_routing, args.baseline)
    else:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Save model settings to the log directory
    model_settings = {
        "model": args.model,
        "enrich_schema": args.enrich_schema,
        "prompt_routing": args.prompt_routing,
        "api_url": args.api_url,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(os.path.join(args.output_dir, "model_settings.json"), 'w') as f:
        json.dump(model_settings, f, indent=2)
    
    print("Spider Test Pipeline")
    print("===================")
    print("This script processes JSON files and generates SQL using the text-to-SQL API.")
    print(f"Model settings: {args.model} with enrich_schema={args.enrich_schema}, prompt_routing={args.prompt_routing}")
    print(f"Make sure your Flask app is running at {args.api_url}")
    print(f"Output will be saved to: {args.output_dir}")
    print(f"SQL generation will retry up to {args.max_retries} times if needed")
    print(f"Checkpoint interval: {args.checkpoint_interval} questions (0 = disabled)")
    print(f"Resume mode: {'Auto-resume from latest' if args.resume else 'No resume' if args.no_resume else 'Resume if checkpoint exists'}\n")
    
    # Process either a single file or a directory
    if args.json_file:
        print(f"Processing single JSON file: {args.json_file}")
        success = run_spider_test_pipeline(
            json_file_path=args.json_file,
            database_dir=args.database_dir,
            output_dir=args.output_dir,
            schema_dir=args.schema_dir,
            api_url=args.api_url,
            start_idx=args.start_idx,
            end_idx=args.end_idx,
            batch_size=args.batch_size,
            delay=args.delay,
            max_retries=args.max_retries,
            retry_delay=args.retry_delay,
            checkpoint_interval=args.checkpoint_interval,
            resume=not args.no_resume
        )
    else:
        print(f"Processing all JSON files in directory: {args.json_dir}")
        success = process_json_dir(
            json_dir=args.json_dir,
            database_dir=args.database_dir,
            output_dir=args.output_dir,
            schema_dir=args.schema_dir,
            api_url=args.api_url,
            start_idx=args.start_idx,
            end_idx=args.end_idx,
            batch_size=args.batch_size,
            delay=args.delay,
            max_retries=args.max_retries,
            retry_delay=args.retry_delay,
            checkpoint_interval=args.checkpoint_interval,
            resume=not args.no_resume
        )
    
    print("\nProcessing completed!")
    if args.ensure_nltk:
        print("Remember that you've already downloaded the required NLTK resources for evaluation.")
    else:
        print("If you plan to use the evaluation script with 'match' mode, make sure to install NLTK resources:")
        print("  pip install nltk")
        print("  python -m nltk.downloader punkt")
        print("Or run this script with --ensure-nltk flag to automatically download them.")
    
    sys.exit(0 if success else 1)