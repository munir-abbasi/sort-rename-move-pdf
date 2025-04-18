import os
import shutil
import PyPDF2
from PyPDF2.errors import PdfReadError
import tiktoken
from openai import OpenAI
import re
import time
import sys
from tqdm import tqdm
import argparse
import json
import requests
from typing import Dict, Any, Optional, List
import unicodedata
import platform

# Platform-specific imports for file locking
if platform.system() == "Windows":
    import msvcrt
else:
    import fcntl

# Import API clients conditionally to avoid hard dependencies
try:
    import google.generativeai as genai
    HAVE_GEMINI = True
except ImportError:
    HAVE_GEMINI = False

try:
    import anthropic
    HAVE_CLAUDE = True
except ImportError:
    HAVE_CLAUDE = False

# No need for try/except for Deepseek since we're using requests
HAVE_DEEPSEEK = True

# Cross-platform file locking
def lock_file(file_obj):
    """Apply a file lock appropriate for the platform"""
    if platform.system() == "Windows":
        msvcrt.locking(file_obj.fileno(), msvcrt.LK_LOCK, 1)
    else:
        fcntl.flock(file_obj.fileno(), fcntl.LOCK_EX)

def unlock_file(file_obj):
    """Release a file lock appropriate for the platform"""
    if platform.system() == "Windows":
        msvcrt.locking(file_obj.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        fcntl.flock(file_obj.fileno(), fcntl.LOCK_UN)

# Initialize encoding once
ENCODING = tiktoken.get_encoding("cl100k_base")
MAX_LENGTH = 15000
MAX_RETRIES = 3
RETRY_DELAY = 2

# Add to constants at the top of the file
ERROR_LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pdf_processing_errors.log")

# Available AI providers
AI_PROVIDERS = {
    "openai": [
        # Current recommended models (as of May 2024)
        "gpt-4o", 
        "gpt-4-turbo",
        "gpt-3.5-turbo",
        # Specific model versions
        "gpt-4o-2024-05-13", 
        "gpt-4-turbo-2024-04-09",
        "gpt-3.5-turbo-0125",
        # Vision-capable models
        "gpt-4-vision-preview",
        "gpt-4o-vision"
    ],
    "gemini": ["gemini-pro"],
    "claude": ["claude-3-haiku", "claude-3-sonnet", "claude-3-opus"],
    "deepseek": ["deepseek-chat"]
}

# Default prompts for each provider
SYSTEM_PROMPTS = {
    "openai": "Generate a descriptive filename based on the document content. Use only English letters, numbers, and underscores. Keep it under 50 characters.",
    "gemini": "Generate a descriptive filename based on the document content. Use only English letters, numbers, and underscores. Keep it under 50 characters.",
    "claude": "Generate a descriptive filename based on the document content. Use only English letters, numbers, and underscores. Keep it under 50 characters.",
    "deepseek": "Generate a descriptive filename based on the document content. Use only English letters, numbers, and underscores. Keep it under 50 characters."
}

# Default models for each provider
DEFAULT_MODELS = {
    "openai": "gpt-3.5-turbo",
    "gemini": "gemini-pro",
    "claude": "claude-3-haiku",
    "deepseek": "deepseek-chat"
}

# Global variable for AI client
ai_client = None

class AIClient:
    """Factory class for different AI clients"""
    
    @staticmethod
    def create(provider: str, model: str, api_key: str) -> Any:
        """Create an AI client based on the provider"""
        if provider == "openai":
            return OpenAIClient(api_key, model)
        elif provider == "gemini":
            if not HAVE_GEMINI:
                raise ImportError("Please install Google Generative AI: pip install google-generativeai")
            return GeminiClient(api_key, model)
        elif provider == "claude":
            if not HAVE_CLAUDE:
                raise ImportError("Please install Anthropic: pip install anthropic")
            return ClaudeClient(api_key, model)
        elif provider == "deepseek":
            return DeepseekClient(api_key, model)
        else:
            raise ValueError(f"Unsupported AI provider: {provider}")

class OpenAIClient:
    """OpenAI API client"""
    
    def __init__(self, api_key: str, model: str):
        self.client = OpenAI(api_key=api_key)
        self.model = model
    
    def generate_filename(self, content: str) -> str:
        """Generate a filename from OpenAI API"""
        try:
            # Use newer OpenAI API parameters
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPTS["openai"]},
                    {"role": "user", "content": content}
                ],
                max_tokens=60,
                temperature=0.3,
                timeout=30,
                seed=1234  # For more consistent outputs
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")

class GeminiClient:
    """Google Gemini API client"""
    
    def __init__(self, api_key: str, model: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
    
    def generate_filename(self, content: str) -> str:
        """Generate a filename using Gemini API"""
        try:
            response = self.model.generate_content(
                [SYSTEM_PROMPTS["gemini"], content],
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=60,
                    temperature=0.2
                )
            )
            
            # Check if response has expected structure
            if not hasattr(response, 'text'):
                raise AttributeError("Unexpected response format from Gemini API")
            
            return response.text
        except Exception as e:
            raise Exception(f"Gemini API error: {str(e)}")

class ClaudeClient:
    """Anthropic Claude API client"""
    
    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
    
    def generate_filename(self, content: str) -> str:
        """Generate a filename using Claude API"""
        try:
            message = self.client.messages.create(
                model=self.model,
                system=SYSTEM_PROMPTS["claude"],
                max_tokens=60,
                messages=[
                    {"role": "user", "content": content}
                ]
            )
            
            # Handle various response formats
            if hasattr(message, 'content') and isinstance(message.content, list) and len(message.content) > 0:
                if hasattr(message.content[0], 'text'):
                    return message.content[0].text
            
            # Try to extract the text from different possible formats
            if hasattr(message, 'content'):
                if isinstance(message.content, str):
                    return message.content
                elif isinstance(message.content, dict) and 'text' in message.content:
                    return message.content['text']
            
            raise ValueError("Unable to extract text from Claude API response")
        except Exception as e:
            raise Exception(f"Claude API error: {str(e)}")

class DeepseekClient:
    """Deepseek API client using direct HTTP requests"""
    
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.deepseek.com/v1/chat/completions"
    
    def generate_filename(self, content: str) -> str:
        """Generate a filename using Deepseek API"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPTS["deepseek"]},
                {"role": "user", "content": content}
            ],
            "max_tokens": 60,
            "temperature": 0.2
        }
        
        try:
            response = requests.post(self.base_url, headers=headers, json=data, timeout=30)
            
            if response.status_code != 200:
                raise Exception(f"Status code: {response.status_code}, Response: {response.text}")
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except requests.RequestException as e:
            raise Exception(f"Deepseek API request error: {str(e)}")
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            raise Exception(f"Deepseek API response parsing error: {str(e)}")
        except Exception as e:
            raise Exception(f"Deepseek API error: {str(e)}")

def get_api_details(provider: str, model: str):
    """Get API key and check if it's valid for the given provider"""
    # Validate provider first
    if provider not in AI_PROVIDERS:
        available_providers = ", ".join(AI_PROVIDERS.keys())
        raise ValueError(f"Unsupported provider '{provider}'. Available providers: {available_providers}")
    
    env_var_name = f"{provider.upper()}_API_KEY"
    api_key = os.environ.get(env_var_name)
    
    if not api_key:
        api_key = input(f"Please enter your {provider.capitalize()} API key: ").strip()
        if not api_key:
            raise ValueError(f"{provider.capitalize()} API key is required.")
    
    # Validate model
    if model not in AI_PROVIDERS.get(provider, []):
        available_models = ", ".join(AI_PROVIDERS.get(provider, ["No models available"]))
        raise ValueError(f"Invalid model '{model}' for provider '{provider}'. Available models: {available_models}")
    
    return api_key

def sort_and_rename_pdfs(input_folder, corrupted_folder, renamed_folder, provider="openai", model=None):
    """Sort and rename PDF files based on their content."""
    # Validate input paths
    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist.")
        return False
    
    # Set default model if not specified
    if model is None:
        model = DEFAULT_MODELS.get(provider, AI_PROVIDERS[provider][0])
    
    # Get API key and create client
    api_key = get_api_details(provider, model)
    global ai_client
    ai_client = AIClient.create(provider, model, api_key)
    
    # Create output directories
    os.makedirs(corrupted_folder, exist_ok=True)
    os.makedirs(renamed_folder, exist_ok=True)

    # Get list of PDF files
    pdf_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.pdf')]
    total_files = len(pdf_files)
    
    if total_files == 0:
        print(f"No PDF files found in {input_folder}")
        return True

    # Create a progress file to track processed files
    progress_file = os.path.join(renamed_folder, ".progress")
    processed_files = set()
    if os.path.exists(progress_file):
        try:
            with open(progress_file, "r") as f:
                processed_files = set(line.strip() for line in f)
        except Exception as e:
            print(f"Warning: Could not read progress file: {str(e)}")
    
    try:
        with open(progress_file, "a") as progress_f:
            with tqdm(total=total_files, desc="Processing PDFs", unit="file") as pbar:
                for filename in pdf_files:
                    if filename in processed_files:
                        pbar.update(1)
                        continue
                        
                    input_path = os.path.normpath(os.path.join(input_folder, filename))
                    if not os.path.exists(input_path):
                        print(f"\nWarning: File {filename} no longer exists, skipping.")
                        pbar.update(1)
                        continue
                        
                    process_pdf(input_path, filename, corrupted_folder, renamed_folder, pbar, progress_f)
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Progress has been saved.")
        return True
    except Exception as e:
        print(f"\nError during processing: {str(e)}")
        return False
    
    return True

def process_pdf(input_path, filename, corrupted_folder, renamed_folder, pbar, progress_f):
    """Process a single PDF file with error handling."""
    try:
        # Check if file exists and is accessible
        if not os.path.isfile(input_path):
            raise FileNotFoundError(f"File not found or not accessible: {input_path}")
            
        # Extract text and validate PDF in one operation
        pdf_content = pdfs_to_text_string(input_path)
        
        # If pdf_content starts with "Error", the file is corrupted
        if pdf_content.startswith("Error"):
            raise PdfReadError(pdf_content)
        
        # Get new filename with retries
        new_file_name = get_new_filename_with_retry(pdf_content)
        new_file_name = validate_and_trim_filename(new_file_name)
        
        # Handle duplicate filenames with sequential numbering
        new_file_name = handle_duplicate_filename(new_file_name, renamed_folder)
        
        # Move file to renamed folder
        new_filepath = os.path.join(renamed_folder, new_file_name + ".pdf")
        shutil.move(input_path, new_filepath)
        pbar.set_postfix({"Status": "Renamed", "New Name": new_file_name})
        
        # Record progress
        try:
            lock_file(progress_f)
            progress_f.write(f"{filename}\n")
            progress_f.flush()
        finally:
            unlock_file(progress_f)
        
    except (PdfReadError, OSError, FileNotFoundError) as e:
        error_msg = f"Error with file {filename}: {str(e)}"
        print(f"\n{error_msg}")
        try:
            # Only attempt to move the file if it exists
            if os.path.exists(input_path):
                corrupted_path = os.path.join(corrupted_folder, filename)
                shutil.move(input_path, corrupted_path)
                pbar.set_postfix({"Status": "Corrupted", "Moved to": corrupted_folder})
            else:
                pbar.set_postfix({"Status": "Error", "Message": "File not found"})
            
            try:
                lock_file(progress_f)
                progress_f.write(f"{filename}\n")
                progress_f.flush()
            finally:
                unlock_file(progress_f)
        except OSError as move_error:
            pbar.set_postfix({"Status": "Error", "Message": str(move_error)})
    except Exception as e:
        error_msg = f"Unexpected error with file {filename}: {str(e)}"
        print(f"\n{error_msg}")
        # Log to error file
        try:
            with open(ERROR_LOG_FILE, "a") as log:
                log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}: {error_msg}\n")
        except Exception as log_error:
            print(f"Warning: Could not write to error log: {str(log_error)}")
        pbar.set_postfix({"Status": "Error", "Message": "Unexpected error"})
    
    pbar.update(1)

def get_new_filename_with_retry(pdf_content, max_retries=MAX_RETRIES):
    """Get new filename with retry mechanism for API failures."""
    timeout_count = 0
    for attempt in range(max_retries):
        try:
            return get_filename_from_ai(pdf_content)
        except Exception as e:
            error_str = str(e).lower()
            
            # Check if it's a timeout-related error
            is_timeout = any(timeout_term in error_str for timeout_term in ["timeout", "timed out", "connection", "network"])
            
            if is_timeout:
                timeout_count += 1
                wait_time = RETRY_DELAY * (2 ** timeout_count)  # Exponential backoff for network issues
                print(f"\nAPI timeout/network error: {str(e)}. Retrying in {wait_time} seconds...")
            else:
                wait_time = RETRY_DELAY * (attempt + 1)  # Linear backoff for other errors
                print(f"\nAPI error: {str(e)}. Retrying in {wait_time} seconds...")
                
            time.sleep(wait_time)
            
            # Last attempt
            if attempt == max_retries - 1:
                print(f"\nFailed to get filename from API after {max_retries} attempts.")
                timestamp = time.strftime('%Y%m%d%H%M%S', time.gmtime())
                
                # Generate different prefix based on error type
                if timeout_count > 0:
                    return f"network_error_{timestamp}"
                else:
                    return f"untitled_document_{timestamp}"

def get_filename_from_ai(pdf_content):
    """Get a filename suggestion from the selected AI model"""
    if ai_client is None:
        raise RuntimeError("AI client not initialized. Call sort_and_rename_pdfs first.")
    return ai_client.generate_filename(pdf_content)

def validate_and_trim_filename(initial_filename):
    """Clean and validate a filename."""
    # Handle empty or None filenames
    if not initial_filename or initial_filename.isspace():
        timestamp = time.strftime('%Y%m%d%H%M%S', time.gmtime())
        return f'empty_file_{timestamp}'
    
    # Normalize unicode and remove characters that aren't allowed
    normalized = unicodedata.normalize('NFKD', initial_filename)
    cleaned_filename = re.sub(r'[^a-zA-Z0-9_]', '', normalized)
    
    # If filename is empty after cleaning, use a default name
    if not cleaned_filename:
        timestamp = time.strftime('%Y%m%d%H%M%S', time.gmtime())
        return f'invalid_name_{timestamp}'
    
    # Trim if too long (100 char max)
    return cleaned_filename[:100]

def handle_duplicate_filename(filename, folder):
    """Handle duplicate filenames by adding a sequential number."""
    base_filename = filename
    counter = 1
    
    while os.path.exists(os.path.join(folder, f"{filename}.pdf")):
        filename = f"{base_filename}_{counter}"
        counter += 1
    
    return filename

def pdfs_to_text_string(filepath, max_pages=None):
    """Extract text from a PDF file, handling errors."""
    try:
        with open(filepath, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            
            # Check if file is encrypted
            if reader.is_encrypted:
                return "Error: PDF is encrypted and cannot be processed without a password."
            
            # Handle very large PDFs
            total_pages = len(reader.pages)
            if max_pages is None and total_pages > 100:  # Default page limit for very large PDFs
                max_pages = 100
                print(f"Warning: PDF has {total_pages} pages. Limiting to first {max_pages} pages for performance.")
            
            # Limit number of pages if specified
            if max_pages and total_pages > max_pages:
                pages_to_process = reader.pages[:max_pages]
                print(f"Processing first {max_pages} pages of {total_pages} total pages")
            else:
                pages_to_process = reader.pages
            
            # Process pages in batches to avoid memory issues
            content = ""
            for page in pages_to_process:
                try:
                    page_text = page.extract_text() or ""
                    content += page_text + " "
                except Exception as e:
                    print(f"Warning: Could not extract text from a page: {str(e)}")
            
            content = content.strip()
            
            if not content:
                return "Empty PDF document with no extractable text."
            
            # Use the dedicated function for token handling
            return truncate_content_to_token_limit(content, MAX_LENGTH)
            
    except (IOError, OSError) as e:
        return f"Error opening PDF: {str(e)}"
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

def truncate_content_to_token_limit(content, max_tokens):
    """Truncate content to fit within token limit in a safe and efficient way."""
    try:
        # Get initial token count
        token_count = len(ENCODING.encode(content))
        
        # If already within limits, return as is
        if token_count <= max_tokens:
            return content
            
        # For very large content, do a quick initial truncation
        if token_count > max_tokens * 2:
            # Estimate bytes per token (rough approximation)
            bytes_per_token = len(content.encode('utf-8')) / token_count
            target_bytes = int(max_tokens * bytes_per_token * 0.9)  # 90% of estimated target
            content = content[:target_bytes]
            token_count = len(ENCODING.encode(content))
        
        # Now do a more precise truncation with binary search
        if token_count > max_tokens:
            # Binary search for the right truncation point
            low, high = 0, len(content)
            while high - low > 100:  # Close enough approximation
                mid = (low + high) // 2
                if len(ENCODING.encode(content[:mid])) <= max_tokens:
                    low = mid
                else:
                    high = mid
            
            # Final truncation
            content = content[:low]
            
        return content
    except Exception as e:
        print(f"Warning: Error during content truncation: {str(e)}")
        # Fallback to a very conservative truncation
        return content[:int(max_tokens * 3)]  # Assuming ~3 chars per token as fallback

def list_available_models():
    """List all available AI models by provider"""
    print("Available AI models by provider:")
    for provider, models in AI_PROVIDERS.items():
        print(f"\n{provider.capitalize()}:")
        for model in models:
            print(f"  - {model}")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Sort and rename PDF files based on their content",
        epilog="""
Examples:
  # Use default OpenAI model
  python sortrenamemovepdf.py -i ./input -c ./corrupted -r ./renamed

  # Use GPT-4o
  python sortrenamemovepdf.py -i ./input -c ./corrupted -r ./renamed -p openai -m gpt-4o
  
  # Use Claude
  python sortrenamemovepdf.py -i ./input -c ./corrupted -r ./renamed -p claude -m claude-3-sonnet
  
  # List all available models
  python sortrenamemovepdf.py --list-models
"""
    )
    parser.add_argument("--input", "-i", help="Input folder containing PDF files")
    parser.add_argument("--corrupted", "-c", help="Folder for corrupted PDF files")
    parser.add_argument("--renamed", "-r", help="Folder for renamed PDF files")
    parser.add_argument("--provider", "-p", help=f"AI provider (options: {', '.join(AI_PROVIDERS.keys())})", 
                        default="openai", choices=AI_PROVIDERS.keys())
    parser.add_argument("--model", "-m", help="Model to use for the selected provider (run with --list-models to see options)")
    parser.add_argument("--api-key", "-k", help="API key for the selected provider")
    parser.add_argument("--list-models", "-l", action="store_true", help="List all available models by provider and exit")
    return parser.parse_args()

if __name__ == "__main__":
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # List available models if requested
        if args.list_models:
            list_available_models()
            sys.exit(0)
        
        # Save original env var state if using command-line key
        original_env = None
        api_key_set = False
        
        try:
            if args.api_key:
                env_var = f"{args.provider.upper()}_API_KEY"
                original_env = (env_var, os.environ.get(env_var))
                os.environ[env_var] = args.api_key
                api_key_set = True
                print(f"Using provided {args.provider.capitalize()} API key")
            
            # Get folder paths from arguments or prompt user
            input_folder = args.input or input("Enter the input folder path: ")
            corrupted_folder = args.corrupted or input("Enter the corrupted PDFs folder path: ")
            renamed_folder = args.renamed or input("Enter the renamed PDFs folder path: ")

            # Process PDFs
            success = sort_and_rename_pdfs(input_folder, corrupted_folder, renamed_folder, args.provider, args.model)
            
            if success:
                print("PDF sorting and renaming completed successfully.")
                return_code = 0
            else:
                print("PDF sorting and renaming failed.")
                return_code = 1
        finally:
            # Restore original environment if modified
            if api_key_set and original_env:
                try:
                    env_var, original_value = original_env
                    if original_value is None:
                        os.environ.pop(env_var, None)
                    else:
                        os.environ[env_var] = original_value
                except Exception as e:
                    print(f"Warning: Could not restore original API key environment: {str(e)}")
            
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
        return_code = 130  # Standard return code for SIGINT
    except Exception as e:
        print(f"\nCritical error: {str(e)}")
        return_code = 1
    
    # Exit with appropriate code
    sys.exit(return_code)
