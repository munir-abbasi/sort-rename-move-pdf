import os
import shutil
import PyPDF2
from PyPDF2.errors import PdfReadError
import tiktoken
from openai import OpenAI
import re
import time
from tqdm import tqdm

client = OpenAI()

max_length = 15000

def sort_and_rename_pdfs(input_folder, corrupted_folder, renamed_folder):
    os.makedirs(corrupted_folder, exist_ok=True)
    os.makedirs(renamed_folder, exist_ok=True)

    pdf_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.pdf')]
    total_files = len(pdf_files)

    with tqdm(total=total_files, desc="Processing PDFs", unit="file") as pbar:
        for filename in pdf_files:
            input_path = os.path.join(input_folder, filename)

            try:
                with open(input_path, "rb") as pdf_file:
                    # Use PdfReader to check if the file is a valid PDF
                    PyPDF2.PdfReader(pdf_file)
                
                # If valid, rename the PDF
                pdf_content = pdfs_to_text_string(input_path)
                new_file_name = get_new_filename_from_openai(pdf_content)
                new_file_name = validate_and_trim_filename(new_file_name)
                
                if new_file_name + ".pdf" in os.listdir(renamed_folder):
                    new_file_name += "_" + os.urandom(4).hex()
                
                new_filepath = os.path.join(renamed_folder, new_file_name + ".pdf")
                shutil.move(input_path, new_filepath)
                pbar.set_postfix({"Status": "Renamed", "New Name": new_file_name})
                
            except (PdfReadError, OSError) as e:
                # File is corrupted or unreadable
                corrupted_path = os.path.join(corrupted_folder, filename)
                try:
                    shutil.move(input_path, corrupted_path)
                    pbar.set_postfix({"Status": "Corrupted", "Moved to": corrupted_folder})
                except OSError as move_error:
                    pbar.set_postfix({"Status": "Error", "Message": str(move_error)})
            
            pbar.update(1)

def get_new_filename_from_openai(pdf_content):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[
            {"role": "system", "content": "You are a helpful assistant designed to output JSON. Please reply with a filename that consists only of English characters, numbers, and underscores, and is no longer than 50 characters. Do not include characters outside of these, as the system may crash. Do not reply in JSON format, just reply with text."},
            {"role": "user", "content": pdf_content}
        ]
    )
    initial_filename = response.choices[0].message.content
    filename = validate_and_trim_filename(initial_filename)
    return filename

def validate_and_trim_filename(initial_filename):
    allowed_chars = r'[^a-zA-Z0-9_]'
    
    if not initial_filename:
        timestamp = time.strftime('%Y%m%d%H%M%S', time.gmtime())
        return f'empty_file_{timestamp}'
    
    cleaned_filename = re.sub(allowed_chars, '', initial_filename)
    return cleaned_filename[:100] if len(cleaned_filename) > 100 else cleaned_filename

def pdfs_to_text_string(filepath):
    try:
        with open(filepath, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            content = ""
            for page in reader.pages:
                content += page.extract_text() + " "  # Use space instead of newline
            content = content.strip()  # Remove leading/trailing whitespace
            if not content:
                content = "Content is empty or contains only whitespace."
            encoding = tiktoken.get_encoding("cl100k_base")
            num_tokens = len(encoding.encode(content))
            if num_tokens > max_length:
                content = content_token_cut(content, num_tokens, max_length)
            return content
    except (IOError, OSError) as e:
        print(f"Error opening PDF {filepath}: {str(e)}")
        return f"Error opening PDF: {str(e)}"
    except Exception as e:
        print(f"Error reading PDF {filepath}: {str(e)}")
        return f"Error reading PDF: {str(e)}"



def content_token_cut(content, num_tokens, max_length):
    encoding = tiktoken.get_encoding("cl100k_base")
    while num_tokens > max_length:
        content = content[:int(len(content) * 0.9)]  # Cut 10% of the content
        num_tokens = len(encoding.encode(content))
    return content


if __name__ == "__main__":
    input_folder = input("Enter the input folder path: ")
    corrupted_folder = input("Enter the corrupted PDFs folder path: ")
    renamed_folder = input("Enter the renamed PDFs folder path: ")

    sort_and_rename_pdfs(input_folder, corrupted_folder, renamed_folder)
    print("PDF sorting and renaming completed.")
