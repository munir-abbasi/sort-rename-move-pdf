# PDF Sort, Rename & Move Utility

A powerful utility for organizing your PDF documents using AI. This tool extracts content from PDFs, uses AI to generate descriptive filenames, and sorts them into appropriate folders.

## Features

- **Multiple AI Provider Support**: Choose between OpenAI, Claude, Gemini, or Deepseek for text analysis
- **Smart Filename Generation**: AI analyzes PDF content to create meaningful filenames
- **Corrupted File Handling**: Automatically detects and segregates corrupted PDFs
- **Progress Tracking**: Resumes processing if interrupted
- **Flexible Command-Line Interface**: Easy to use with command-line arguments or interactive prompts

## Requirements

### Core Dependencies
```
python >= 3.7
PyPDF2 >= 3.0.0
tiktoken >= 0.5.0
tqdm >= 4.66.0
requests >= 2.31.0
```

### Provider-Specific Dependencies
Install only what you need based on your preferred AI provider:

```
# For OpenAI
openai >= 1.0.0

# For Claude
anthropic >= 0.5.0

# For Gemini
google-generativeai >= 0.3.0
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/sort-rename-move-pdf.git
   cd sort-rename-move-pdf
   ```

2. Install the core dependencies:
   ```
   pip install PyPDF2 tiktoken tqdm requests
   ```

3. Install provider-specific dependencies (choose at least one):
   ```
   # For OpenAI
   pip install openai
   
   # For Claude
   pip install anthropic
   
   # For Gemini
   pip install google-generativeai
   ```

## API Keys

You'll need an API key for at least one of the supported AI providers:

- **OpenAI**: Set as environment variable `OPENAI_API_KEY` or provide with `--api-key`
- **Claude**: Set as environment variable `CLAUDE_API_KEY` or provide with `--api-key`
- **Gemini**: Set as environment variable `GEMINI_API_KEY` or provide with `--api-key`
- **Deepseek**: Set as environment variable `DEEPSEEK_API_KEY` or provide with `--api-key`

## Usage

### Basic Usage

```bash
python sortrenamemovepdf.py -i /path/to/pdfs -c /path/to/corrupted -r /path/to/renamed
```

### Using Different AI Providers

```bash
# Use OpenAI's GPT-4o
python sortrenamemovepdf.py -i ./input -c ./corrupted -r ./renamed -p openai -m gpt-4o

# Use Claude
python sortrenamemovepdf.py -i ./input -c ./corrupted -r ./renamed -p claude -m claude-3-sonnet

# Use Gemini
python sortrenamemovepdf.py -i ./input -c ./corrupted -r ./renamed -p gemini
```

### List Available Models

```bash
python sortrenamemovepdf.py --list-models
```

### Command-Line Arguments

```
-i, --input       Input folder containing PDF files
-c, --corrupted   Folder for corrupted PDF files
-r, --renamed     Folder for renamed PDF files
-p, --provider    AI provider (openai, claude, gemini, deepseek)
-m, --model       Model to use for the selected provider
-k, --api-key     API key for the selected provider
-l, --list-models List all available models by provider and exit
```

## How It Works

1. The script scans the input folder for PDF files
2. For each PDF:
   - Extracts text content
   - Sends the text to the selected AI provider
   - Generates a descriptive filename
   - Handles any duplicates with sequential numbering
   - Moves the file to the renamed folder
3. Corrupted PDFs are moved to the corrupted folder
4. Progress is tracked to allow resuming if interrupted

## Examples

### Organizing Research Papers

```bash
python sortrenamemovepdf.py -i ./research_papers -c ./corrupted_papers -r ./organized_papers
```

### Processing Legal Documents with Claude

```bash
python sortrenamemovepdf.py -i ./legal_docs -c ./damaged_docs -r ./processed_docs -p claude -m claude-3-opus
```

## Troubleshooting

- **Missing Dependencies**: Ensure you've installed the required packages for your chosen provider
- **API Key Issues**: Check that your API key is valid and has been set correctly
- **Processing Errors**: Corrupted PDFs will be automatically moved to the corrupted folder
- **Rate Limiting**: The script includes exponential backoff for API retries

## License

MIT License
