# PDF Sort, Rename & Move Utility

A powerful utility for organizing PDF documents using AI. This tool extracts content from PDFs, uses AI to generate descriptive filenames based on the content, and sorts them into appropriate folders.

## Features

- **Multiple AI Provider Support**: Choose between OpenAI, Claude, Gemini, or Deepseek for PDF content analysis
- **Smart Filename Generation**: AI analyzes PDF content to create meaningful filenames
- **Corrupted File Handling**: Automatically detects and segregates corrupted or password-protected PDFs
- **Progress Tracking**: Resumes processing if interrupted
- **Cross-Platform Compatible**: Works on Windows, macOS, and Linux

## Requirements

### Core Dependencies
```bash
python >= 3.7
PyPDF2 >= 3.0.0
tiktoken >= 0.5.0
tqdm >= 4.66.0
requests >= 2.31.0
```

### Provider-Specific Dependencies
Install only what you need based on your preferred AI provider:

```bash
# For OpenAI (default)
pip install openai

# For Claude
pip install anthropic

# For Gemini
pip install google-generativeai
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/sort-rename-move-pdf.git
   cd sort-rename-move-pdf
   ```

2. Install the core dependencies:
   ```bash
   pip install PyPDF2 tiktoken tqdm requests
   ```

3. Install provider-specific dependencies (choose at least one):
   ```bash
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
   - Extracts text content with intelligent page limiting for large files
   - Sends the content to the selected AI provider
   - Generates a descriptive filename
   - Handles duplicates with sequential numbering
   - Moves the file to the renamed folder
3. Corrupted or password-protected PDFs are moved to the corrupted folder
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

### Processing Large Batches of Files

For large batches, the tool automatically tracks progress and can be safely interrupted and resumed:

```bash
python sortrenamemovepdf.py -i ./large_batch -c ./corrupted -r ./processed
```

## Troubleshooting

- **Missing Dependencies**: Ensure you've installed the required packages for your chosen provider
- **API Key Issues**: Check that your API key is valid and has been set correctly
- **Processing Errors**: If you see specific errors, check the `pdf_processing_errors.log` file for details
- **Rate Limiting**: The script includes exponential backoff for API retries

## Performance Considerations

- Very large PDFs are automatically limited to the first 100 pages for performance
- For batch processing, the script uses efficient token counting and file handling
- Progress is saved after each file, so it's safe to interrupt and resume later

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
