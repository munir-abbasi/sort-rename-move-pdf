
# PDF Sorter and Renamer
This Python script automates the process of sorting and renaming PDF files using OpenAI's GPT model. It organizes large collections of PDFs by generating meaningful filenames based on their content.

## Credits
This script is a modified version of the original work by Brandon-c-tech. The original repository can be found at:
https://github.com/Brandon-c-tech/PDFs-AI-rename
I appreciate the foundation provided by the original author, which has been extended and modified to suit additional requirements.

## Features
- Processes PDF files from a specified input folder
- Extracts text content from PDFs
- Uses OpenAI's GPT-3.5-turbo model to generate relevant filenames
- Handles corrupted or unreadable PDFs by moving them to a separate folder
- Ensures unique filenames to prevent overwriting
- Provides a progress bar for visual feedback during processing

## Requirements
- Python 3.6+
- PyPDF2
- tiktoken
- openai
- tqdm

## Installation
1. Clone this repository:

`git clone https://github.com/yourusername/pdf-sorter-renamer.git cd pdf-sorter-renamer`

2. Install the required packages:

`pip install PyPDF2 tiktoken openai tqdm`

3. Set up your OpenAI API key as an environment variable:

`export OPENAI_API_KEY='your-api-key-here'`

## Usage
Run the script from the command line:
`python sortrenamemovepdf.py`

You will be prompted to enter:
1. The input folder path containing the PDFs to process
2. The folder path for storing corrupted PDFs
3. The folder path for storing renamed PDFs
The script will then process all PDF files in the input folder, renaming them based on their content and moving any corrupted files to the specified folder.

## How it works
1. script performs the following main functions:
2. Sorts and renames PDF files from an input folder.
3. The script takes three folder paths as input:
    An input folder containing the original PDF files
    A folder for corrupted PDFs
    A folder for renamed PDFs
4. For each PDF file in the input folder, the script: 
  a. Checks if the file is a valid PDF using PyPDF2. 
  b. If valid, it extracts the text content from the PDF. 
  c. Sends the extracted content to OpenAI's GPT model to generate a new filename. 
  d. Validates and trims the generated filename to ensure it only contains allowed characters and is not too long. 
  e. Moves the PDF to the renamed folder with the new filename. 
  f. If a file with the same name already exists, it appends a random hex string to make it unique.

If a PDF is corrupted or unreadable, it moves the file to the corrupted folder.
The script uses OpenAI's API to generate meaningful filenames based on the content of the PDFs. It includes functions to handle token limits for the OpenAI API, ensuring that large PDFs are truncated if necessary. The script provides progress feedback using a progress bar (tqdm) to show the status of each file being processed. It includes error handling for various scenarios like file reading errors, API errors, etc. In essence, this script automates the process of organizing and renaming PDF files based on their content, separating valid and corrupted PDFs, and using AI to generate relevant filenames. This can be particularly useful for managing large collections of PDF documents with non-descriptive original filenames.

## Limitations
- The script uses the OpenAI API, which requires an API key and may incur costs.
- Large PDF files or a large number of files may take significant time to process.
- The quality of renamed files depends on the OpenAI model's interpretation of the PDF content.

## Contributing
Contributions, issues, and feature requests are welcome. Feel free to check [issues page](https://github.com/yourusername/pdf-sorter-renamer/issues) if you want to contribute.

## License
This project is licensed under the GNU General Public License v3.0 (GPL-3.0). See the [LICENSE](LICENSE) file for details.

### GNU General Public License v3.0 (GPL-3.0)
This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
