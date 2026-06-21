# Contract Risk Agent

A RAG-based legal contract risk analysis tool with built-in evaluation framework.

## Features

- Contract parsing and chunking
- Risk analysis using LLMs
- Evaluation framework for model performance
- Streamlit interface for easy interaction

## Project Structure

```
contract-risk-agent/
├── src/                  (package source code)
├── data/raw/             (for downloaded datasets)
├── data/processed/       (for chunked/processed data)
├── eval/                 (evaluation scripts and results)
├── eval/results/
├── tests/
├── README.md (this file)
├── requirements.txt
├── .gitignore (standard Python gitignore, plus data/ and .env)
└── pyproject.toml for the package
```

## Installation

1. Clone this repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Data Preparation

### Downloading and Processing CUAD Dataset

The project uses the CUAD dataset for training and testing. To download and process the dataset:

```bash
python -m data_prep.download_cuad
```

You can limit processing to a small number of contracts for faster iteration:
```bash
python -m data_prep.download_cuad --limit 10
```

### Processing Contract Documents

To process contract documents into chunks for analysis:
```bash
python -m data_prep.process_contracts
```

## Usage

To run the application:
```bash
python -m contract_risk_agent
```

## Development

- Add new features to `src/`
- Write tests in `tests/`
- Add evaluation scripts in `eval/`