# Deploying to Hugging Face Spaces

This project can be deployed as a Gradio application on Hugging Face Spaces.

## Recommended Space Type

- **Gradio**

## Python Version

- Python 3.8+

## Expected Startup Command

```bash
python app.py
```

## Required Files

- `app.py` - Main Gradio application
- `requirements.txt` - Dependencies
- `data/processed/cuad_chunks.parquet` - Contract chunks data
- `data/processed/cuad_embeddings.npy` - Embeddings data  
- `data/processed/cuad_embedding_metadata.parquet` - Embedding metadata

## How to Deploy

1. Create a new Space on Hugging Face with the Gradio template
2. Upload all required files to your space repository
3. Ensure `requirements.txt` includes all dependencies
4. Set the startup command to `python app.py`
5. Push to deploy

## Notes

- The application requires the data files to be present in the `data/processed/` directory
- Make sure all dependencies listed in `requirements.txt` are included
- The Gradio interface will be automatically served at the space URL