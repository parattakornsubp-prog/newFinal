---
title: Image API
emoji: 🖼️
colorFrom: blue
colorTo: pink
sdk: docker
pinned: false
---

# Image Classification API

A high-throughput Image Classification API using FastAPI and Vision Transformer (ViT) in ONNX format.

## Features
- **FastAPI**: Modern, fast web framework for building APIs.
- **ONNX Runtime**: Efficient model inference.
- **Multiprocessing**: Uses `ProcessPoolExecutor` for non-blocking inference.
- **Dockerized**: Ready for deployment on Hugging Face Spaces or any Docker-compatible platform.

## Project Structure
```text
.
├── .github/workflows/   # CI/CD for GitHub Actions
├── app/
│   ├── main.py          # API implementation
│   ├── config.py        # Configuration settings
│   ├── schemas.py       # Pydantic models
│   └── labels.py        # ImageNet labels
├── model_final.onnx     # ONNX Model file
├── Dockerfile           # Docker configuration
└── requirements.txt     # Python dependencies
```

## Getting Started

### Prerequisites
- Python 3.11+ (Compatible with 3.14)
- `model_final.onnx` in the root directory.

### Local Installation
1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the API:
   ```bash
   uvicorn app.main:app --reload
   ```

### API Endpoints
- **GET /health**: Check system status and model path.
- **POST /predict**: Upload an image to get classification results.
- **GET /docs**: Swagger UI documentation.

## Deployment

### Hugging Face Spaces
This project is configured to deploy automatically to Hugging Face Spaces via GitHub Actions.

**Setup Secrets in GitHub:**
- `HF_TOKEN`: Your Hugging Face API Token (Write access).
- `HF_USERNAME`: Your Hugging Face username.
- `HF_SPACE_NAME`: The name of your Hugging Face Space.

## License
MIT
