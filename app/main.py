import asyncio
import io
import logging
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import asynccontextmanager
from functools import partial

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from app.config import settings
from app.schemas import PredictionResponse, HealthResponse
from app.labels import IMAGENET_LABELS

from fastapi.responses import RedirectResponse

@app.get("/")
async def root():
    return RedirectResponse(url="/docs")


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global process pool for CPU-bound inference
process_pool: ProcessPoolExecutor = None


def run_inference(image_bytes: bytes, model_path: str) -> tuple[str, float, list]:
    """
    Run model inference in a separate process (CPU-bound task).
    Returns (label, confidence, top5_results)
    """
    import onnxruntime as ort
    import numpy as np
    from PIL import Image
    import io

    # Load image
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Preprocess: resize to 224x224, normalize with ImageNet stats
    image = image.resize((224, 224), Image.BILINEAR)
    img_array = np.array(image, dtype=np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_array = (img_array - mean) / std

    # HWC -> NCHW
    img_array = np.transpose(img_array, (2, 0, 1))
    img_array = np.expand_dims(img_array, axis=0)

    # Run ONNX inference
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 2
    session = ort.InferenceSession(
        model_path,
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: img_array})
    logits = outputs[0][0]  # shape: (num_classes,)

    # Softmax
    logits = logits - np.max(logits)
    exp_logits = np.exp(logits)
    probs = exp_logits / np.sum(exp_logits)

    top5_idx = np.argsort(probs)[::-1][:5]
    top5 = [{"label": IMAGENET_LABELS[i], "confidence": float(probs[i])} for i in top5_idx]

    return top5[0]["label"], float(probs[top5_idx[0]]), top5


@asynccontextmanager
async def lifespan(app: FastAPI):
    global process_pool
    logger.info("Starting process pool executor...")
    process_pool = ProcessPoolExecutor(max_workers=settings.MAX_WORKERS)
    logger.info(f"Process pool started with {settings.MAX_WORKERS} workers")
    yield
    logger.info("Shutting down process pool...")
    process_pool.shutdown(wait=True)


app = FastAPI(
    title="Image Classification API",
    description="High-Throughput Image Classification using ViT ONNX model",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="ok", model=settings.MODEL_PATH)


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    # --- Validate content type ---
    if file.content_type not in settings.ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{file.content_type}'. Allowed: {settings.ALLOWED_CONTENT_TYPES}",
        )

    # --- Read and size-check ---
    image_bytes = await file.read()
    if len(image_bytes) > settings.MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({len(image_bytes)} bytes). Max allowed: {settings.MAX_FILE_SIZE_BYTES} bytes.",
        )
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # --- Validate it's a real image ---
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.verify()  # raises if corrupted
    except Exception:
        raise HTTPException(status_code=400, detail="File is corrupted or not a valid image.")

    # --- Run inference in process pool (non-blocking) ---
    loop = asyncio.get_event_loop()
    start = time.perf_counter()
    try:
        label, confidence, top5 = await loop.run_in_executor(
            process_pool,
            partial(run_inference, image_bytes, settings.MODEL_PATH),
        )
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail="Model inference failed.")
    latency_ms = (time.perf_counter() - start) * 1000

    return PredictionResponse(
        label=label,
        confidence=round(confidence, 4),
        top5=top5,
        latency_ms=round(latency_ms, 2),
    )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})
