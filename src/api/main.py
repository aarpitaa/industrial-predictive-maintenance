from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from predict import predict_rul, _load  # noqa: E402
from api.schemas import PredictRequest, PredictResponse  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model + scaler once, at startup — not on the first request.
    _load()
    print("Model and scaler loaded.")
    yield
    # (nothing needed on shutdown)


app = FastAPI(
    title="Turbofan RUL Prediction API",
    description="Predicts Remaining Useful Life (in cycles) for aircraft turbofan engines from recent sensor readings.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    try:
        recent_cycles = [cycle.model_dump() for cycle in request.recent_cycles]
        predicted_rul = predict_rul(recent_cycles)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    return PredictResponse(
        predicted_rul=predicted_rul,
        cycles_used=len(recent_cycles),
    )
