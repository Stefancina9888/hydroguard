"""FastAPI academic demonstrator: one local model load per application lifespan."""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from hydroguard.data import DataValidationError
from pydantic import BaseModel, ConfigDict, Field

from hydroguard_serving.runtime import Runtime
from hydroguard_serving.validation import PredictionRequest


class PredictionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    predicted_class: Literal[0, 1, 2] = Field(alias="class")
    label: str
    classes: Annotated[list[int], Field(min_length=3, max_length=3)]
    probabilities: Annotated[list[float], Field(min_length=3, max_length=3)]
    model_version: str


def create_app(model_directory: Path | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        directory = model_directory or Path(
            os.environ.get("HYDROGUARD_MODEL_DIR", "artifacts/models/phase3b-v1")
        )
        # A missing/corrupt/incompatible bundle fails startup; never report ready.
        application.state.runtime = Runtime(directory)
        try:
            yield
        finally:
            application.state.runtime = None

    application = FastAPI(
        title="HydroGuard academic demonstrator",
        version="1.0.0",
        description=(
            "Fixed Phase 3B model. Final-test macro-F1=0.675047; weak-leakage recall=18/103. "
            "Not validated for industrial operation. Probabilities are uncalibrated."
        ),
        lifespan=lifespan,
    )

    @application.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, error: RequestValidationError):
        errors = error.errors()
        # Never echo Pydantic's input/context (which can contain the entire sensor array).
        details = [
            {
                "loc": [part if isinstance(part, int) else str(part)[:80] for part in item["loc"]],
                "msg": item["msg"][:200],
                "type": item["type"],
            }
            for item in errors[:8]
        ]
        return JSONResponse(
            status_code=422, content={"detail": details, "error_count": len(errors)}
        )

    def runtime() -> Runtime:
        loaded = getattr(application.state, "runtime", None)
        if loaded is None:
            raise HTTPException(status_code=503, detail="Model is not ready")
        return loaded

    @application.get("/health")
    def health():
        return {"ready": True, "model_version": runtime().version}

    @application.post("/predict", response_model=PredictionResponse)
    def predict(request: PredictionRequest):
        try:
            return runtime().predict(request.model_dump())
        except DataValidationError as error:
            # Finite raw samples may still overflow statistics; do not echo values.
            raise HTTPException(
                status_code=422, detail="Sensor values cannot produce finite model features"
            ) from error

    return application


app = create_app()
