"""
Med-AGI Imaging Service
Handles medical imaging inference with Triton GPU acceleration
"""

import os
import logging
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, File, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import numpy as np
from PIL import Image
import io

from triton_client import TritonClient
from auth import verify_jwt, get_current_user
from opa_client import check_permission
from metrics import setup_metrics, track_inference

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Triton client
triton_client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global triton_client
    triton_url = os.getenv("TRITON_URL", "localhost:8000")
    triton_client = TritonClient(triton_url)
    logger.info(f"Connected to Triton at {triton_url}")
    yield
    if triton_client:
        triton_client.close()


# Create FastAPI app
app = FastAPI(
    title="Med-AGI Imaging Service",
    description="Medical imaging inference with GPU acceleration",
    version="1.0.0",
    lifespan=lifespan
)

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup metrics
setup_metrics(app)


# Request/Response models
class HealthResponse(BaseModel):
    available: bool
    healthy: bool
    triton_status: Optional[str] = None
    gpu_available: bool = False


class InferenceRequest(BaseModel):
    image_id: str = Field(..., description="Unique identifier for the image")
    modality: str = Field(..., description="Imaging modality (CXR, CT, MRI)")
    model_name: Optional[str] = Field(None, description="Specific model to use")
    return_uncertainty: bool = Field(False, description="Return uncertainty estimates")


class InferenceResponse(BaseModel):
    image_id: str
    predictions: Dict[str, float]
    uncertainty: Optional[float] = None
    model_name: str
    model_version: str
    inference_time_ms: float
    backend: str


class DicomMetadata(BaseModel):
    patient_id: str
    study_uid: str
    series_uid: str
    modality: str
    study_date: Optional[str] = None
    institution: Optional[str] = None


# Health endpoints
@app.get("/v1/triton/health", response_model=HealthResponse)
async def triton_health():
    """Check Triton server health status"""
    try:
        if not triton_client:
            return HealthResponse(
                available=False,
                healthy=False,
                triton_status="Not connected"
            )
        
        is_healthy = triton_client.is_healthy()
        gpu_available = triton_client.has_gpu()
        
        return HealthResponse(
            available=True,
            healthy=is_healthy,
            triton_status="Connected" if is_healthy else "Unhealthy",
            gpu_available=gpu_available
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            available=False,
            healthy=False,
            triton_status=str(e)
        )


@app.get("/health")
async def health():
    """Basic health check"""
    return {"status": "healthy"}


# Inference endpoints
@app.post("/v1/imaging/infer", response_model=InferenceResponse)
@track_inference
async def infer_image(
    file: UploadFile = File(...),
    modality: str = "CXR",
    model_name: Optional[str] = None,
    return_uncertainty: bool = False,
    current_user: Dict = Depends(get_current_user)
):
    """
    Perform inference on medical image
    
    Supports CXR, CT, and MRI modalities
    """
    # Check permissions
    if not await check_permission(current_user, "imaging:infer", {"modality": modality}):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions for imaging inference"
        )
    
    try:
        # Read and preprocess image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize for model input
        image = image.resize((224, 224))
        
        # Convert to numpy array and normalize
        img_array = np.array(image).astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        
        # Select model based on modality
        if not model_name:
            model_map = {
                "CXR": "densenet121_cxr",
                "CT": "resnet50_ct",
                "MRI": "efficientnet_mri"
            }
            model_name = model_map.get(modality, "densenet121_cxr")
        
        # Run inference
        result = triton_client.infer(
            model_name=model_name,
            inputs={"input": img_array},
            outputs=["predictions", "uncertainty"] if return_uncertainty else ["predictions"]
        )
        
        # Parse results
        predictions = result["predictions"][0]
        
        # Map predictions to labels
        labels = get_labels_for_modality(modality)
        pred_dict = {label: float(pred) for label, pred in zip(labels, predictions)}
        
        # Get uncertainty if requested
        uncertainty = None
        if return_uncertainty and "uncertainty" in result:
            uncertainty = float(result["uncertainty"][0])
        
        return InferenceResponse(
            image_id=file.filename,
            predictions=pred_dict,
            uncertainty=uncertainty,
            model_name=model_name,
            model_version="1.0",
            inference_time_ms=result.get("inference_time", 0),
            backend="GPU" if triton_client.has_gpu() else "CPU"
        )
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {str(e)}"
        )


@app.post("/v1/imaging/batch_infer")
async def batch_infer(
    files: List[UploadFile] = File(...),
    modality: str = "CXR",
    current_user: Dict = Depends(get_current_user)
):
    """Batch inference on multiple images"""
    # Check permissions
    if not await check_permission(current_user, "imaging:batch_infer", {"modality": modality}):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions for batch inference"
        )
    
    results = []
    for file in files:
        try:
            # Process each image
            result = await infer_image(
                file=file,
                modality=modality,
                current_user=current_user
            )
            results.append(result)
        except Exception as e:
            results.append({
                "image_id": file.filename,
                "error": str(e)
            })
    
    return {"results": results, "total": len(files)}


# DICOM endpoints
@app.post("/v1/dicom/retrieve")
async def retrieve_dicom(
    study_uid: str,
    series_uid: Optional[str] = None,
    current_user: Dict = Depends(get_current_user)
):
    """Retrieve DICOM images from DICOMweb server"""
    # Check permissions
    if not await check_permission(current_user, "dicom:retrieve", {}):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions for DICOM retrieval"
        )
    
    dicomweb_base = os.getenv("DICOMWEB_BASE")
    if not dicomweb_base:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="DICOMweb service not configured"
        )
    
    # Implementation would connect to actual DICOMweb server
    # This is a placeholder response
    return {
        "study_uid": study_uid,
        "series_uid": series_uid,
        "status": "retrieved",
        "images": []
    }


@app.get("/v1/dicom/metadata/{study_uid}", response_model=DicomMetadata)
async def get_dicom_metadata(
    study_uid: str,
    current_user: Dict = Depends(get_current_user)
):
    """Get DICOM metadata for a study"""
    # Check permissions
    if not await check_permission(current_user, "dicom:metadata", {}):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions for DICOM metadata"
        )
    
    # This would fetch actual metadata from DICOM server
    # Placeholder implementation
    return DicomMetadata(
        patient_id="PATIENT001",
        study_uid=study_uid,
        series_uid="1.2.3.4.5",
        modality="CXR",
        study_date="2024-01-15",
        institution="Medical Center"
    )


# Model management endpoints
@app.get("/v1/models")
async def list_models(current_user: Dict = Depends(get_current_user)):
    """List available models in Triton"""
    try:
        models = triton_client.list_models()
        return {"models": models}
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        return {"models": [], "error": str(e)}


@app.get("/v1/models/{model_name}")
async def get_model_info(
    model_name: str,
    current_user: Dict = Depends(get_current_user)
):
    """Get detailed information about a specific model"""
    try:
        info = triton_client.get_model_info(model_name)
        return info
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_name} not found: {str(e)}"
        )


# Utility functions
def get_labels_for_modality(modality: str) -> List[str]:
    """Get prediction labels based on imaging modality"""
    labels_map = {
        "CXR": [
            "No Finding", "Cardiomegaly", "Edema", "Consolidation",
            "Atelectasis", "Pleural Effusion", "Pneumonia", "Pneumothorax"
        ],
        "CT": [
            "Normal", "Lung Nodule", "Mass", "Pneumonia",
            "Interstitial Disease", "Emphysema", "Pleural Effusion"
        ],
        "MRI": [
            "Normal", "Tumor", "Edema", "Hemorrhage",
            "Infarction", "White Matter Disease"
        ]
    }
    return labels_map.get(modality, ["Class_0", "Class_1", "Class_2"])


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8006)