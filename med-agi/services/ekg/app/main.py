"""
Med-AGI EKG Service
Processes EKG waveform data with uncertainty quantification
"""

import os
import logging
import time
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import numpy as np
from scipy import signal
from scipy.stats import entropy

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
    logger.info(f"EKG Service connected to Triton at {triton_url}")
    yield
    if triton_client:
        triton_client.close()


# Create FastAPI app
app = FastAPI(
    title="Med-AGI EKG Service",
    description="EKG waveform analysis with uncertainty quantification",
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
class EKGInferenceRequest(BaseModel):
    samples: List[float] = Field(..., min_items=1000, max_items=10000, 
                                 description="EKG waveform samples")
    sampling_rate: int = Field(500, description="Sampling rate in Hz")
    lead: str = Field("II", description="EKG lead (I, II, III, aVR, aVL, aVF, V1-V6)")
    patient_age: Optional[int] = Field(None, ge=0, le=150, description="Patient age")
    patient_sex: Optional[str] = Field(None, description="Patient sex (M/F)")
    
    @validator('samples')
    def validate_samples(cls, v):
        if len(v) < 1000:
            raise ValueError("Minimum 1000 samples required")
        return v
    
    @validator('lead')
    def validate_lead(cls, v):
        valid_leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF'] + [f'V{i}' for i in range(1, 7)]
        if v not in valid_leads:
            raise ValueError(f"Invalid lead. Must be one of {valid_leads}")
        return v


class EKGInferenceResponse(BaseModel):
    probs: Dict[str, float] = Field(..., description="Condition probabilities")
    uncertainty: float = Field(..., description="Prediction uncertainty (0-1)")
    rhythm: str = Field(..., description="Detected rhythm")
    heart_rate: int = Field(..., description="Calculated heart rate (bpm)")
    pr_interval: Optional[float] = Field(None, description="PR interval in ms")
    qrs_duration: Optional[float] = Field(None, description="QRS duration in ms")
    qt_interval: Optional[float] = Field(None, description="QT interval in ms")
    qtc: Optional[float] = Field(None, description="Corrected QT interval")
    features: Dict[str, Any] = Field(default_factory=dict, description="Extracted features")
    backend: str = Field(..., description="Inference backend (GPU/CPU)")
    inference_time_ms: float = Field(..., description="Inference time in milliseconds")


class HealthResponse(BaseModel):
    available: bool
    healthy: bool
    triton_status: Optional[str] = None
    models_loaded: List[str] = Field(default_factory=list)


class BatchEKGRequest(BaseModel):
    recordings: List[EKGInferenceRequest]
    batch_id: str = Field(..., description="Batch identifier")


# Health endpoints
@app.get("/v1/ekg/health", response_model=HealthResponse)
async def health_check():
    """Check EKG service health"""
    try:
        if not triton_client:
            return HealthResponse(
                available=False,
                healthy=False,
                triton_status="Not connected"
            )
        
        is_healthy = triton_client.is_healthy()
        models = triton_client.list_models()
        
        return HealthResponse(
            available=True,
            healthy=is_healthy,
            triton_status="Connected" if is_healthy else "Unhealthy",
            models_loaded=models
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            available=False,
            healthy=False,
            triton_status=str(e)
        )


@app.get("/health")
async def basic_health():
    """Basic health check"""
    return {"status": "healthy", "service": "ekg"}


# EKG Analysis endpoints
@app.post("/v1/ekg/infer", response_model=EKGInferenceResponse)
@track_inference
async def analyze_ekg(
    request: EKGInferenceRequest,
    current_user: Dict = Depends(get_current_user)
):
    """
    Analyze EKG waveform and detect cardiac conditions
    
    Detects:
    - Atrial Fibrillation (AFib)
    - Atrial Flutter
    - Bradycardia
    - Tachycardia
    - Premature Ventricular Contractions (PVCs)
    - ST Elevation
    - ST Depression
    - Bundle Branch Blocks
    """
    # Check permissions
    if not await check_permission(current_user, "ekg:infer", {"lead": request.lead}):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions for EKG analysis"
        )
    
    start_time = time.time()
    
    try:
        # Extract features from waveform
        features = extract_ekg_features(
            request.samples,
            request.sampling_rate,
            request.lead
        )
        
        # Prepare input for model
        input_array = preprocess_ekg(request.samples, request.sampling_rate)
        
        # Run inference
        if triton_client and triton_client.is_healthy():
            # Use Triton for GPU inference
            result = triton_client.infer(
                model_name="ekg_resnet",
                inputs={"waveform": input_array},
                outputs=["predictions", "features"]
            )
            predictions = result["predictions"][0]
            backend = "GPU" if triton_client.has_gpu() else "CPU"
        else:
            # Fallback to CPU inference
            predictions = cpu_inference(input_array, features)
            backend = "CPU"
        
        # Calculate uncertainty
        uncertainty = calculate_uncertainty(predictions)
        
        # Map predictions to conditions
        conditions = [
            "Normal", "AFib", "AFlutter", "Bradycardia", "Tachycardia",
            "PVC", "PAC", "LBBB", "RBBB", "ST_Elevation", "ST_Depression",
            "T_Wave_Abnormal", "Long_QT", "Short_PR"
        ]
        
        probs = {cond: float(prob) for cond, prob in zip(conditions, predictions)}
        
        # Determine primary rhythm
        rhythm = determine_rhythm(probs, features)
        
        # Calculate intervals
        intervals = calculate_intervals(request.samples, request.sampling_rate)
        
        inference_time = (time.time() - start_time) * 1000
        
        return EKGInferenceResponse(
            probs=probs,
            uncertainty=uncertainty,
            rhythm=rhythm,
            heart_rate=features["heart_rate"],
            pr_interval=intervals.get("pr_interval"),
            qrs_duration=intervals.get("qrs_duration"),
            qt_interval=intervals.get("qt_interval"),
            qtc=intervals.get("qtc"),
            features=features,
            backend=backend,
            inference_time_ms=inference_time
        )
        
    except Exception as e:
        logger.error(f"EKG analysis failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"EKG analysis failed: {str(e)}"
        )


@app.post("/v1/ekg/batch")
async def batch_analyze(
    request: BatchEKGRequest,
    current_user: Dict = Depends(get_current_user)
):
    """Batch analysis of multiple EKG recordings"""
    # Check permissions
    if not await check_permission(current_user, "ekg:batch", {}):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions for batch EKG analysis"
        )
    
    results = []
    for recording in request.recordings:
        try:
            result = await analyze_ekg(recording, current_user)
            results.append({
                "status": "success",
                "result": result.dict()
            })
        except Exception as e:
            results.append({
                "status": "error",
                "error": str(e)
            })
    
    return {
        "batch_id": request.batch_id,
        "total": len(request.recordings),
        "successful": sum(1 for r in results if r["status"] == "success"),
        "results": results
    }


@app.post("/v1/ekg/validate")
async def validate_waveform(
    samples: List[float],
    sampling_rate: int = 500,
    current_user: Dict = Depends(get_current_user)
):
    """Validate EKG waveform quality"""
    try:
        # Check signal quality
        quality_metrics = check_signal_quality(samples, sampling_rate)
        
        return {
            "valid": quality_metrics["is_valid"],
            "quality_score": quality_metrics["quality_score"],
            "issues": quality_metrics["issues"],
            "recommendations": quality_metrics["recommendations"]
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Validation failed: {str(e)}"
        )


@app.get("/v1/ekg/features")
async def get_feature_definitions(current_user: Dict = Depends(get_current_user)):
    """Get definitions of extracted EKG features"""
    return {
        "heart_rate": "Heart rate in beats per minute",
        "hrv_sdnn": "Heart rate variability - standard deviation of NN intervals",
        "hrv_rmssd": "Root mean square of successive differences",
        "pnn50": "Percentage of successive RR intervals that differ by more than 50ms",
        "qrs_amplitude": "Average QRS complex amplitude",
        "t_wave_amplitude": "Average T wave amplitude",
        "p_wave_present": "Whether P waves are detected",
        "rhythm_regularity": "Regularity score of rhythm (0-1)",
        "noise_level": "Estimated noise level in signal"
    }


# Utility functions
def extract_ekg_features(samples: List[float], sampling_rate: int, lead: str) -> Dict[str, Any]:
    """Extract features from EKG waveform"""
    samples_array = np.array(samples)
    
    # Detect R peaks
    r_peaks = detect_r_peaks(samples_array, sampling_rate)
    
    # Calculate heart rate
    if len(r_peaks) > 1:
        rr_intervals = np.diff(r_peaks) / sampling_rate
        heart_rate = int(60 / np.mean(rr_intervals))
    else:
        heart_rate = 0
    
    # Calculate HRV metrics
    hrv_metrics = calculate_hrv(r_peaks, sampling_rate)
    
    # Detect other waves
    p_waves = detect_p_waves(samples_array, sampling_rate, r_peaks)
    t_waves = detect_t_waves(samples_array, sampling_rate, r_peaks)
    
    # Calculate amplitudes
    qrs_amplitude = calculate_qrs_amplitude(samples_array, r_peaks)
    t_wave_amplitude = calculate_t_wave_amplitude(samples_array, t_waves) if len(t_waves) > 0 else 0
    
    # Assess rhythm regularity
    rhythm_regularity = assess_rhythm_regularity(r_peaks)
    
    # Estimate noise level
    noise_level = estimate_noise_level(samples_array, sampling_rate)
    
    return {
        "heart_rate": heart_rate,
        "hrv_sdnn": hrv_metrics.get("sdnn", 0),
        "hrv_rmssd": hrv_metrics.get("rmssd", 0),
        "pnn50": hrv_metrics.get("pnn50", 0),
        "r_peaks_count": len(r_peaks),
        "p_wave_present": len(p_waves) > 0,
        "qrs_amplitude": float(qrs_amplitude),
        "t_wave_amplitude": float(t_wave_amplitude),
        "rhythm_regularity": float(rhythm_regularity),
        "noise_level": float(noise_level),
        "lead": lead
    }


def detect_r_peaks(signal_data: np.ndarray, sampling_rate: int) -> np.ndarray:
    """Detect R peaks in EKG signal"""
    # Simple peak detection - in production, use more sophisticated algorithm
    from scipy.signal import find_peaks
    
    # Bandpass filter
    nyquist = sampling_rate / 2
    low = 5 / nyquist
    high = 15 / nyquist
    
    if high < 1:
        b, a = signal.butter(2, [low, high], btype='band')
        filtered = signal.filtfilt(b, a, signal_data)
    else:
        filtered = signal_data
    
    # Square to enhance R peaks
    squared = filtered ** 2
    
    # Find peaks
    peaks, _ = find_peaks(squared, distance=sampling_rate*0.4, height=np.percentile(squared, 75))
    
    return peaks


def detect_p_waves(signal_data: np.ndarray, sampling_rate: int, r_peaks: np.ndarray) -> np.ndarray:
    """Detect P waves before R peaks"""
    p_waves = []
    
    for r_peak in r_peaks:
        # Look for P wave in window before R peak (50-200ms)
        start = max(0, r_peak - int(0.2 * sampling_rate))
        end = max(0, r_peak - int(0.05 * sampling_rate))
        
        if start < end and end <= len(signal_data):
            window = signal_data[start:end]
            if len(window) > 0:
                p_peak = start + np.argmax(window)
                p_waves.append(p_peak)
    
    return np.array(p_waves)


def detect_t_waves(signal_data: np.ndarray, sampling_rate: int, r_peaks: np.ndarray) -> np.ndarray:
    """Detect T waves after R peaks"""
    t_waves = []
    
    for i, r_peak in enumerate(r_peaks[:-1]):
        # Look for T wave in window after R peak (100-400ms)
        start = min(len(signal_data)-1, r_peak + int(0.1 * sampling_rate))
        end = min(len(signal_data), r_peak + int(0.4 * sampling_rate))
        
        # Don't overlap with next R peak
        if i < len(r_peaks) - 1:
            end = min(end, r_peaks[i+1] - int(0.05 * sampling_rate))
        
        if start < end and end <= len(signal_data):
            window = signal_data[start:end]
            if len(window) > 0:
                t_peak = start + np.argmax(np.abs(window))
                t_waves.append(t_peak)
    
    return np.array(t_waves)


def calculate_hrv(r_peaks: np.ndarray, sampling_rate: int) -> Dict[str, float]:
    """Calculate heart rate variability metrics"""
    if len(r_peaks) < 2:
        return {"sdnn": 0, "rmssd": 0, "pnn50": 0}
    
    # RR intervals in milliseconds
    rr_intervals = np.diff(r_peaks) * 1000 / sampling_rate
    
    # SDNN - standard deviation of NN intervals
    sdnn = np.std(rr_intervals)
    
    # RMSSD - root mean square of successive differences
    rr_diff = np.diff(rr_intervals)
    rmssd = np.sqrt(np.mean(rr_diff ** 2)) if len(rr_diff) > 0 else 0
    
    # pNN50 - percentage of successive RR intervals that differ by more than 50ms
    if len(rr_diff) > 0:
        pnn50 = (np.sum(np.abs(rr_diff) > 50) / len(rr_diff)) * 100
    else:
        pnn50 = 0
    
    return {
        "sdnn": float(sdnn),
        "rmssd": float(rmssd),
        "pnn50": float(pnn50)
    }


def calculate_qrs_amplitude(signal_data: np.ndarray, r_peaks: np.ndarray) -> float:
    """Calculate average QRS complex amplitude"""
    if len(r_peaks) == 0:
        return 0.0
    
    amplitudes = []
    for r_peak in r_peaks:
        # Get QRS window around R peak
        start = max(0, r_peak - 20)
        end = min(len(signal_data), r_peak + 20)
        
        if start < end:
            qrs_window = signal_data[start:end]
            amplitude = np.max(qrs_window) - np.min(qrs_window)
            amplitudes.append(amplitude)
    
    return np.mean(amplitudes) if amplitudes else 0.0


def calculate_t_wave_amplitude(signal_data: np.ndarray, t_waves: np.ndarray) -> float:
    """Calculate average T wave amplitude"""
    if len(t_waves) == 0:
        return 0.0
    
    amplitudes = []
    for t_peak in t_waves:
        # Get T wave window
        start = max(0, t_peak - 30)
        end = min(len(signal_data), t_peak + 30)
        
        if start < end:
            t_window = signal_data[start:end]
            amplitude = np.max(np.abs(t_window))
            amplitudes.append(amplitude)
    
    return np.mean(amplitudes) if amplitudes else 0.0


def assess_rhythm_regularity(r_peaks: np.ndarray) -> float:
    """Assess rhythm regularity (0=irregular, 1=regular)"""
    if len(r_peaks) < 3:
        return 0.0
    
    rr_intervals = np.diff(r_peaks)
    if len(rr_intervals) < 2:
        return 0.0
    
    # Calculate coefficient of variation
    cv = np.std(rr_intervals) / np.mean(rr_intervals) if np.mean(rr_intervals) > 0 else 1
    
    # Convert to regularity score (inverse of CV, bounded 0-1)
    regularity = max(0, min(1, 1 - cv))
    
    return regularity


def estimate_noise_level(signal_data: np.ndarray, sampling_rate: int) -> float:
    """Estimate noise level in signal"""
    # High-pass filter to isolate noise
    nyquist = sampling_rate / 2
    high_cutoff = 40 / nyquist
    
    if high_cutoff < 1:
        b, a = signal.butter(2, high_cutoff, btype='high')
        noise = signal.filtfilt(b, a, signal_data)
        noise_level = np.std(noise)
    else:
        noise_level = 0
    
    return noise_level


def preprocess_ekg(samples: List[float], sampling_rate: int) -> np.ndarray:
    """Preprocess EKG signal for model input"""
    signal_data = np.array(samples, dtype=np.float32)
    
    # Remove baseline wander
    signal_data = remove_baseline_wander(signal_data, sampling_rate)
    
    # Normalize
    signal_data = (signal_data - np.mean(signal_data)) / (np.std(signal_data) + 1e-8)
    
    # Resample to standard rate if needed (e.g., 250 Hz)
    target_rate = 250
    if sampling_rate != target_rate:
        num_samples = int(len(signal_data) * target_rate / sampling_rate)
        signal_data = signal.resample(signal_data, num_samples)
    
    # Pad or truncate to fixed length
    target_length = 2500  # 10 seconds at 250 Hz
    if len(signal_data) < target_length:
        signal_data = np.pad(signal_data, (0, target_length - len(signal_data)), mode='constant')
    else:
        signal_data = signal_data[:target_length]
    
    # Add batch and channel dimensions
    signal_data = signal_data.reshape(1, 1, -1)
    
    return signal_data


def remove_baseline_wander(signal_data: np.ndarray, sampling_rate: int) -> np.ndarray:
    """Remove baseline wander using high-pass filter"""
    nyquist = sampling_rate / 2
    low_cutoff = 0.5 / nyquist
    
    if low_cutoff < 1:
        b, a = signal.butter(2, low_cutoff, btype='high')
        filtered = signal.filtfilt(b, a, signal_data)
        return filtered
    return signal_data


def cpu_inference(input_array: np.ndarray, features: Dict[str, Any]) -> np.ndarray:
    """CPU-based inference fallback"""
    # Simple rule-based classification based on features
    predictions = np.zeros(14)  # 14 conditions
    
    # Normal
    if features["heart_rate"] >= 60 and features["heart_rate"] <= 100:
        predictions[0] = 0.7
    
    # AFib - irregular rhythm with absent P waves
    if features["rhythm_regularity"] < 0.3 and not features["p_wave_present"]:
        predictions[1] = 0.8
    
    # Bradycardia
    if features["heart_rate"] < 60:
        predictions[3] = 0.9
    
    # Tachycardia
    if features["heart_rate"] > 100:
        predictions[4] = 0.9
    
    # Add some randomness for other conditions
    predictions += np.random.random(14) * 0.1
    
    # Normalize to probabilities
    predictions = predictions / predictions.sum()
    
    return predictions


def calculate_uncertainty(predictions: np.ndarray) -> float:
    """Calculate prediction uncertainty using entropy"""
    # Clip to avoid log(0)
    predictions = np.clip(predictions, 1e-10, 1.0)
    
    # Calculate entropy
    ent = entropy(predictions)
    
    # Normalize to 0-1 range
    max_entropy = np.log(len(predictions))
    uncertainty = ent / max_entropy if max_entropy > 0 else 0
    
    return float(uncertainty)


def determine_rhythm(probs: Dict[str, float], features: Dict[str, Any]) -> str:
    """Determine primary rhythm from probabilities and features"""
    # Get condition with highest probability
    primary_condition = max(probs, key=probs.get)
    
    # Override with feature-based rules
    if features["heart_rate"] < 60:
        return "Bradycardia"
    elif features["heart_rate"] > 100:
        return "Tachycardia"
    elif probs.get("AFib", 0) > 0.5:
        return "Atrial Fibrillation"
    elif probs.get("Normal", 0) > 0.7:
        return "Normal Sinus Rhythm"
    else:
        return primary_condition


def calculate_intervals(samples: List[float], sampling_rate: int) -> Dict[str, float]:
    """Calculate EKG intervals"""
    signal_data = np.array(samples)
    
    # Detect waves
    r_peaks = detect_r_peaks(signal_data, sampling_rate)
    p_waves = detect_p_waves(signal_data, sampling_rate, r_peaks)
    t_waves = detect_t_waves(signal_data, sampling_rate, r_peaks)
    
    intervals = {}
    
    # PR interval (P wave to R peak)
    if len(p_waves) > 0 and len(r_peaks) > 0:
        pr_intervals = []
        for p, r in zip(p_waves[:min(len(p_waves), len(r_peaks))], r_peaks[:len(p_waves)]):
            if r > p:
                pr_intervals.append((r - p) * 1000 / sampling_rate)
        if pr_intervals:
            intervals["pr_interval"] = np.mean(pr_intervals)
    
    # QRS duration (simplified)
    if len(r_peaks) > 0:
        intervals["qrs_duration"] = 80.0  # Typical value, would need Q and S detection
    
    # QT interval (Q to T wave end)
    if len(r_peaks) > 0 and len(t_waves) > 0:
        qt_intervals = []
        for r, t in zip(r_peaks[:min(len(r_peaks), len(t_waves))], t_waves[:len(r_peaks)]):
            if t > r:
                qt_intervals.append((t - r) * 1000 / sampling_rate)
        if qt_intervals:
            intervals["qt_interval"] = np.mean(qt_intervals)
            
            # Calculate QTc (Bazett's formula)
            if "heart_rate" in intervals and intervals["heart_rate"] > 0:
                rr_interval = 60000 / intervals["heart_rate"]  # in ms
                intervals["qtc"] = intervals["qt_interval"] / np.sqrt(rr_interval / 1000)
    
    return intervals


def check_signal_quality(samples: List[float], sampling_rate: int) -> Dict[str, Any]:
    """Check EKG signal quality"""
    signal_data = np.array(samples)
    
    issues = []
    recommendations = []
    
    # Check for flat line
    if np.std(signal_data) < 0.01:
        issues.append("Flat line detected")
        recommendations.append("Check electrode connections")
    
    # Check for clipping
    max_val = np.max(np.abs(signal_data))
    if max_val > 0.99:
        issues.append("Signal clipping detected")
        recommendations.append("Reduce gain or check amplifier settings")
    
    # Check for excessive noise
    noise_level = estimate_noise_level(signal_data, sampling_rate)
    if noise_level > 0.2:
        issues.append("High noise level")
        recommendations.append("Check for electrical interference or movement artifacts")
    
    # Check for missing data
    if np.any(np.isnan(signal_data)) or np.any(np.isinf(signal_data)):
        issues.append("Missing or invalid data points")
        recommendations.append("Re-record the signal")
    
    # Calculate quality score
    quality_score = 1.0
    quality_score -= len(issues) * 0.25
    quality_score = max(0, min(1, quality_score))
    
    return {
        "is_valid": len(issues) == 0,
        "quality_score": quality_score,
        "issues": issues,
        "recommendations": recommendations
    }


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
    uvicorn.run(app, host="0.0.0.0", port=8016)