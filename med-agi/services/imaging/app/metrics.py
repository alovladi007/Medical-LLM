"""
Metrics module for Prometheus monitoring
"""

import time
import logging
from functools import wraps
from typing import Callable
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from fastapi import FastAPI, Response

logger = logging.getLogger(__name__)

# Define metrics
inference_counter = Counter(
    'medagi_imaging_inference_total',
    'Total number of inference requests',
    ['model', 'modality', 'status']
)

inference_duration = Histogram(
    'medagi_imaging_inference_duration_seconds',
    'Inference request duration',
    ['model', 'modality']
)

active_requests = Gauge(
    'medagi_imaging_active_requests',
    'Number of active requests'
)

model_accuracy = Gauge(
    'medagi_imaging_model_accuracy',
    'Model accuracy metric',
    ['model']
)


def setup_metrics(app: FastAPI):
    """Setup Prometheus metrics endpoint"""
    
    @app.get("/metrics")
    async def metrics():
        """Prometheus metrics endpoint"""
        return Response(
            content=generate_latest(),
            media_type="text/plain"
        )


def track_inference(func: Callable) -> Callable:
    """Decorator to track inference metrics"""
    
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        active_requests.inc()
        
        model = kwargs.get('model_name', 'unknown')
        modality = kwargs.get('modality', 'unknown')
        status = 'success'
        
        try:
            result = await func(*args, **kwargs)
            return result
        except Exception as e:
            status = 'error'
            raise
        finally:
            # Record metrics
            duration = time.time() - start_time
            inference_counter.labels(
                model=model,
                modality=modality,
                status=status
            ).inc()
            
            inference_duration.labels(
                model=model,
                modality=modality
            ).observe(duration)
            
            active_requests.dec()
    
    return wrapper