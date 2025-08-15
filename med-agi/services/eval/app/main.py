"""
Med-AGI Evaluation Service
Tracks and evaluates model performance metrics
"""

import os
import logging
import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
import asyncio

from fastapi import FastAPI, HTTPException, Depends, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field, validator
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine, Column, String, Float, Integer, DateTime, JSON, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

from auth import verify_jwt, get_current_user
from opa_client import check_permission
from metrics import setup_metrics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database setup
DATABASE_URL = os.getenv("DB_URL", "postgresql://medagi:medagi@postgres:5432/medagi")
ASYNC_DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")

Base = declarative_base()
engine = None
async_engine = None
AsyncSessionLocal = None


# Database Models
class ModelEvaluation(Base):
    __tablename__ = "model_evaluations"
    
    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String, index=True)
    model_version = Column(String, index=True)
    dataset_name = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Metrics
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    auc_roc = Column(Float)
    
    # Additional info
    num_samples = Column(Integer)
    confusion_matrix = Column(JSON)
    classification_report = Column(JSON)
    feature_importance = Column(JSON)
    metadata = Column(JSON)


class ModelPerformance(Base):
    __tablename__ = "model_performance"
    
    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Real-time metrics
    inference_count = Column(Integer, default=0)
    avg_inference_time = Column(Float)
    error_rate = Column(Float)
    throughput = Column(Float)
    
    # Resource usage
    gpu_utilization = Column(Float)
    memory_usage = Column(Float)
    
    metadata = Column(JSON)


class DatasetInfo(Base):
    __tablename__ = "datasets"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Dataset stats
    num_samples = Column(Integer)
    num_features = Column(Integer)
    num_classes = Column(Integer)
    class_distribution = Column(JSON)
    
    # Data quality
    missing_values = Column(JSON)
    data_quality_score = Column(Float)
    
    metadata = Column(JSON)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global engine, async_engine, AsyncSessionLocal
    
    # Create database engine
    engine = create_engine(DATABASE_URL)
    async_engine = create_async_engine(ASYNC_DATABASE_URL)
    AsyncSessionLocal = async_sessionmaker(async_engine, class_=AsyncSession)
    
    # Create tables
    Base.metadata.create_all(bind=engine)
    
    logger.info("Evaluation Service initialized")
    yield
    
    # Cleanup
    if async_engine:
        await async_engine.dispose()


# Create FastAPI app
app = FastAPI(
    title="Med-AGI Evaluation Service",
    description="Model evaluation and performance tracking",
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
class EvaluationRequest(BaseModel):
    model_name: str = Field(..., description="Name of the model")
    model_version: str = Field(..., description="Version of the model")
    dataset_name: str = Field(..., description="Name of the evaluation dataset")
    predictions: List[int] = Field(..., description="Model predictions")
    ground_truth: List[int] = Field(..., description="Ground truth labels")
    probabilities: Optional[List[List[float]]] = Field(None, description="Prediction probabilities")
    feature_importance: Optional[Dict[str, float]] = Field(None, description="Feature importance scores")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    
    @validator('predictions', 'ground_truth')
    def validate_length(cls, v, values):
        if 'ground_truth' in values and len(v) != len(values['ground_truth']):
            raise ValueError("Predictions and ground truth must have same length")
        return v


class EvaluationResponse(BaseModel):
    evaluation_id: int
    model_name: str
    model_version: str
    dataset_name: str
    timestamp: datetime
    metrics: Dict[str, float]
    confusion_matrix: List[List[int]]
    classification_report: Dict[str, Any]


class PerformanceMetrics(BaseModel):
    model_name: str
    period: str  # "1h", "24h", "7d", "30d"
    avg_accuracy: float
    avg_inference_time: float
    total_inferences: int
    error_rate: float
    availability: float
    trends: Dict[str, Any]


class ModelComparison(BaseModel):
    models: List[str]
    dataset: str
    metrics: Dict[str, Dict[str, float]]
    best_model: str
    comparison_chart: Optional[str] = None


# Dependency to get database session
async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


# Health endpoints
@app.get("/health")
async def health():
    """Basic health check"""
    return {"status": "healthy", "service": "evaluation"}


@app.get("/v1/eval/health")
async def detailed_health():
    """Detailed health check"""
    try:
        # Check database connection
        async with AsyncSessionLocal() as session:
            result = await session.execute("SELECT 1")
            db_healthy = result.scalar() == 1
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        db_healthy = False
    
    return {
        "status": "healthy" if db_healthy else "degraded",
        "database": "connected" if db_healthy else "disconnected",
        "service": "evaluation"
    }


# Evaluation endpoints
@app.post("/v1/eval/evaluate", response_model=EvaluationResponse)
async def evaluate_model(
    request: EvaluationRequest,
    db: AsyncSession = Depends(get_db),
    current_user: Dict = Depends(get_current_user)
):
    """
    Evaluate model performance on a dataset
    """
    # Check permissions
    if not await check_permission(current_user, "eval:evaluate", {"model": request.model_name}):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions for model evaluation"
        )
    
    try:
        # Calculate metrics
        metrics = calculate_metrics(
            request.ground_truth,
            request.predictions,
            request.probabilities
        )
        
        # Generate confusion matrix
        cm = confusion_matrix(request.ground_truth, request.predictions)
        
        # Generate classification report
        report = classification_report(
            request.ground_truth,
            request.predictions,
            output_dict=True
        )
        
        # Create evaluation record
        eval_record = ModelEvaluation(
            model_name=request.model_name,
            model_version=request.model_version,
            dataset_name=request.dataset_name,
            accuracy=metrics['accuracy'],
            precision=metrics['precision'],
            recall=metrics['recall'],
            f1_score=metrics['f1_score'],
            auc_roc=metrics.get('auc_roc'),
            num_samples=len(request.ground_truth),
            confusion_matrix=cm.tolist(),
            classification_report=report,
            feature_importance=request.feature_importance,
            metadata=request.metadata
        )
        
        db.add(eval_record)
        await db.commit()
        await db.refresh(eval_record)
        
        return EvaluationResponse(
            evaluation_id=eval_record.id,
            model_name=eval_record.model_name,
            model_version=eval_record.model_version,
            dataset_name=eval_record.dataset_name,
            timestamp=eval_record.timestamp,
            metrics=metrics,
            confusion_matrix=cm.tolist(),
            classification_report=report
        )
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evaluation failed: {str(e)}"
        )


@app.get("/v1/eval/summary")
async def get_evaluation_summary(
    model_name: Optional[str] = None,
    dataset_name: Optional[str] = None,
    days: int = Query(30, description="Number of days to look back"),
    db: AsyncSession = Depends(get_db),
    current_user: Dict = Depends(get_current_user)
):
    """Get evaluation summary for models"""
    # Check permissions
    if not await check_permission(current_user, "eval:read", {}):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions to view evaluations"
        )
    
    try:
        # Build query
        from sqlalchemy import select, func
        
        query = select(ModelEvaluation)
        
        if model_name:
            query = query.where(ModelEvaluation.model_name == model_name)
        if dataset_name:
            query = query.where(ModelEvaluation.dataset_name == dataset_name)
        
        # Time filter
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        query = query.where(ModelEvaluation.timestamp >= cutoff_date)
        
        result = await db.execute(query)
        evaluations = result.scalars().all()
        
        if not evaluations:
            return {
                "message": "No evaluations found",
                "filters": {
                    "model_name": model_name,
                    "dataset_name": dataset_name,
                    "days": days
                }
            }
        
        # Calculate summary statistics
        summary = {
            "total_evaluations": len(evaluations),
            "models": list(set(e.model_name for e in evaluations)),
            "datasets": list(set(e.dataset_name for e in evaluations)),
            "time_range": {
                "start": min(e.timestamp for e in evaluations).isoformat(),
                "end": max(e.timestamp for e in evaluations).isoformat()
            },
            "metrics_summary": calculate_summary_stats(evaluations),
            "recent_evaluations": [
                {
                    "id": e.id,
                    "model": e.model_name,
                    "version": e.model_version,
                    "dataset": e.dataset_name,
                    "accuracy": e.accuracy,
                    "f1_score": e.f1_score,
                    "timestamp": e.timestamp.isoformat()
                }
                for e in sorted(evaluations, key=lambda x: x.timestamp, reverse=True)[:10]
            ]
        }
        
        return summary
        
    except Exception as e:
        logger.error(f"Failed to get evaluation summary: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get summary: {str(e)}"
        )


@app.post("/v1/eval/compare")
async def compare_models(
    model_names: List[str],
    dataset_name: str,
    db: AsyncSession = Depends(get_db),
    current_user: Dict = Depends(get_current_user)
) -> ModelComparison:
    """Compare performance of multiple models"""
    # Check permissions
    if not await check_permission(current_user, "eval:compare", {}):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions for model comparison"
        )
    
    try:
        from sqlalchemy import select
        
        # Get evaluations for each model
        comparisons = {}
        
        for model_name in model_names:
            query = select(ModelEvaluation).where(
                ModelEvaluation.model_name == model_name,
                ModelEvaluation.dataset_name == dataset_name
            ).order_by(ModelEvaluation.timestamp.desc()).limit(1)
            
            result = await db.execute(query)
            eval_record = result.scalar_one_or_none()
            
            if eval_record:
                comparisons[model_name] = {
                    "accuracy": eval_record.accuracy,
                    "precision": eval_record.precision,
                    "recall": eval_record.recall,
                    "f1_score": eval_record.f1_score,
                    "auc_roc": eval_record.auc_roc or 0
                }
        
        if not comparisons:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No evaluations found for specified models"
            )
        
        # Determine best model
        best_model = max(comparisons.keys(), key=lambda k: comparisons[k]["f1_score"])
        
        # Generate comparison chart
        chart_path = generate_comparison_chart(comparisons, dataset_name)
        
        return ModelComparison(
            models=model_names,
            dataset=dataset_name,
            metrics=comparisons,
            best_model=best_model,
            comparison_chart=chart_path
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Model comparison failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Comparison failed: {str(e)}"
        )


@app.post("/v1/eval/performance")
async def track_performance(
    model_name: str,
    inference_count: int,
    avg_inference_time: float,
    error_rate: float,
    gpu_utilization: Optional[float] = None,
    memory_usage: Optional[float] = None,
    metadata: Optional[Dict] = None,
    db: AsyncSession = Depends(get_db),
    current_user: Dict = Depends(get_current_user)
):
    """Track real-time model performance metrics"""
    # Check permissions
    if not await check_permission(current_user, "eval:track", {"model": model_name}):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions to track performance"
        )
    
    try:
        # Calculate throughput
        throughput = inference_count / avg_inference_time if avg_inference_time > 0 else 0
        
        # Create performance record
        perf_record = ModelPerformance(
            model_name=model_name,
            inference_count=inference_count,
            avg_inference_time=avg_inference_time,
            error_rate=error_rate,
            throughput=throughput,
            gpu_utilization=gpu_utilization,
            memory_usage=memory_usage,
            metadata=metadata or {}
        )
        
        db.add(perf_record)
        await db.commit()
        
        return {
            "status": "recorded",
            "model_name": model_name,
            "timestamp": perf_record.timestamp.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to track performance: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to track performance: {str(e)}"
        )


@app.get("/v1/eval/performance/{model_name}", response_model=PerformanceMetrics)
async def get_performance_metrics(
    model_name: str,
    period: str = Query("24h", regex="^(1h|24h|7d|30d)$"),
    db: AsyncSession = Depends(get_db),
    current_user: Dict = Depends(get_current_user)
):
    """Get performance metrics for a model over time"""
    # Check permissions
    if not await check_permission(current_user, "eval:read", {"model": model_name}):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions to view performance"
        )
    
    try:
        # Parse period
        period_map = {
            "1h": timedelta(hours=1),
            "24h": timedelta(days=1),
            "7d": timedelta(days=7),
            "30d": timedelta(days=30)
        }
        
        cutoff_time = datetime.utcnow() - period_map[period]
        
        # Query performance data
        from sqlalchemy import select, func
        
        query = select(ModelPerformance).where(
            ModelPerformance.model_name == model_name,
            ModelPerformance.timestamp >= cutoff_time
        )
        
        result = await db.execute(query)
        records = result.scalars().all()
        
        if not records:
            return PerformanceMetrics(
                model_name=model_name,
                period=period,
                avg_accuracy=0,
                avg_inference_time=0,
                total_inferences=0,
                error_rate=0,
                availability=0,
                trends={}
            )
        
        # Calculate metrics
        total_inferences = sum(r.inference_count for r in records)
        avg_inference_time = np.mean([r.avg_inference_time for r in records])
        avg_error_rate = np.mean([r.error_rate for r in records])
        
        # Calculate availability (time with error_rate < 0.01)
        good_records = [r for r in records if r.error_rate < 0.01]
        availability = len(good_records) / len(records) if records else 0
        
        # Calculate trends
        trends = calculate_trends(records)
        
        # Get accuracy from evaluations
        eval_query = select(func.avg(ModelEvaluation.accuracy)).where(
            ModelEvaluation.model_name == model_name,
            ModelEvaluation.timestamp >= cutoff_time
        )
        eval_result = await db.execute(eval_query)
        avg_accuracy = eval_result.scalar() or 0
        
        return PerformanceMetrics(
            model_name=model_name,
            period=period,
            avg_accuracy=float(avg_accuracy),
            avg_inference_time=float(avg_inference_time),
            total_inferences=total_inferences,
            error_rate=float(avg_error_rate),
            availability=float(availability),
            trends=trends
        )
        
    except Exception as e:
        logger.error(f"Failed to get performance metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get metrics: {str(e)}"
        )


@app.post("/v1/eval/dataset")
async def register_dataset(
    name: str,
    description: str,
    num_samples: int,
    num_features: int,
    num_classes: int,
    class_distribution: Dict[str, int],
    missing_values: Optional[Dict[str, int]] = None,
    metadata: Optional[Dict] = None,
    db: AsyncSession = Depends(get_db),
    current_user: Dict = Depends(get_current_user)
):
    """Register a new dataset for evaluation"""
    # Check permissions
    if not await check_permission(current_user, "eval:dataset:create", {}):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions to register dataset"
        )
    
    try:
        # Calculate data quality score
        quality_score = calculate_data_quality(
            num_samples,
            num_features,
            missing_values,
            class_distribution
        )
        
        # Create dataset record
        dataset = DatasetInfo(
            name=name,
            description=description,
            num_samples=num_samples,
            num_features=num_features,
            num_classes=num_classes,
            class_distribution=class_distribution,
            missing_values=missing_values or {},
            data_quality_score=quality_score,
            metadata=metadata or {}
        )
        
        db.add(dataset)
        await db.commit()
        await db.refresh(dataset)
        
        return {
            "dataset_id": dataset.id,
            "name": dataset.name,
            "quality_score": dataset.data_quality_score,
            "created_at": dataset.created_at.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to register dataset: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to register dataset: {str(e)}"
        )


@app.get("/v1/eval/datasets")
async def list_datasets(
    db: AsyncSession = Depends(get_db),
    current_user: Dict = Depends(get_current_user)
):
    """List all registered datasets"""
    # Check permissions
    if not await check_permission(current_user, "eval:dataset:read", {}):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions to view datasets"
        )
    
    try:
        from sqlalchemy import select
        
        query = select(DatasetInfo).order_by(DatasetInfo.created_at.desc())
        result = await db.execute(query)
        datasets = result.scalars().all()
        
        return [
            {
                "id": d.id,
                "name": d.name,
                "description": d.description,
                "num_samples": d.num_samples,
                "num_classes": d.num_classes,
                "quality_score": d.data_quality_score,
                "created_at": d.created_at.isoformat()
            }
            for d in datasets
        ]
        
    except Exception as e:
        logger.error(f"Failed to list datasets: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list datasets: {str(e)}"
        )


@app.get("/v1/eval/report/{evaluation_id}")
async def get_evaluation_report(
    evaluation_id: int,
    format: str = Query("json", regex="^(json|html|pdf)$"),
    db: AsyncSession = Depends(get_db),
    current_user: Dict = Depends(get_current_user)
):
    """Get detailed evaluation report"""
    # Check permissions
    if not await check_permission(current_user, "eval:report", {}):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions to view reports"
        )
    
    try:
        from sqlalchemy import select
        
        query = select(ModelEvaluation).where(ModelEvaluation.id == evaluation_id)
        result = await db.execute(query)
        evaluation = result.scalar_one_or_none()
        
        if not evaluation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Evaluation not found"
            )
        
        if format == "json":
            return {
                "evaluation_id": evaluation.id,
                "model_name": evaluation.model_name,
                "model_version": evaluation.model_version,
                "dataset_name": evaluation.dataset_name,
                "timestamp": evaluation.timestamp.isoformat(),
                "metrics": {
                    "accuracy": evaluation.accuracy,
                    "precision": evaluation.precision,
                    "recall": evaluation.recall,
                    "f1_score": evaluation.f1_score,
                    "auc_roc": evaluation.auc_roc
                },
                "confusion_matrix": evaluation.confusion_matrix,
                "classification_report": evaluation.classification_report,
                "feature_importance": evaluation.feature_importance,
                "metadata": evaluation.metadata
            }
        elif format == "html":
            # Generate HTML report
            html_content = generate_html_report(evaluation)
            return HTMLResponse(content=html_content)
        elif format == "pdf":
            # Generate PDF report (would need additional library)
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="PDF export not yet implemented"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get evaluation report: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get report: {str(e)}"
        )


# Utility functions
def calculate_metrics(
    ground_truth: List[int],
    predictions: List[int],
    probabilities: Optional[List[List[float]]] = None
) -> Dict[str, float]:
    """Calculate evaluation metrics"""
    metrics = {
        "accuracy": accuracy_score(ground_truth, predictions),
        "precision": precision_score(ground_truth, predictions, average='weighted', zero_division=0),
        "recall": recall_score(ground_truth, predictions, average='weighted', zero_division=0),
        "f1_score": f1_score(ground_truth, predictions, average='weighted', zero_division=0)
    }
    
    # Calculate AUC-ROC if probabilities provided
    if probabilities:
        try:
            # For multi-class, calculate macro average
            from sklearn.preprocessing import label_binarize
            
            n_classes = len(probabilities[0])
            if n_classes == 2:
                # Binary classification
                probs = [p[1] for p in probabilities]
                metrics["auc_roc"] = roc_auc_score(ground_truth, probs)
            else:
                # Multi-class
                y_true_bin = label_binarize(ground_truth, classes=list(range(n_classes)))
                metrics["auc_roc"] = roc_auc_score(y_true_bin, probabilities, average='macro')
        except Exception as e:
            logger.warning(f"Could not calculate AUC-ROC: {e}")
            metrics["auc_roc"] = None
    
    return metrics


def calculate_summary_stats(evaluations: List[ModelEvaluation]) -> Dict[str, Any]:
    """Calculate summary statistics from evaluations"""
    if not evaluations:
        return {}
    
    return {
        "accuracy": {
            "mean": np.mean([e.accuracy for e in evaluations]),
            "std": np.std([e.accuracy for e in evaluations]),
            "min": min(e.accuracy for e in evaluations),
            "max": max(e.accuracy for e in evaluations)
        },
        "f1_score": {
            "mean": np.mean([e.f1_score for e in evaluations]),
            "std": np.std([e.f1_score for e in evaluations]),
            "min": min(e.f1_score for e in evaluations),
            "max": max(e.f1_score for e in evaluations)
        },
        "total_samples": sum(e.num_samples for e in evaluations)
    }


def calculate_trends(records: List[ModelPerformance]) -> Dict[str, Any]:
    """Calculate performance trends"""
    if len(records) < 2:
        return {}
    
    # Sort by timestamp
    sorted_records = sorted(records, key=lambda r: r.timestamp)
    
    # Calculate trends
    inference_times = [r.avg_inference_time for r in sorted_records]
    error_rates = [r.error_rate for r in sorted_records]
    
    return {
        "inference_time_trend": "improving" if inference_times[-1] < inference_times[0] else "degrading",
        "error_rate_trend": "improving" if error_rates[-1] < error_rates[0] else "degrading",
        "latest_vs_oldest": {
            "inference_time_change": (inference_times[-1] - inference_times[0]) / inference_times[0] * 100,
            "error_rate_change": (error_rates[-1] - error_rates[0]) / (error_rates[0] + 1e-10) * 100
        }
    }


def calculate_data_quality(
    num_samples: int,
    num_features: int,
    missing_values: Optional[Dict[str, int]],
    class_distribution: Dict[str, int]
) -> float:
    """Calculate data quality score"""
    score = 1.0
    
    # Penalize small datasets
    if num_samples < 100:
        score *= 0.5
    elif num_samples < 1000:
        score *= 0.8
    
    # Penalize missing values
    if missing_values:
        total_missing = sum(missing_values.values())
        missing_ratio = total_missing / (num_samples * num_features)
        score *= (1 - missing_ratio)
    
    # Penalize class imbalance
    if class_distribution:
        total = sum(class_distribution.values())
        proportions = [count/total for count in class_distribution.values()]
        max_prop = max(proportions)
        if max_prop > 0.9:  # Severe imbalance
            score *= 0.7
        elif max_prop > 0.7:  # Moderate imbalance
            score *= 0.9
    
    return max(0, min(1, score))


def generate_comparison_chart(comparisons: Dict[str, Dict[str, float]], dataset_name: str) -> str:
    """Generate comparison chart"""
    try:
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Model Comparison on {dataset_name}', fontsize=16)
        
        models = list(comparisons.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            values = [comparisons[model][metric] for model in models]
            
            bars = ax.bar(models, values)
            ax.set_title(metric.replace('_', ' ').title())
            ax.set_ylim(0, 1)
            ax.set_ylabel('Score')
            
            # Color best performer
            best_idx = values.index(max(values))
            bars[best_idx].set_color('green')
        
        plt.tight_layout()
        
        # Save to file
        chart_path = f"/tmp/comparison_{dataset_name}_{datetime.now().timestamp()}.png"
        plt.savefig(chart_path)
        plt.close()
        
        return chart_path
        
    except Exception as e:
        logger.error(f"Failed to generate comparison chart: {e}")
        return None


def generate_html_report(evaluation: ModelEvaluation) -> str:
    """Generate HTML evaluation report"""
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Evaluation Report - {evaluation.model_name}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #333; }}
            .metric {{ margin: 10px 0; }}
            .metric-label {{ font-weight: bold; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <h1>Model Evaluation Report</h1>
        <h2>{evaluation.model_name} v{evaluation.model_version}</h2>
        
        <div class="metric">
            <span class="metric-label">Dataset:</span> {evaluation.dataset_name}
        </div>
        <div class="metric">
            <span class="metric-label">Timestamp:</span> {evaluation.timestamp}
        </div>
        <div class="metric">
            <span class="metric-label">Samples:</span> {evaluation.num_samples}
        </div>
        
        <h3>Performance Metrics</h3>
        <div class="metric">
            <span class="metric-label">Accuracy:</span> {evaluation.accuracy:.4f}
        </div>
        <div class="metric">
            <span class="metric-label">Precision:</span> {evaluation.precision:.4f}
        </div>
        <div class="metric">
            <span class="metric-label">Recall:</span> {evaluation.recall:.4f}
        </div>
        <div class="metric">
            <span class="metric-label">F1 Score:</span> {evaluation.f1_score:.4f}
        </div>
        
        <h3>Confusion Matrix</h3>
        <table>
            {"".join(f"<tr>{''.join(f'<td>{val}</td>' for val in row)}</tr>" for row in evaluation.confusion_matrix)}
        </table>
    </body>
    </html>
    """
    return html


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
    uvicorn.run(app, host="0.0.0.0", port=8005)