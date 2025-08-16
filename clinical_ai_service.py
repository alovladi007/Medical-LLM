"""
Med-AGI Clinical AI Engine
Advanced medical intelligence service for diagnosis, treatment planning, and clinical decision support
"""

import os
import logging
import asyncio
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from collections import defaultdict

# FastAPI and async components
from fastapi import FastAPI, HTTPException, Depends, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from contextlib import asynccontextmanager

# Medical AI Components
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import joblib

# NLP and Knowledge Graph
import spacy
import networkx as nx
from transformers import pipeline, AutoTokenizer, AutoModel
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Data Models and Enums
# ============================================================================

class Urgency(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    ROUTINE = "routine"


class ConfidenceLevel(str, Enum):
    VERY_HIGH = "very_high"  # >90%
    HIGH = "high"            # 75-90%
    MODERATE = "moderate"    # 50-75%
    LOW = "low"              # 25-50%
    VERY_LOW = "very_low"    # <25%


@dataclass
class VitalSigns:
    heart_rate: int
    blood_pressure_systolic: int
    blood_pressure_diastolic: int
    temperature: float
    respiratory_rate: int
    oxygen_saturation: int
    
    def is_critical(self) -> bool:
        """Check if any vital sign is in critical range"""
        return (
            self.heart_rate < 40 or self.heart_rate > 150 or
            self.blood_pressure_systolic < 90 or self.blood_pressure_systolic > 180 or
            self.temperature < 95 or self.temperature > 104 or
            self.oxygen_saturation < 90 or
            self.respiratory_rate < 8 or self.respiratory_rate > 30
        )


@dataclass
class Symptom:
    name: str
    severity: str  # mild, moderate, severe
    duration: str  # acute, subacute, chronic
    onset: str
    associated_symptoms: List[str] = None


@dataclass
class Diagnosis:
    condition: str
    probability: float
    icd10_code: str
    evidence: List[str]
    differential_diagnoses: List[str]
    urgency: Urgency
    confidence_level: ConfidenceLevel


@dataclass
class Treatment:
    medication: str
    dosage: str
    route: str
    frequency: str
    duration: str
    contraindications: List[str]
    interactions: List[str]
    monitoring_required: List[str]


@dataclass
class ClinicalDecision:
    diagnosis: Diagnosis
    treatments: List[Treatment]
    diagnostic_tests: List[str]
    referrals: List[str]
    follow_up: str
    clinical_pearls: List[str]
    risk_factors: List[str]
    prognosis: str


# ============================================================================
# Medical Knowledge Base
# ============================================================================

class MedicalKnowledgeGraph:
    """Graph-based medical knowledge representation"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self._initialize_knowledge_base()
    
    def _initialize_knowledge_base(self):
        """Initialize with medical knowledge"""
        # Add disease nodes
        diseases = {
            "myocardial_infarction": {
                "icd10": "I21",
                "symptoms": ["chest_pain", "dyspnea", "diaphoresis", "nausea"],
                "risk_factors": ["hypertension", "diabetes", "smoking", "hyperlipidemia"],
                "treatments": ["aspirin", "nitroglycerin", "morphine", "PCI"],
                "urgency": "critical"
            },
            "pneumonia": {
                "icd10": "J18",
                "symptoms": ["fever", "cough", "dyspnea", "chest_pain"],
                "risk_factors": ["age>65", "smoking", "COPD", "immunosuppression"],
                "treatments": ["antibiotics", "oxygen", "fluids"],
                "urgency": "high"
            },
            "heart_failure": {
                "icd10": "I50",
                "symptoms": ["dyspnea", "edema", "fatigue", "orthopnea"],
                "risk_factors": ["hypertension", "CAD", "diabetes", "obesity"],
                "treatments": ["diuretics", "ACE_inhibitors", "beta_blockers"],
                "urgency": "high"
            }
        }
        
        for disease, attrs in diseases.items():
            self.graph.add_node(disease, **attrs)
            
            # Add symptom relationships
            for symptom in attrs["symptoms"]:
                self.graph.add_edge(symptom, disease, relationship="indicates")
            
            # Add risk factor relationships
            for risk in attrs["risk_factors"]:
                self.graph.add_edge(risk, disease, relationship="increases_risk")
    
    def find_related_conditions(self, symptoms: List[str]) -> List[Tuple[str, float]]:
        """Find conditions related to given symptoms"""
        conditions = defaultdict(float)
        
        for symptom in symptoms:
            if symptom in self.graph:
                for successor in self.graph.successors(symptom):
                    if self.graph.edges[symptom, successor].get("relationship") == "indicates":
                        conditions[successor] += 1.0 / len(symptoms)
        
        return sorted(conditions.items(), key=lambda x: x[1], reverse=True)
    
    def get_treatment_plan(self, condition: str) -> List[str]:
        """Get treatment options for a condition"""
        if condition in self.graph:
            return self.graph.nodes[condition].get("treatments", [])
        return []


# ============================================================================
# Clinical AI Engine
# ============================================================================

class ClinicalAIEngine:
    """Core AI engine for clinical decision support"""
    
    def __init__(self):
        self.knowledge_graph = MedicalKnowledgeGraph()
        self.nlp_model = self._initialize_nlp()
        self.diagnostic_model = self._initialize_diagnostic_model()
        self.risk_predictor = self._initialize_risk_model()
        
    def _initialize_nlp(self):
        """Initialize medical NLP model"""
        try:
            # In production, use BioBERT or ClinicalBERT
            return pipeline("ner", model="dmis-lab/biobert-base-cased-v1.1")
        except:
            # Fallback to basic NER
            return None
    
    def _initialize_diagnostic_model(self):
        """Initialize diagnostic prediction model"""
        # In production, load pre-trained model
        # For demo, create a simple model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        return model
    
    def _initialize_risk_model(self):
        """Initialize risk assessment model"""
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        return model
    
    async def analyze_patient(
        self,
        symptoms: List[Symptom],
        vitals: VitalSigns,
        medical_history: Dict[str, Any],
        lab_results: Optional[Dict[str, Any]] = None
    ) -> ClinicalDecision:
        """Comprehensive patient analysis"""
        
        # Extract symptom names
        symptom_names = [s.name.lower().replace(" ", "_") for s in symptoms]
        
        # Find related conditions
        potential_conditions = self.knowledge_graph.find_related_conditions(symptom_names)
        
        # Calculate urgency based on vitals
        urgency = self._calculate_urgency(vitals, symptoms)
        
        # Generate diagnoses
        diagnoses = []
        for condition, score in potential_conditions[:5]:  # Top 5 differentials
            diagnosis = Diagnosis(
                condition=condition.replace("_", " ").title(),
                probability=min(score * 1.5, 1.0),  # Normalize probability
                icd10_code=self.knowledge_graph.graph.nodes[condition].get("icd10", ""),
                evidence=self._gather_evidence(condition, symptoms, vitals, lab_results),
                differential_diagnoses=[c[0] for c in potential_conditions[1:4]],
                urgency=urgency,
                confidence_level=self._calculate_confidence(score)
            )
            diagnoses.append(diagnosis)
        
        # Get primary diagnosis
        primary_diagnosis = diagnoses[0] if diagnoses else self._create_default_diagnosis()
        
        # Generate treatment plan
        treatments = self._generate_treatment_plan(
            primary_diagnosis.condition.lower().replace(" ", "_"),
            medical_history
        )
        
        # Recommend diagnostic tests
        diagnostic_tests = self._recommend_tests(symptoms, vitals, lab_results)
        
        # Generate clinical decision
        decision = ClinicalDecision(
            diagnosis=primary_diagnosis,
            treatments=treatments,
            diagnostic_tests=diagnostic_tests,
            referrals=self._generate_referrals(primary_diagnosis),
            follow_up=self._determine_followup(urgency),
            clinical_pearls=self._get_clinical_pearls(primary_diagnosis.condition),
            risk_factors=self._identify_risk_factors(medical_history),
            prognosis=self._estimate_prognosis(primary_diagnosis, medical_history)
        )
        
        return decision
    
    def _calculate_urgency(self, vitals: VitalSigns, symptoms: List[Symptom]) -> Urgency:
        """Calculate clinical urgency"""
        if vitals.is_critical():
            return Urgency.CRITICAL
        
        severe_symptoms = [s for s in symptoms if s.severity == "severe"]
        if len(severe_symptoms) >= 2:
            return Urgency.HIGH
        elif len(severe_symptoms) == 1:
            return Urgency.MODERATE
        
        return Urgency.ROUTINE
    
    def _calculate_confidence(self, score: float) -> ConfidenceLevel:
        """Calculate confidence level from probability score"""
        if score > 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif score > 0.75:
            return ConfidenceLevel.HIGH
        elif score > 0.5:
            return ConfidenceLevel.MODERATE
        elif score > 0.25:
            return ConfidenceLevel.LOW
        return ConfidenceLevel.VERY_LOW
    
    def _gather_evidence(
        self,
        condition: str,
        symptoms: List[Symptom],
        vitals: VitalSigns,
        lab_results: Optional[Dict]
    ) -> List[str]:
        """Gather supporting evidence for diagnosis"""
        evidence = []
        
        # Symptom evidence
        condition_symptoms = self.knowledge_graph.graph.nodes.get(condition, {}).get("symptoms", [])
        for symptom in symptoms:
            if symptom.name.lower().replace(" ", "_") in condition_symptoms:
                evidence.append(f"{symptom.name} ({symptom.severity})")
        
        # Vital signs evidence
        if vitals.is_critical():
            evidence.append("Critical vital signs")
        
        # Lab evidence
        if lab_results:
            for test, value in lab_results.items():
                if self._is_abnormal(test, value):
                    evidence.append(f"Abnormal {test}: {value}")
        
        return evidence
    
    def _is_abnormal(self, test: str, value: Any) -> bool:
        """Check if lab value is abnormal"""
        # Simplified normal ranges
        normal_ranges = {
            "wbc": (4.5, 11.0),
            "hemoglobin": (12.0, 17.0),
            "platelets": (150, 450),
            "creatinine": (0.6, 1.2),
            "glucose": (70, 100),
            "troponin": (0, 0.04)
        }
        
        if test.lower() in normal_ranges:
            min_val, max_val = normal_ranges[test.lower()]
            try:
                val = float(value)
                return val < min_val or val > max_val
            except:
                return False
        return False
    
    def _generate_treatment_plan(
        self,
        condition: str,
        medical_history: Dict[str, Any]
    ) -> List[Treatment]:
        """Generate personalized treatment plan"""
        treatments = []
        
        # Get standard treatments from knowledge graph
        standard_treatments = self.knowledge_graph.get_treatment_plan(condition)
        
        # Convert to Treatment objects with personalization
        for treatment_name in standard_treatments[:3]:  # Top 3 treatments
            treatment = self._create_treatment(treatment_name, medical_history)
            treatments.append(treatment)
        
        return treatments
    
    def _create_treatment(self, treatment_name: str, medical_history: Dict) -> Treatment:
        """Create detailed treatment recommendation"""
        # Simplified treatment database
        treatment_db = {
            "aspirin": {
                "dosage": "325mg",
                "route": "PO",
                "frequency": "once daily",
                "duration": "indefinite",
                "contraindications": ["bleeding_disorder", "aspirin_allergy"],
                "interactions": ["warfarin", "NSAIDs"],
                "monitoring": ["bleeding", "GI_symptoms"]
            },
            "nitroglycerin": {
                "dosage": "0.4mg",
                "route": "sublingual",
                "frequency": "q5min x3 PRN",
                "duration": "as needed",
                "contraindications": ["hypotension", "viagra_use"],
                "interactions": ["sildenafil", "tadalafil"],
                "monitoring": ["blood_pressure", "headache"]
            },
            "antibiotics": {
                "dosage": "500mg",
                "route": "PO",
                "frequency": "twice daily",
                "duration": "7-10 days",
                "contraindications": ["penicillin_allergy"],
                "interactions": ["warfarin", "oral_contraceptives"],
                "monitoring": ["rash", "diarrhea"]
            }
        }
        
        details = treatment_db.get(treatment_name, {
            "dosage": "standard",
            "route": "PO",
            "frequency": "as directed",
            "duration": "as prescribed",
            "contraindications": [],
            "interactions": [],
            "monitoring": []
        })
        
        return Treatment(
            medication=treatment_name.replace("_", " ").title(),
            dosage=details["dosage"],
            route=details["route"],
            frequency=details["frequency"],
            duration=details["duration"],
            contraindications=details["contraindications"],
            interactions=details["interactions"],
            monitoring_required=details["monitoring"]
        )
    
    def _recommend_tests(
        self,
        symptoms: List[Symptom],
        vitals: VitalSigns,
        existing_labs: Optional[Dict]
    ) -> List[str]:
        """Recommend diagnostic tests"""
        tests = []
        
        # Basic tests based on symptoms
        symptom_names = [s.name.lower() for s in symptoms]
        
        if "chest pain" in symptom_names:
            tests.extend(["EKG", "Troponin", "Chest X-ray", "D-dimer"])
        
        if "fever" in symptom_names:
            tests.extend(["CBC with differential", "Blood cultures", "Urinalysis"])
        
        if "dyspnea" in symptom_names:
            tests.extend(["ABG", "BNP", "Chest X-ray", "CT Angiography"])
        
        # Remove duplicates and already performed tests
        if existing_labs:
            tests = [t for t in tests if t.lower() not in [k.lower() for k in existing_labs.keys()]]
        
        return list(set(tests))[:6]  # Return top 6 unique tests
    
    def _generate_referrals(self, diagnosis: Diagnosis) -> List[str]:
        """Generate specialist referrals"""
        referrals = []
        
        specialty_map = {
            "myocardial": ["Cardiology", "Cardiac Catheterization Lab"],
            "pneumonia": ["Pulmonology", "Infectious Disease"],
            "heart failure": ["Cardiology", "Heart Failure Clinic"],
            "diabetes": ["Endocrinology", "Diabetes Education"],
            "cancer": ["Oncology", "Palliative Care"]
        }
        
        condition_lower = diagnosis.condition.lower()
        for keyword, specialists in specialty_map.items():
            if keyword in condition_lower:
                referrals.extend(specialists)
                break
        
        return referrals[:2]  # Top 2 referrals
    
    def _determine_followup(self, urgency: Urgency) -> str:
        """Determine follow-up timing"""
        followup_map = {
            Urgency.CRITICAL: "Immediate admission",
            Urgency.HIGH: "Follow up within 24-48 hours",
            Urgency.MODERATE: "Follow up within 1 week",
            Urgency.LOW: "Follow up within 2-4 weeks",
            Urgency.ROUTINE: "Follow up in 1-3 months"
        }
        return followup_map.get(urgency, "As needed")
    
    def _get_clinical_pearls(self, condition: str) -> List[str]:
        """Get clinical pearls for condition"""
        pearls_db = {
            "Myocardial Infarction": [
                "Time is muscle - door to balloon time <90 minutes",
                "Atypical presentations common in women and diabetics",
                "Serial troponins may be needed if initial negative"
            ],
            "Pneumonia": [
                "CURB-65 score helps determine admission need",
                "Consider atypical pathogens in young adults",
                "Procalcitonin can guide antibiotic duration"
            ],
            "Heart Failure": [
                "BNP >500 pg/mL suggests heart failure",
                "Daily weights most sensitive for volume status",
                "ACE inhibitors reduce mortality by 20-30%"
            ]
        }
        
        return pearls_db.get(condition, [
            "Consider patient's baseline functional status",
            "Reassess if no improvement in 48-72 hours",
            "Document clinical reasoning thoroughly"
        ])
    
    def _identify_risk_factors(self, medical_history: Dict) -> List[str]:
        """Identify relevant risk factors"""
        risk_factors = []
        
        if medical_history.get("smoking"):
            risk_factors.append("Smoking history")
        
        if medical_history.get("diabetes"):
            risk_factors.append("Diabetes mellitus")
        
        if medical_history.get("hypertension"):
            risk_factors.append("Hypertension")
        
        if medical_history.get("age", 0) > 65:
            risk_factors.append("Age >65 years")
        
        if medical_history.get("bmi", 0) > 30:
            risk_factors.append("Obesity (BMI >30)")
        
        return risk_factors
    
    def _estimate_prognosis(self, diagnosis: Diagnosis, medical_history: Dict) -> str:
        """Estimate prognosis"""
        base_prognosis = {
            Urgency.CRITICAL: "Guarded - requires immediate intervention",
            Urgency.HIGH: "Fair with appropriate treatment",
            Urgency.MODERATE: "Good with treatment compliance",
            Urgency.LOW: "Excellent with routine care",
            Urgency.ROUTINE: "Excellent"
        }
        
        prognosis = base_prognosis.get(diagnosis.urgency, "Good")
        
        # Modify based on risk factors
        risk_count = len(self._identify_risk_factors(medical_history))
        if risk_count > 3:
            prognosis += " (multiple risk factors present)"
        
        return prognosis
    
    def _create_default_diagnosis(self) -> Diagnosis:
        """Create default diagnosis when no clear match"""
        return Diagnosis(
            condition="Undifferentiated Syndrome",
            probability=0.0,
            icd10_code="R69",
            evidence=["Insufficient data for specific diagnosis"],
            differential_diagnoses=["Requires further evaluation"],
            urgency=Urgency.MODERATE,
            confidence_level=ConfidenceLevel.VERY_LOW
        )


# ============================================================================
# FastAPI Application
# ============================================================================

# Initialize engine
clinical_ai = ClinicalAIEngine()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("Clinical AI Engine starting...")
    yield
    logger.info("Clinical AI Engine shutting down...")

# Create FastAPI app
app = FastAPI(
    title="Med-AGI Clinical AI Service",
    description="Advanced clinical decision support and medical intelligence",
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


# ============================================================================
# API Request/Response Models
# ============================================================================

class SymptomRequest(BaseModel):
    name: str
    severity: str = Field(default="moderate", regex="^(mild|moderate|severe)$")
    duration: str = Field(default="acute", regex="^(acute|subacute|chronic)$")
    onset: str = Field(default="sudden")
    associated_symptoms: List[str] = Field(default_factory=list)


class VitalSignsRequest(BaseModel):
    heart_rate: int = Field(ge=30, le=250)
    blood_pressure_systolic: int = Field(ge=50, le=300)
    blood_pressure_diastolic: int = Field(ge=30, le=200)
    temperature: float = Field(ge=90, le=110)
    respiratory_rate: int = Field(ge=5, le=60)
    oxygen_saturation: int = Field(ge=50, le=100)


class PatientAnalysisRequest(BaseModel):
    symptoms: List[SymptomRequest]
    vitals: VitalSignsRequest
    medical_history: Dict[str, Any] = Field(default_factory=dict)
    lab_results: Optional[Dict[str, Any]] = None


class DiagnosisResponse(BaseModel):
    condition: str
    probability: float
    icd10_code: str
    evidence: List[str]
    differential_diagnoses: List[str]
    urgency: str
    confidence_level: str


class TreatmentResponse(BaseModel):
    medication: str
    dosage: str
    route: str
    frequency: str
    duration: str
    contraindications: List[str]
    interactions: List[str]
    monitoring_required: List[str]


class ClinicalDecisionResponse(BaseModel):
    diagnosis: DiagnosisResponse
    treatments: List[TreatmentResponse]
    diagnostic_tests: List[str]
    referrals: List[str]
    follow_up: str
    clinical_pearls: List[str]
    risk_factors: List[str]
    prognosis: str


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "clinical-ai",
        "timestamp": datetime.now().isoformat()
    }


@app.post("/v1/clinical/analyze", response_model=ClinicalDecisionResponse)
async def analyze_patient(request: PatientAnalysisRequest):
    """
    Analyze patient data and generate clinical decision support
    """
    try:
        # Convert request models to dataclasses
        symptoms = [
            Symptom(
                name=s.name,
                severity=s.severity,
                duration=s.duration,
                onset=s.onset,
                associated_symptoms=s.associated_symptoms
            )
            for s in request.symptoms
        ]
        
        vitals = VitalSigns(
            heart_rate=request.vitals.heart_rate,
            blood_pressure_systolic=request.vitals.blood_pressure_systolic,
            blood_pressure_diastolic=request.vitals.blood_pressure_diastolic,
            temperature=request.vitals.temperature,
            respiratory_rate=request.vitals.respiratory_rate,
            oxygen_saturation=request.vitals.oxygen_saturation
        )
        
        # Analyze patient
        decision = await clinical_ai.analyze_patient(
            symptoms=symptoms,
            vitals=vitals,
            medical_history=request.medical_history,
            lab_results=request.lab_results
        )
        
        # Convert to response model
        response = ClinicalDecisionResponse(
            diagnosis=DiagnosisResponse(
                condition=decision.diagnosis.condition,
                probability=decision.diagnosis.probability,
                icd10_code=decision.diagnosis.icd10_code,
                evidence=decision.diagnosis.evidence,
                differential_diagnoses=decision.diagnosis.differential_diagnoses,
                urgency=decision.diagnosis.urgency.value,
                confidence_level=decision.diagnosis.confidence_level.value
            ),
            treatments=[
                TreatmentResponse(
                    medication=t.medication,
                    dosage=t.dosage,
                    route=t.route,
                    frequency=t.frequency,
                    duration=t.duration,
                    contraindications=t.contraindications,
                    interactions=t.interactions,
                    monitoring_required=t.monitoring_required
                )
                for t in decision.treatments
            ],
            diagnostic_tests=decision.diagnostic_tests,
            referrals=decision.referrals,
            follow_up=decision.follow_up,
            clinical_pearls=decision.clinical_pearls,
            risk_factors=decision.risk_factors,
            prognosis=decision.prognosis
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error analyzing patient: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}"
        )


@app.post("/v1/clinical/differential")
async def generate_differential(symptoms: List[str]):
    """
    Generate differential diagnosis from symptoms
    """
    try:
        # Find related conditions
        conditions = clinical_ai.knowledge_graph.find_related_conditions(symptoms)
        
        return {
            "differentials": [
                {
                    "condition": cond.replace("_", " ").title(),
                    "probability": prob,
                    "supporting_symptoms": symptoms
                }
                for cond, prob in conditions[:10]
            ]
        }
        
    except Exception as e:
        logger.error(f"Error generating differential: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Differential generation failed: {str(e)}"
        )


@app.post("/v1/clinical/risk-assessment")
async def assess_risk(
    vitals: VitalSignsRequest,
    medical_history: Dict[str, Any]
):
    """
    Assess patient risk level
    """
    try:
        vitals_obj = VitalSigns(
            heart_rate=vitals.heart_rate,
            blood_pressure_systolic=vitals.blood_pressure_systolic,
            blood_pressure_diastolic=vitals.blood_pressure_diastolic,
            temperature=vitals.temperature,
            respiratory_rate=vitals.respiratory_rate,
            oxygen_saturation=vitals.oxygen_saturation
        )
        
        # Calculate risk score
        risk_score = 0
        risk_factors = []
        
        if vitals_obj.is_critical():
            risk_score += 50
            risk_factors.append("Critical vital signs")
        
        # Age risk
        age = medical_history.get("age", 0)
        if age > 65:
            risk_score += 10
            risk_factors.append(f"Age {age} years")
        
        # Comorbidities
        if medical_history.get("diabetes"):
            risk_score += 15
            risk_factors.append("Diabetes")
        
        if medical_history.get("hypertension"):
            risk_score += 10
            risk_factors.append("Hypertension")
        
        if medical_history.get("heart_disease"):
            risk_score += 20
            risk_factors.append("Heart disease")
        
        # Determine risk level
        if risk_score >= 50:
            risk_level = "HIGH"
        elif risk_score >= 25:
            risk_level = "MODERATE"
        else:
            risk_level = "LOW"
        
        return {
            "risk_level": risk_level,
            "risk_score": risk_score,
            "risk_factors": risk_factors,
            "recommendations": [
                "Close monitoring required" if risk_level == "HIGH" else "Routine monitoring",
                "Consider admission" if risk_score >= 50 else "Outpatient management appropriate"
            ]
        }
        
    except Exception as e:
        logger.error(f"Error assessing risk: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Risk assessment failed: {str(e)}"
        )


@app.get("/v1/clinical/treatment-options/{condition}")
async def get_treatment_options(condition: str):
    """
    Get treatment options for a specific condition
    """
    try:
        condition_key = condition.lower().replace(" ", "_")
        treatments = clinical_ai.knowledge_graph.get_treatment_plan(condition_key)
        
        if not treatments:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No treatment options found for {condition}"
            )
        
        # Get detailed treatment information
        detailed_treatments = []
        for treatment_name in treatments:
            treatment = clinical_ai._create_treatment(treatment_name, {})
            detailed_treatments.append({
                "medication": treatment.medication,
                "dosage": treatment.dosage,
                "route": treatment.route,
                "frequency": treatment.frequency,
                "duration": treatment.duration,
                "contraindications": treatment.contraindications,
                "interactions": treatment.interactions,
                "monitoring": treatment.monitoring_required
            })
        
        return {
            "condition": condition,
            "treatments": detailed_treatments,
            "guidelines": f"Follow current clinical guidelines for {condition}",
            "monitoring": "Regular follow-up recommended"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting treatment options: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get treatment options: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8020)
