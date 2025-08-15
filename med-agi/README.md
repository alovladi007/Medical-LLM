# Med-AGI System v16 - Pilot Ready

A comprehensive medical AI gateway system with GPU-accelerated inference, authentication, and monitoring capabilities.

## System Architecture

The Med-AGI system consists of multiple microservices:
- **Imaging Service**: Handles medical imaging with Triton GPU acceleration
- **EKG Service**: Processes EKG waveform data with uncertainty quantification  
- **Eval Service**: Model evaluation and metrics tracking
- **Anchor Service**: Medical record anchoring and search
- **ModelCards Service**: Model documentation and versioning
- **Ops Service**: SIEM integration and operational monitoring

## Quick Start

### Prerequisites
- Docker and Docker Compose
- Python 3.11+
- Node.js 18+ (for UI components)
- NVIDIA GPU with CUDA support (optional, for Triton acceleration)

### Environment Setup
```bash
# Copy environment template
cp .env.example .env

# Edit .env with your configuration
# Required variables:
# - OIDC_ISSUER, OIDC_AUDIENCE, OIDC_JWKS_URL (for authentication)
# - OPA_URL (for authorization)
# - SIEM_URL, SIEM_MODE (for monitoring)
# - DICOMWEB_BASE, TRITON_URL (for imaging)
```

### Running the System

#### Development Mode
```bash
# Start all services in dev mode
./start_pilot.sh dev

# Or use bootstrap directly
python3 bootstrap.py --profile dev
```

#### Production Mode
```bash
# Start with production profile
./start_pilot.sh prod

# Run with custom endpoints
python3 bootstrap.py --profile prod \
  --imaging http://imaging.example.com \
  --eval http://eval.example.com \
  --anchor http://anchor.example.com \
  --modelcards http://modelcards.example.com \
  --ops http://ops.example.com
```

### Health Checks
```bash
# Run smoke tests
python3 smoke/smoke.py --imaging http://localhost:8006

# Full system validation
python3 tests/test_endpoints.py
```

### Load Testing
```bash
# Test CXR endpoint
k6 run scripts/k6/cxr.js -e IMAGING=http://localhost:8006

# Test EKG endpoint  
k6 run scripts/k6/ekg.js -e EKG=http://localhost:8016

# Test DICOM endpoint
k6 run scripts/k6/dicom.js -e IMAGING=http://localhost:8006
```

### Chaos Engineering
```bash
# Test Triton failover
IMAGING=http://localhost:8006 bash scripts/chaos_triton_down.sh
```

## Service Endpoints

| Service | Port | Health Check | Main API |
|---------|------|--------------|----------|
| Imaging | 8006 | /v1/triton/health | /v1/imaging/infer |
| EKG | 8016 | /v1/ekg/health | /v1/ekg/infer |
| Eval | 8005 | /v1/eval/health | /v1/eval/summary |
| Anchor | 8007 | /v1/anchor/health | /v1/anchor_search |
| ModelCards | 8008 | /v1/modelcards/health | /v1/modelcards/list |
| Ops | 8010 | /v1/ops/health | /v1/ops/siem/stats |

## UI Components

The system includes React/Next.js UI components:
- `StatusWidget`: Real-time system status indicator
- `EKG Page`: Interactive EKG inference interface
- `ModelCard Links`: Model documentation browser

## Security

- JWT-based authentication via OIDC
- OPA policy-based authorization
- PHI data encryption at rest and in transit
- Audit logging via SIEM integration

## Monitoring

- Prometheus metrics exposed on /metrics
- SIEM integration for security events
- Alert policies for error rates and latency
- Data retention policies compliant with regulations

## Documentation

- [RUNBOOK.md](./RUNBOOK.md) - Operational procedures
- [PILOT_CHECKLIST.md](./PILOT_CHECKLIST.md) - Pre-deployment checklist
- [docs/DATA_RETENTION.md](./docs/DATA_RETENTION.md) - Data retention policies
- [docs/ARCHITECTURE.md](./docs/ARCHITECTURE.md) - System architecture details

## License

Proprietary - All rights reserved