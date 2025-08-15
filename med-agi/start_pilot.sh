#!/bin/bash
#
# Med-AGI Pilot Startup Script
# Quick launcher for the bootstrap process
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default profile
PROFILE=${1:-dev}

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}    Med-AGI System Startup${NC}"
echo -e "${GREEN}    Profile: $PROFILE${NC}"
echo -e "${GREEN}========================================${NC}"

# Check for .env file
if [ ! -f .env ]; then
    echo -e "${YELLOW}Warning: .env file not found${NC}"
    
    if [ -f .env.example ]; then
        echo "Creating .env from .env.example..."
        cp .env.example .env
        echo -e "${YELLOW}Please edit .env with your configuration${NC}"
    fi
fi

# Load environment variables
if [ -f .env ]; then
    echo "Loading environment variables..."
    set -a
    source .env
    set +a
fi

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is not installed${NC}"
    exit 1
fi

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed${NC}"
    exit 1
fi

# Run bootstrap
echo -e "\n${GREEN}Starting bootstrap process...${NC}\n"

python3 bootstrap.py \
    --profile "$PROFILE" \
    --imaging "${IMAGING_URL:-http://localhost:8006}" \
    --ekg "${EKG_URL:-http://localhost:8016}" \
    --eval "${EVAL_URL:-http://localhost:8005}" \
    --anchor "${ANCHOR_URL:-http://localhost:8007}" \
    --modelcards "${MODELCARDS_URL:-http://localhost:8008}" \
    --ops "${OPS_URL:-http://localhost:8010}"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo -e "\n${GREEN}========================================${NC}"
    echo -e "${GREEN}    System started successfully!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "Service URLs:"
    echo "  Imaging:     ${IMAGING_URL:-http://localhost:8006}"
    echo "  EKG:         ${EKG_URL:-http://localhost:8016}"
    echo "  Eval:        ${EVAL_URL:-http://localhost:8005}"
    echo "  Anchor:      ${ANCHOR_URL:-http://localhost:8007}"
    echo "  ModelCards:  ${MODELCARDS_URL:-http://localhost:8008}"
    echo "  Ops:         ${OPS_URL:-http://localhost:8010}"
    echo ""
    echo "Check pilot_bootstrap_report.json for detailed status"
else
    echo -e "\n${RED}========================================${NC}"
    echo -e "${RED}    System startup failed!${NC}"
    echo -e "${RED}========================================${NC}"
    echo ""
    echo "Check pilot_bootstrap_report.json for details"
    exit $EXIT_CODE
fi