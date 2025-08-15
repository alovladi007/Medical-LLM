"""
OPA (Open Policy Agent) client for authorization
"""

import os
import logging
from typing import Dict, Any
import httpx

logger = logging.getLogger(__name__)


async def check_permission(
    user: Dict[str, Any],
    action: str,
    resource: Dict[str, Any]
) -> bool:
    """Check if user has permission for action on resource"""
    
    # Dev mode bypass
    if user.get("dev_mode") or os.getenv("SIEM_MODE") == "dev":
        return True
    
    # Admin bypass
    if "admin" in user.get("roles", []):
        return True
    
    opa_url = os.getenv("OPA_URL")
    if not opa_url:
        logger.warning("OPA_URL not configured, allowing access")
        return True
    
    try:
        # Prepare OPA input
        opa_input = {
            "input": {
                "user": user,
                "action": action,
                "resource": resource
            }
        }
        
        # Query OPA
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{opa_url}/v1/data/medagi/allow",
                json=opa_input,
                timeout=5.0
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("result", False)
            else:
                logger.error(f"OPA returned status {response.status_code}")
                return False
                
    except Exception as e:
        logger.error(f"OPA check failed: {e}")
        # Fail open in dev, fail closed in prod
        return os.getenv("SIEM_MODE") == "dev"