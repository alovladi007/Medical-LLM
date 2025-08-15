"""
Authentication module for JWT verification
"""

import os
import logging
from typing import Optional, Dict, Any
from fastapi import HTTPException, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError
import httpx

logger = logging.getLogger(__name__)

# Security scheme
security = HTTPBearer(auto_error=False)

# Cache for JWKS
_jwks_cache = None


async def get_jwks():
    """Fetch JWKS from OIDC provider"""
    global _jwks_cache
    
    if _jwks_cache:
        return _jwks_cache
    
    jwks_url = os.getenv("OIDC_JWKS_URL")
    if not jwks_url:
        logger.warning("OIDC_JWKS_URL not configured")
        return None
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(jwks_url)
            response.raise_for_status()
            _jwks_cache = response.json()
            return _jwks_cache
    except Exception as e:
        logger.error(f"Failed to fetch JWKS: {e}")
        return None


async def verify_jwt(credentials: HTTPAuthorizationCredentials = Security(security)) -> Dict[str, Any]:
    """Verify JWT token"""
    # Dev mode bypass
    if os.getenv("SIEM_MODE") == "dev" and not os.getenv("OIDC_ISSUER"):
        return {
            "sub": "dev-user",
            "email": "dev@example.com",
            "roles": ["admin"],
            "dev_mode": True
        }
    
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header missing",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token = credentials.credentials
    
    try:
        # Get JWKS
        jwks = await get_jwks()
        if not jwks:
            # Fallback to simple decode in dev mode
            if os.getenv("SIEM_MODE") == "dev":
                payload = jwt.decode(
                    token,
                    options={"verify_signature": False}
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="JWKS not available"
                )
        else:
            # Verify token with JWKS
            unverified_header = jwt.get_unverified_header(token)
            rsa_key = {}
            
            for key in jwks.get("keys", []):
                if key["kid"] == unverified_header["kid"]:
                    rsa_key = {
                        "kty": key["kty"],
                        "kid": key["kid"],
                        "use": key["use"],
                        "n": key["n"],
                        "e": key["e"]
                    }
                    break
            
            if not rsa_key:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Unable to find appropriate key"
                )
            
            payload = jwt.decode(
                token,
                rsa_key,
                algorithms=["RS256"],
                audience=os.getenv("OIDC_AUDIENCE"),
                issuer=os.getenv("OIDC_ISSUER")
            )
        
        return payload
        
    except JWTError as e:
        logger.error(f"JWT verification failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user(token_data: Dict = Security(verify_jwt)) -> Dict[str, Any]:
    """Get current user from token"""
    return {
        "id": token_data.get("sub"),
        "email": token_data.get("email"),
        "roles": token_data.get("roles", []),
        "permissions": token_data.get("permissions", []),
        "dev_mode": token_data.get("dev_mode", False)
    }