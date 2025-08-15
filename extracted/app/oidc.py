import os, requests
from jose import jwt
from jose.utils import base64url_decode
class OIDC:
    def __init__(self, issuer=None, audience=None, jwks_url=None):
        self.issuer = issuer or os.getenv("OIDC_ISSUER","")
        self.audience = audience or os.getenv("OIDC_AUDIENCE","")
        self.jwks_url = jwks_url or os.getenv("OIDC_JWKS_URL","")
        self.jwks = requests.get(self.jwks_url, timeout=5).json() if self.jwks_url else {"keys":[]}
    def verify(self, token:str):
        header = jwt.get_unverified_header(token); kid = header.get("kid")
        key = next((k for k in self.jwks.get("keys",[]) if k.get("kid")==kid), None)
        if not key: raise ValueError("No JWKS key")
        return jwt.decode(token, key, algorithms=[key.get("alg","RS256")], audience=self.audience, issuer=self.issuer)
