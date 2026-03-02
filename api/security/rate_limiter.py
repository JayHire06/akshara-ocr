from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from fastapi import Request

# Define standard limit configurations
UNAUTHENTICATED_LIMIT = "10/minute"
AUTH_FREE_LIMIT = "60/hour"
AUTH_PRO_LIMIT = "1000/hour"

def get_user_or_ip(request: Request) -> str:
    """
    Returns the user ID if authenticated, else IP address.
    Because rate_limiter comes early in request cycle, we check request.state
    or fallback to IP.
    """
    user = getattr(request.state, "user", None)
    if user:
        return str(user.id)
    return get_remote_address(request)

limiter = Limiter(key_func=get_user_or_ip)

def ocr_rate_limit(request: Request) -> str:
    """
    Returns a dynamic rate limit depending on the user's tier.
    """
    user = getattr(request.state, "user", None)
    if not user:
        return UNAUTHENTICATED_LIMIT
    
    # Assume tier is 'pro' if they have a specific attribute, otherwise 'free'
    tier = getattr(user, "tier", "free")
    if tier == "pro":
        return AUTH_PRO_LIMIT
    return AUTH_FREE_LIMIT
