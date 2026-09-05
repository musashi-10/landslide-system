import uuid
from typing import Dict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.schemas import AuthResponse, LoginRequest, RegisterRequest, UserProfile

router = APIRouter(prefix="/auth", tags=["auth"])

# Mock database
mock_users: Dict[str, dict] = {
    "user@example.com": {
        "password": "password",
        "profile": UserProfile(
            id="USR_001",
            name="Tenzing Sherpa",
            email="user@example.com",
            mobile="+91-9876543210",
            role="user",
            location_id="LOC_001",
        )
    },
    "authority@example.com": {
        "password": "password",
        "profile": UserProfile(
            id="AUTH_001",
            name="District Collector",
            email="authority@example.com",
            mobile="+91-9876543211",
            role="authority",
            location_id=None,
        )
    }
}


@router.post("/login", response_model=AuthResponse)
def login(req: LoginRequest):
    user = mock_users.get(req.email)
    if not user or user["password"] != req.password:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    return AuthResponse(
        token=f"mock_jwt_{uuid.uuid4()}",
        user=user["profile"]
    )


@router.post("/register", response_model=AuthResponse)
def register(req: RegisterRequest):
    if req.email in mock_users:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    new_profile = UserProfile(
        id=f"USR_{uuid.uuid4().hex[:8]}",
        name=req.name,
        email=req.email,
        mobile=req.mobile,
        role="user",
        location_id=req.location_id or "LOC_001"
    )
    
    mock_users[req.email] = {
        "password": req.password,
        "profile": new_profile
    }
    
    return AuthResponse(
        token=f"mock_jwt_{uuid.uuid4()}",
        user=new_profile
    )
