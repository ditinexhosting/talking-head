from fastapi import APIRouter
from controllers.rest import get_root, get_health

router = APIRouter(prefix="/api", tags=["rest"])


@router.get("/")
def root():
    return get_root()


@router.get("/health")
def health():
    return get_health()
