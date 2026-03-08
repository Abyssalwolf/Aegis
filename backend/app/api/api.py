from fastapi import APIRouter

from app.api.endpoints import auth, admin, officer, cases, documents, analysis

api_router = APIRouter()
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(admin.router, prefix="/admin", tags=["admin"])
api_router.include_router(officer.router, prefix="/officer", tags=["officer"])
api_router.include_router(cases.router, prefix="/cases", tags=["cases"])
# the documents and analysis modules are mounted on /cases in the requirements
api_router.include_router(documents.router, prefix="/cases", tags=["documents"])
api_router.include_router(analysis.router, prefix="/cases", tags=["analysis"])
