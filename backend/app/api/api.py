from fastapi import APIRouter

from app.api.endpoints import auth, admin, officer, cases, documents, analysis, rag_query

api_router = APIRouter()
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(admin.router, prefix="/admin", tags=["admin"])
api_router.include_router(officer.router, prefix="/officer", tags=["officer"])
api_router.include_router(cases.router, prefix="/cases", tags=["cases"])
# documents, analysis, and rag_query are all mounted on /cases
api_router.include_router(documents.router, prefix="/cases", tags=["documents"])
api_router.include_router(analysis.router, prefix="/cases", tags=["analysis"])
api_router.include_router(rag_query.router, prefix="/cases", tags=["rag"])
