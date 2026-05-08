from fastapi import APIRouter
from ..services import data_service

router = APIRouter(prefix="/api/eda", tags=["EDA"])


@router.get("/summary")
def summary():
    return data_service.get_summary()


@router.get("/preview")
def preview():
    return data_service.get_preview()


@router.get("/distribution")
def distribution():
    return data_service.get_target_distribution()


@router.get("/boxplot")
def boxplot():
    return data_service.get_gpa_by_target()


@router.get("/correlation")
def correlation():
    return data_service.get_correlation_matrix()
