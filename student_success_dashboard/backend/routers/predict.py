from fastapi import APIRouter
from pydantic import BaseModel
from ..services import model_service, shap_service, lime_service

router = APIRouter(prefix="/api", tags=["Predict"])


class PredictRequest(BaseModel):
    gender: str
    region: str
    parent_education_level: str
    internet_access_quality: str
    num_courses_enrolled: int
    study_hours_per_week: float
    sleep_hours_avg: float
    extracurricular_count: int
    prev_gpa: float
    assignment_submission_rate: float
    midterm_score: float
    financial_stress_index: int


@router.post("/predict")
def predict(req: PredictRequest):
    input_data = req.model_dump()
    prediction = model_service.predict_student(input_data)

    # Get SHAP explanation for the processed features
    shap_explanation = shap_service.get_shap_for_input(
        prediction["processed_features"], "XGBoost"
    )

    # Get LIME explanation for the processed features
    lime_explanation = lime_service.get_lime_for_input(
        prediction["processed_features"], "XGBoost"
    )

    return {
        **prediction,
        "shap": shap_explanation,
        "lime": lime_explanation,
    }
