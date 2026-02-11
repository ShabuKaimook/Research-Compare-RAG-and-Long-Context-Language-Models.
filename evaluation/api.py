from fastapi import APIRouter
from evaluation import run_evaluation

router = APIRouter(prefix="/evaluate", tags=["Evaluation"])

