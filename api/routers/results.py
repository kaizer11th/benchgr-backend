from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
from core.database import get_db
from core.auth import get_current_user
from models.user import User, BenchmarkResult

router = APIRouter(prefix="/results", tags=["results"])


# ── Pydantic schemas ──────────────────────────────────────

class SubmitResultRequest(BaseModel):
    gpu_name: str
    gpu_arch: Optional[str] = None
    vram_gb: Optional[int] = None
    driver_version: Optional[str] = None
    cuda_version: Optional[str] = None
    tokens_per_sec: Optional[float] = None
    images_per_sec: Optional[float] = None
    tflops_fp16: Optional[float] = None
    memory_bw_gbps: Optional[float] = None
    agent_version: Optional[str] = None
    notes: Optional[str] = None


class ResultResponse(BaseModel):
    id: int
    gpu_name: str
    gpu_arch: Optional[str]
    vram_gb: Optional[int]
    tokens_per_sec: Optional[float]
    images_per_sec: Optional[float]
    tflops_fp16: Optional[float]
    memory_bw_gbps: Optional[float]
    neural_score: Optional[float]
    submitted_at: datetime
    username: str

    class Config:
        from_attributes = True


class LeaderboardEntry(BaseModel):
    rank: int
    id: int
    gpu_name: str
    gpu_arch: Optional[str]
    vram_gb: Optional[int]
    tokens_per_sec: Optional[float]
    images_per_sec: Optional[float]
    tflops_fp16: Optional[float]
    memory_bw_gbps: Optional[float]
    neural_score: Optional[float]
    submitted_at: datetime
    username: str


# ── Score computation ─────────────────────────────────────

def compute_neural_score(
    tokens: Optional[float],
    images: Optional[float],
    tflops: Optional[float],
    membw: Optional[float],
) -> float:
    """
    Weighted composite score across all 4 benchmark categories.
    Weights: inference 40%, image gen 25%, cuda 25%, memory 10%
    Normalized against reference values (RTX 4090 baseline).
    """
    REF = {"tokens": 150, "images": 12, "tflops": 82.6, "membw": 1008}
    WEIGHTS = {"tokens": 0.40, "images": 0.25, "tflops": 0.25, "membw": 0.10}
    BASE_SCORE = 10000  # RTX 4090 baseline score

    score = 0.0
    total_weight = 0.0

    if tokens is not None:
        score += WEIGHTS["tokens"] * (tokens / REF["tokens"])
        total_weight += WEIGHTS["tokens"]
    if images is not None:
        score += WEIGHTS["images"] * (images / REF["images"])
        total_weight += WEIGHTS["images"]
    if tflops is not None:
        score += WEIGHTS["tflops"] * (tflops / REF["tflops"])
        total_weight += WEIGHTS["tflops"]
    if membw is not None:
        score += WEIGHTS["membw"] * (membw / REF["membw"])
        total_weight += WEIGHTS["membw"]

    if total_weight == 0:
        return 0.0

    normalized = score / total_weight
    return round(normalized * BASE_SCORE, 1)


# ── API Key auth for agent submissions ───────────────────

def get_user_by_api_key(api_key: str, db: Session) -> User:
    user = db.query(User).filter(User.api_key == api_key).first()
    if not user:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return user


# ── Endpoints ─────────────────────────────────────────────

@router.post("/submit")
def submit_result(
    body: SubmitResultRequest,
    api_key: str = Query(..., description="Your BenchGR API key"),
    db: Session = Depends(get_db),
):
    """Agent submits benchmark results using API key auth."""
    user = get_user_by_api_key(api_key, db)

    neural_score = compute_neural_score(
        body.tokens_per_sec,
        body.images_per_sec,
        body.tflops_fp16,
        body.memory_bw_gbps,
    )

    result = BenchmarkResult(
        user_id=user.id,
        gpu_name=body.gpu_name,
        gpu_arch=body.gpu_arch,
        vram_gb=body.vram_gb,
        driver_version=body.driver_version,
        cuda_version=body.cuda_version,
        tokens_per_sec=body.tokens_per_sec,
        images_per_sec=body.images_per_sec,
        tflops_fp16=body.tflops_fp16,
        memory_bw_gbps=body.memory_bw_gbps,
        neural_score=neural_score,
        agent_version=body.agent_version,
        notes=body.notes,
    )

    db.add(result)
    db.commit()
    db.refresh(result)

    return {"message": "Result submitted", "neural_score": neural_score, "result_id": result.id}


@router.get("/leaderboard", response_model=List[LeaderboardEntry])
def leaderboard(
    sort_by: str = Query("neural_score", enum=["neural_score", "tokens_per_sec", "images_per_sec", "tflops_fp16", "memory_bw_gbps"]),
    gen: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    limit: int = Query(50, le=200),
    offset: int = Query(0),
    db: Session = Depends(get_db),
):
    """
    Returns best result per GPU model (not per submission).
    Only the top result per gpu_name appears on the board.
    """
    sort_col = getattr(BenchmarkResult, sort_by)

    # Subquery: best score per gpu_name
    subq = (
        db.query(
            BenchmarkResult.gpu_name,
            func.max(BenchmarkResult.neural_score).label("best_score"),
        )
        .group_by(BenchmarkResult.gpu_name)
        .subquery()
    )

    query = (
        db.query(BenchmarkResult, User.username)
        .join(User, User.id == BenchmarkResult.user_id)
        .join(subq, (BenchmarkResult.gpu_name == subq.c.gpu_name) & (BenchmarkResult.neural_score == subq.c.best_score))
    )

    if search:
        query = query.filter(BenchmarkResult.gpu_name.ilike(f"%{search}%"))

    total = query.count()
    rows = query.order_by(desc(sort_col)).offset(offset).limit(limit).all()

    entries = []
    for rank, (result, username) in enumerate(rows, start=offset + 1):
        entries.append(LeaderboardEntry(
            rank=rank,
            id=result.id,
            gpu_name=result.gpu_name,
            gpu_arch=result.gpu_arch,
            vram_gb=result.vram_gb,
            tokens_per_sec=result.tokens_per_sec,
            images_per_sec=result.images_per_sec,
            tflops_fp16=result.tflops_fp16,
            memory_bw_gbps=result.memory_bw_gbps,
            neural_score=result.neural_score,
            submitted_at=result.submitted_at,
            username=username,
        ))

    return entries


@router.get("/my-submissions")
def my_submissions(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    results = (
        db.query(BenchmarkResult)
        .filter(BenchmarkResult.user_id == current_user.id)
        .order_by(desc(BenchmarkResult.submitted_at))
        .limit(20)
        .all()
    )
    return results


@router.get("/stats")
def platform_stats(db: Session = Depends(get_db)):
    """Summary stats for the hero KPI strip."""
    total_submissions = db.query(func.count(BenchmarkResult.id)).scalar()
    total_gpu_models  = db.query(func.count(func.distinct(BenchmarkResult.gpu_name))).scalar()
    total_users       = db.query(func.count(User.id)).scalar()
    best_inference    = db.query(func.max(BenchmarkResult.tokens_per_sec)).scalar()
    best_image        = db.query(func.max(BenchmarkResult.images_per_sec)).scalar()
    best_tflops       = db.query(func.max(BenchmarkResult.tflops_fp16)).scalar()
    best_membw        = db.query(func.max(BenchmarkResult.memory_bw_gbps)).scalar()

    return {
        "total_submissions": total_submissions or 0,
        "total_gpu_models": total_gpu_models or 0,
        "total_users": total_users or 0,
        "best_inference_tps": best_inference,
        "best_image_ips": best_image,
        "best_tflops": best_tflops,
        "best_membw_gbps": best_membw,
    }
