import os
import secrets
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Optional, List

from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, ForeignKey, Text, func, desc
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from pydantic import BaseModel, EmailStr
from pydantic_settings import BaseSettings
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql://user:pass@localhost:5432/benchgr"
    SECRET_KEY: str = "dev_secret_change_in_production"
    ENVIRONMENT: str = "development"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 10080
    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()

settings = get_settings()
DATABASE_URL = settings.DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class User(Base):
    __tablename__ = "users"
    id            = Column(Integer, primary_key=True, index=True)
    username      = Column(String(50), unique=True, index=True, nullable=False)
    email         = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    api_key       = Column(String(64), unique=True, index=True, default=lambda: secrets.token_hex(32))
    created_at    = Column(DateTime(timezone=True), server_default=func.now())
    results       = relationship("BenchmarkResult", back_populates="user")

class BenchmarkResult(Base):
    __tablename__ = "benchmark_results"
    id             = Column(Integer, primary_key=True, index=True)
    user_id        = Column(Integer, ForeignKey("users.id"), nullable=False)
    gpu_name       = Column(String(100), nullable=False)
    gpu_arch       = Column(String(100))
    vram_gb        = Column(Integer)
    driver_version = Column(String(50))
    cuda_version   = Column(String(50))
    tokens_per_sec = Column(Float)
    images_per_sec = Column(Float)
    tflops_fp16    = Column(Float)
    memory_bw_gbps = Column(Float)
    neural_score   = Column(Float)
    agent_version  = Column(String(20))
    submitted_at   = Column(DateTime(timezone=True), server_default=func.now())
    notes          = Column(Text, nullable=True)
    user           = relationship("User", back_populates="results")

Base.metadata.create_all(bind=engine)

ALGORITHM = "HS256"
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
bearer_scheme = HTTPBearer()

def hash_password(p): return pwd_context.hash(p)
def verify_password(p, h): return pwd_context.verify(p, h)

def create_access_token(data, expires_delta=None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme), db: Session = Depends(get_db)):
    try:
        payload = jwt.decode(credentials.credentials, settings.SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        if not user_id: raise HTTPException(status_code=401, detail="Invalid token")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    user = db.query(User).filter(User.id == int(user_id)).first()
    if not user: raise HTTPException(status_code=401, detail="User not found")
    return user

def get_user_by_api_key(api_key, db):
    user = db.query(User).filter(User.api_key == api_key).first()
    if not user: raise HTTPException(status_code=401, detail="Invalid API key")
    return user

def compute_neural_score(tokens, images, tflops, membw):
    REF = {"tokens": 150, "images": 12, "tflops": 82.6, "membw": 1008}
    W = {"tokens": 0.40, "images": 0.25, "tflops": 0.25, "membw": 0.10}
    s = w = 0.0
    for k, v in [("tokens", tokens), ("images", images), ("tflops", tflops), ("membw", membw)]:
        if v is not None: s += W[k] * (v / REF[k]); w += W[k]
    return round((s / w) * 10000, 1) if w else 0.0

class RegisterRequest(BaseModel):
    username: str
    email: EmailStr
    password: str

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    id: int; username: str; email: str; api_key: str
    class Config: from_attributes = True

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

class LeaderboardEntry(BaseModel):
    rank: int; id: int; gpu_name: str; gpu_arch: Optional[str]; vram_gb: Optional[int]
    tokens_per_sec: Optional[float]; images_per_sec: Optional[float]
    tflops_fp16: Optional[float]; memory_bw_gbps: Optional[float]
    neural_score: Optional[float]; submitted_at: datetime; username: str

app = FastAPI(title="BenchGR API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/")
def root(): return {"status": "ok", "service": "BenchGR API v1.0"}

@app.get("/health")
def health(): return {"status": "healthy"}

@app.post("/api/auth/register")
def register(body: RegisterRequest, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == body.email).first():
        raise HTTPException(400, "Email already registered")
    if db.query(User).filter(User.username == body.username).first():
        raise HTTPException(400, "Username taken")
    user = User(username=body.username, email=body.email, password_hash=hash_password(body.password))
    db.add(user); db.commit(); db.refresh(user)
    return {"access_token": create_access_token({"sub": str(user.id)}), "token_type": "bearer"}

@app.post("/api/auth/login")
def login(body: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == body.email).first()
    if not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(401, "Invalid credentials")
    return {"access_token": create_access_token({"sub": str(user.id)}), "token_type": "bearer"}

@app.get("/api/auth/me", response_model=UserResponse)
def me(current_user: User = Depends(get_current_user)): return current_user

@app.post("/api/auth/rotate-key", response_model=UserResponse)
def rotate_key(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    current_user.api_key = secrets.token_hex(32)
    db.commit(); db.refresh(current_user); return current_user

@app.post("/api/results/submit")
def submit_result(body: SubmitResultRequest, api_key: str = Query(...), db: Session = Depends(get_db)):
    user = get_user_by_api_key(api_key, db)
    score = compute_neural_score(body.tokens_per_sec, body.images_per_sec, body.tflops_fp16, body.memory_bw_gbps)
    r = BenchmarkResult(user_id=user.id, gpu_name=body.gpu_name, gpu_arch=body.gpu_arch,
        vram_gb=body.vram_gb, driver_version=body.driver_version, cuda_version=body.cuda_version,
        tokens_per_sec=body.tokens_per_sec, images_per_sec=body.images_per_sec,
        tflops_fp16=body.tflops_fp16, memory_bw_gbps=body.memory_bw_gbps,
        neural_score=score, agent_version=body.agent_version, notes=body.notes)
    db.add(r); db.commit(); db.refresh(r)
    return {"message": "Result submitted", "neural_score": score, "result_id": r.id}

@app.get("/api/results/leaderboard", response_model=List[LeaderboardEntry])
def leaderboard(sort_by: str = Query("neural_score"), search: Optional[str] = Query(None),
    limit: int = Query(50, le=200), offset: int = Query(0), db: Session = Depends(get_db)):
    sort_col = getattr(BenchmarkResult, sort_by, BenchmarkResult.neural_score)
    subq = (db.query(BenchmarkResult.gpu_name, func.max(BenchmarkResult.neural_score).label("best"))
        .group_by(BenchmarkResult.gpu_name).subquery())
    q = (db.query(BenchmarkResult, User.username).join(User, User.id == BenchmarkResult.user_id)
        .join(subq, (BenchmarkResult.gpu_name == subq.c.gpu_name) & (BenchmarkResult.neural_score == subq.c.best)))
    if search: q = q.filter(BenchmarkResult.gpu_name.ilike(f"%{search}%"))
    rows = q.order_by(desc(sort_col)).offset(offset).limit(limit).all()
    return [LeaderboardEntry(rank=i+offset+1, id=r.id, gpu_name=r.gpu_name, gpu_arch=r.gpu_arch,
        vram_gb=r.vram_gb, tokens_per_sec=r.tokens_per_sec, images_per_sec=r.images_per_sec,
        tflops_fp16=r.tflops_fp16, memory_bw_gbps=r.memory_bw_gbps, neural_score=r.neural_score,
        submitted_at=r.submitted_at, username=u) for i, (r, u) in enumerate(rows)]

@app.get("/api/results/my-submissions")
def my_submissions(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return db.query(BenchmarkResult).filter(BenchmarkResult.user_id == current_user.id).order_by(desc(BenchmarkResult.submitted_at)).limit(20).all()

@app.get("/api/results/stats")
def stats(db: Session = Depends(get_db)):
    return {"total_submissions": db.query(func.count(BenchmarkResult.id)).scalar() or 0,
        "total_gpu_models": db.query(func.count(func.distinct(BenchmarkResult.gpu_name))).scalar() or 0,
        "total_users": db.query(func.count(User.id)).scalar() or 0,
        "best_inference_tps": db.query(func.max(BenchmarkResult.tokens_per_sec)).scalar(),
        "best_image_ips": db.query(func.max(BenchmarkResult.images_per_sec)).scalar(),
        "best_tflops": db.query(func.max(BenchmarkResult.tflops_fp16)).scalar(),
        "best_membw_gbps": db.query(func.max(BenchmarkResult.memory_bw_gbps)).scalar()}
