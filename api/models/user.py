from sqlalchemy import Column, Integer, String, DateTime, Float, ForeignKey, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from core.database import Base
import secrets


class User(Base):
    __tablename__ = "users"

    id           = Column(Integer, primary_key=True, index=True)
    username     = Column(String(50), unique=True, index=True, nullable=False)
    email        = Column(String(255), unique=True, index=True, nullable=False)
    password_hash= Column(String(255), nullable=False)
    api_key      = Column(String(64), unique=True, index=True, default=lambda: secrets.token_hex(32))
    created_at   = Column(DateTime(timezone=True), server_default=func.now())

    results = relationship("BenchmarkResult", back_populates="user")


class BenchmarkResult(Base):
    __tablename__ = "benchmark_results"

    id           = Column(Integer, primary_key=True, index=True)
    user_id      = Column(Integer, ForeignKey("users.id"), nullable=False)

    # GPU info
    gpu_name     = Column(String(100), nullable=False)
    gpu_arch     = Column(String(100))
    vram_gb      = Column(Integer)
    driver_version = Column(String(50))
    cuda_version = Column(String(50))

    # Benchmark scores
    tokens_per_sec  = Column(Float)   # AI Inference
    images_per_sec  = Column(Float)   # Image Generation
    tflops_fp16     = Column(Float)   # CUDA Tensor ops
    memory_bw_gbps  = Column(Float)   # Memory bandwidth

    # Computed overall score
    neural_score = Column(Float)

    # Meta
    agent_version = Column(String(20))
    submitted_at  = Column(DateTime(timezone=True), server_default=func.now())
    notes         = Column(Text, nullable=True)

    user = relationship("User", back_populates="results")
