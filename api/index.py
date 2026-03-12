from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum

app = FastAPI(
    title="BenchGR API",
    description="GPU Benchmark Leaderboard — REST API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy import so DB errors don't crash the entire cold start
try:
    from core.database import Base, engine
    from routers import auth, results
    Base.metadata.create_all(bind=engine)
    app.include_router(auth.router, prefix="/api")
    app.include_router(results.router, prefix="/api")
    DB_OK = True
except Exception as e:
    DB_OK = False
    DB_ERROR = str(e)


@app.get("/")
def root():
    return {"status": "ok", "service": "BenchGR API v1.0", "db": DB_OK}


@app.get("/health")
def health():
    if not DB_OK:
        return {"status": "db_error", "detail": DB_ERROR}
    return {"status": "healthy"}


# Mangum wraps FastAPI for AWS Lambda / Vercel serverless
handler = Mangum(app, lifespan="off")
