from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import orjson

from .routers import resume, description, recommendations


def orjson_dumps(v, *, default):
	return orjson.dumps(v, default=default).decode()


app = FastAPI(title="AI-manvian", version="0.1.0")

# CORS: allow local dev frontends and backends
origins = [
	"http://localhost:3000",
	"http://127.0.0.1:3000",
	"http://localhost:5173",
	"http://127.0.0.1:5173",
	"*",
]

app.add_middleware(
	CORSMiddleware,
	allow_origins=origins,
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)


@app.get("/health")
async def health():
	return {"status": "ok"}


app.include_router(resume.router, prefix="/api")
app.include_router(description.router, prefix="/api")
app.include_router(recommendations.router, prefix="/api")
