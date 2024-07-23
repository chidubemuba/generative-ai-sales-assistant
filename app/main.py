from fastapi import FastAPI
from app.routers.search.search import router as search_router


def create_app() -> FastAPI:
    _app = FastAPI(title="sl-vista-back")
    _app.include_router(search_router)
    return _app


app = create_app()


# Healthcheck and Readiness check
@app.get("/admin/healthcheck", status_code=200, include_in_schema=False)
async def healthcheck():
    return "sl-vista-backed is ready to go!"


@app.get("/admin/readiness", status_code=400)
async def readiness():
    return {"status": "ok"}


@app.get("/")
async def hello():
    return "Hello from sl-vista-backed"

import uvicorn

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8080)
