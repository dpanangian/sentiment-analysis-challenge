import os

import uvicorn
from fastapi import FastAPI

from app.router import router as api_router


def get_application() -> FastAPI:
    application = FastAPI(
        title="Sentiment Analysis API",
    )

    application.include_router(api_router)

    return application

app = get_application()

#if __name__ == "__main__":
#    uvicorn.run(app, host="0.0.0.0", port=8000)
