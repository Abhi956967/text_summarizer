from fastapi import FastAPI, Query
import uvicorn
from starlette.responses import RedirectResponse
from fastapi.responses import JSONResponse
from src.text_summarizer.pipeline.predicition_pipeline import PredictionPipeline
import os

app = FastAPI()

@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")


@app.get("/train")
async def training():
    try:
        os.system("python main.py")
        return JSONResponse(content={"message": "Training successful"})
    except Exception as e:
        return JSONResponse(content={"error": str(e)})


@app.post("/predict")
async def predict_route(text: str = Query(...)):
    try:
        obj = PredictionPipeline()
        summary = obj.predict(text)
        return {"summary": summary}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
