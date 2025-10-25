import os
import base64
from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Form
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
import httpx

app = FastAPI(
    title="Livability Score API",
    description="Predicts livability score and estimated habitable years based on solar radiation and AQI with AI explanation.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models..
class LivabilityResult(BaseModel):
    location_latitude: float
    location_longitude: float
    solar_radiation_score: float
    air_quality_index: float
    calculated_livability_score: str
    estimated_habitable_years: int
    gemini_analysis: str
    data_source: str = Field("Solar Radiation + AQI dataset + Eco Prediction Insight")


# LOGIC
def calculate_livability_score(solar: float, aqi: float) -> (str, int):
    """
    Calculate livability score logically based on AQI and solar radiation.
    Returns (score_label, estimated_habitable_years)
    """
    # Normalize scores
    solar_score = min(max(solar / 5 * 100, 0), 100)  # assuming solar is 0-5
    aqi_score = max(0, 100 - aqi)  # lower AQI = better

    # Weighted combination
    combined_score = 0.5 * solar_score + 0.5 * aqi_score

    # Determine High/Medium/Low
    if combined_score >= 70:
        score_label = "High"
        habitable_years = 80
    elif combined_score >= 40:
        score_label = "Medium"
        habitable_years = 50
    else:
        score_label = "Low"
        habitable_years = 20

    return score_label, habitable_years


# Gemini..
async def ask_gemini_analysis(lat: float, lon: float, solar: float, aqi: float, image_base64: str = None) -> str:
    """
    Ask Gemini AI to explain livability based on given scores.
    Returns textual analysis/recommendations.
    """
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_api_key:
        return "GEMINI_API_KEY not set. Skipping AI analysis."

    prompt_text = (
        f"Location: lat={lat}, lon={lon}\n"
        f"Solar radiation score: {solar}\n"
        f"Air quality index: {aqi}\n"
        "Explain the livability of this location based on these scores, "
        "and suggest interventions or improvements. "
    )
    if image_base64:
        prompt_text += "Analyze this image together with the scores.\n"

    gemini_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    headers = {"Content-Type": "application/json"}
    payload = {"contents": [{"parts": [{"text": prompt_text}]}]}
    if image_base64:
        payload["contents"][0]["parts"].append({"inline_data": {"mime_type": "image/jpeg", "data": image_base64}})

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(f"{gemini_url}?key={gemini_api_key}", headers=headers, json=payload, timeout=30.0)
            response.raise_for_status()
            gemini_data = response.json()
        return gemini_data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "No AI analysis returned.")
    except Exception as e:
        return f"Failed to get Gemini AI analysis: {str(e)}"


# GET endpoint..
@app.get("/livability_score", response_model=LivabilityResult)
async def get_livability_score(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    solar: float = Query(..., description="Solar radiation score (0-5)"),
    aqi: float = Query(..., description="Air Quality Index (0-500)")
):
    score_label, habitable_years = calculate_livability_score(solar, aqi)
    gemini_text = await ask_gemini_analysis(lat, lon, solar, aqi)

    return LivabilityResult(
        location_latitude=lat,
        location_longitude=lon,
        solar_radiation_score=solar,
        air_quality_index=aqi,
        calculated_livability_score=score_label,
        estimated_habitable_years=habitable_years,
        gemini_analysis=gemini_text
    )


# POST Method..

# @app.post("/livability_score", response_model=LivabilityResult)
# async def post_livability_score(
#     lat: float = Form(...),
#     lon: float = Form(...),
#     solar: float = Form(...),
#     aqi: float = Form(...),
#     file: UploadFile = File(None)
# ):
#     image_base64 = None
#     if file:
#         image_bytes = await file.read()
#         image_base64 = base64.b64encode(image_bytes).decode("utf-8")

#     score_label, habitable_years = calculate_livability_score(solar, aqi)
#     gemini_text = await ask_gemini_analysis(lat, lon, solar, aqi, image_base64=image_base64)

#     return LivabilityResult(
#         location_latitude=lat,
#         location_longitude=lon,
#         solar_radiation_score=solar,
#         air_quality_index=aqi,
#         calculated_livability_score=score_label,
#         estimated_habitable_years=habitable_years,
#         gemini_analysis=gemini_text
#     )
