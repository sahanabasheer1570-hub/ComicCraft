from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from google import genai
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Initialize New GenAI Client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    # Fixed: request parameter separated from context dict
    return templates.TemplateResponse(
        request=request, 
        name="index.html"
    )

@app.post("/generate", response_class=HTMLResponse)
async def generate_comic(request: Request, prompt: str = Form(...)):
    full_prompt = f"Create a detailed 4-panel comic script with dialogue and scene descriptions based on this idea: {prompt}"
    
    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=full_prompt,
    )
    
    # Fixed: request=request and context={"story": response.text} separately
    return templates.TemplateResponse(
        request=request, 
        name="index.html", 
        context={"story": response.text}
    )