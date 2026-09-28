import os
import json
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
import google.generativeai as genai
import torch
from diffusers import StableDiffusionPipeline
from fpdf import FPDF

load_dotenv()

# Gemini Configuration
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model_flash = genai.GenerativeModel("gemini-1.5-flash")

# FastAPI Setup
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

os.makedirs("static/generated_images", exist_ok=True)

pipe = None

def get_sd_pipeline():
    global pipe
    if pipe is None:
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", 
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        if torch.cuda.is_available():
            pipe = pipe.to("cuda")
    return pipe


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "panels": None})


@app.post("/generate", response_class=HTMLResponse)
async def generate_comic(
    request: Request,
    theme: str = Form(...),
    characters: str = Form(...),
    style: str = Form(...)
):
    prompt = f"""
    Create a 3-panel comic story outline based on:
    Theme: {theme}
    Characters: {characters}
    Art Style: {style}

    Return strictly a JSON array with 3 panel objects. Each object must have:
    - "panel_number": integer
    - "visual_description": brief image description for AI image generator
    - "narration": story text
    - "dialogue": character dialogues
    """
    
    response = model_flash.generate_content(prompt)
    
    try:
        raw_text = response.text.strip().replace("```json", "").replace("```", "")
        panels_data = json.loads(raw_text)
    except Exception:
        panels_data = [
            {"panel_number": 1, "visual_description": f"{theme} introducing {characters}", "narration": "The journey begins.", "dialogue": f"{characters}: Let's go!"},
            {"panel_number": 2, "visual_description": f"{characters} facing a challenge in {style} style", "narration": "A sudden obstacle appears.", "dialogue": f"{characters}: What's that?"},
            {"panel_number": 3, "visual_description": f"{characters} victorious, {style} style", "narration": "Victory achieved!", "dialogue": f"{characters}: We did it!"}
        ]

    sd = get_sd_pipeline()
    generated_panels = []

    for panel in panels_data:
        image_prompt = f"{panel['visual_description']}, {style} style, comic book panel, high quality"
        image = sd(image_prompt, num_inference_steps=20).images[0]
        
        image_path = f"static/generated_images/panel_{panel['panel_number']}.png"
        image.save(image_path)
        
        panel["image_url"] = f"/{image_path}"
        generated_panels.append(panel)

    return templates.TemplateResponse(
        "index.html", 
        {"request": request, "panels": generated_panels, "theme": theme}
    )


@app.post("/download-pdf")
async def download_pdf(theme: str = Form(...)):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, f"Comic: {theme}", ln=True, align='C')
    pdf.ln(10)

    for i in range(1, 4):
        img_path = f"static/generated_images/panel_{i}.png"
        if os.path.exists(img_path):
            pdf.image(img_path, x=30, w=150)
            pdf.ln(5)

    pdf_output_path = "static/comic_output.pdf"
    pdf.output(pdf_output_path)
    return FileResponse(path=pdf_output_path, filename="comic.pdf", media_type="application/pdf")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)