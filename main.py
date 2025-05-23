from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from model_manager import load_model
app = FastAPI()
templetes = Jinja2Templates(directory="templates")
 
 
@app.get("/")
async def root():
    return templetes.TemplateResponse("index.html", {"request": {},"name":"javier"})
 

@app.post("/calcular_precio", response_class=HTMLResponse)
async def calcular_precio(
    request: Request,
    brand: str = Form(...),
    type: str = Form(...),
    frame_material: str = Form(...),
    gear_count: int = Form(...),
    wheel_size: float = Form(...),
    weight_kg: float = Form(...),
):
    model = load_model()

    return templetes.TemplateResponse(
        "index.html",
        {
            "request": request,
            "username": username,
        },
    )


@app.post("/post")
async def post_metho(data: dict):
    return {"message": "Hola de nuevo", "data": data}

