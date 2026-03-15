from .model import WikiLink
from fastapi import FastAPI
from pydantic import BaseModel

from fastapi.responses import HTMLResponse

from os.path import join
from os.path import dirname

app = FastAPI()
torch_path = join(dirname(__file__), './torch/mlp_epoch_7_550.pth')
html_path = join(dirname(__file__), './public/index.html')
instance = WikiLink(torch_path, "cpu")

class Text(BaseModel):
    text: str

@app.post("/generate")
def generate(text: Text):
    text = text.text
    html = instance.generate(text)

    return {
        "html": html
    }

@app.get("/")
def front():
    with open(html_path) as source:
        content = source.read()
    return HTMLResponse(content)
