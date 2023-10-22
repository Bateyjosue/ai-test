import json
import requests
import os

from fastapi import FastAPI, Request, File, UploadFile
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from typing import Annotated

import whisper




app= FastAPI(title = "aitest")

app.mount('/static', StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

def get_files(path):
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            yield file

@app.get('/', response_class=HTMLResponse)
def index(requests: Request):
  return templates.TemplateResponse("index.html", context= {
    "request": requests
})

@app.get('/audio', response_class=HTMLResponse)
def audio(requests: Request):
  files = get_files('./assets')

  model = whisper.load_model('base')
  text = model.transribe('assets/'+ filename)
  return templates.TemplateResponse("audio.html", context= {
    "request": requests,
    'filename': files
})

@app.post('/audio')
async def upload(request: Request, ):
  files = get_files('./assets')
  return templates.TemplateResponse("index.html", context= {
    "request": requests,
})