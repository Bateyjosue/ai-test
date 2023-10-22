import json
import requests
import os

from fastapi import FastAPI, Request, File, UploadFile, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from typing import Annotated, List

import whisper
import torch

from tempfile import NamedTemporaryFile

torch.cuda.is_available()

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

model = whisper.load_model('base', DEVICE)







app= FastAPI(title = "aitest")

@app.post('/whisper')
async def handler(files: List[UploadFile] = File(...)):
  if not files:
    raise HTTPException(status_code=400, detail="No files were uploaded")
  
  results = []

  for file in files:
    with NamedTemporaryFile(delete=True) as temp:
      with open(temp.name, 'wb+') as temp_file:
        temp_file.write(file.file.read())
        
        result = model.transcribe('temp.name')
        results.append(
          {
            "filename": file.filename,
            "transribe": result['text']
          }
        )
  return JSONResponse(content={
    'results': results
  })

@app.get('/', response_class=RedirectResponse)
async def redirect_docs():
  return '/docs'

# app.mount('/static', StaticFiles(directory="static"), name="static")
# templates = Jinja2Templates(directory="templates")

# async def transribe(filename):
#   model = await whisper.load_model('base')
#   text = await model.transcribe('./assets/'+ filename)
#   print(text)
#   return text['text']

# def get_files(path):
#     for file in os.listdir(path):
#         if os.path.isfile(os.path.join(path, file)):
#             yield file

# @app.get('/', response_class=HTMLResponse)
# def index(requests: Request):
#   return templates.TemplateResponse("index.html", context= {
#     "request": requests
# })

# @app.get('/audio', response_class=HTMLResponse)
# def audio(requests: Request):
#   files = get_files('./assets')
#   return templates.TemplateResponse("audio.html", context= {
#     "request": requests,
#     'filename': files
# })

# @app.post('/audio')
# async def upload(request: Request, audio:str):
#   text = await transribe(audio)

#   return templates.TemplateResponse("index.html", context= {
#     "request": requests,
#     "transcribe": text
# })