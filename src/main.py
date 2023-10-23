import json
import requests
import os

from fastapi import FastAPI, Request, File, UploadFile, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from typing import Annotated, List

import whisper
import hifi_gan
import torch

import numpy as np
import torchaudio

from tempfile import NamedTemporaryFile

torch.cuda.is_available()

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

model = whisper.load_model('base', DEVICE)

# dims_value = 80


# whisper_model = whisper.Whisper(dims=dims_value)

# Create an instance of the HIFI GAN model
hifi_gan_model = hifi_gan.initialize_model()



app= FastAPI(title = "aitest")

@app.post('/whisper')
async def handler(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No audio file was uploaded")

    # Read the audio file
    audio_data = file.file.read()

    # Perform audio transcription with Whisper
    transcription_result = model.transcribe(audio_data)

    return JSONResponse(content={
        'transcription': transcription_result
    })

@app.post('/hifi')
async def generate_audio(text: str):
    if not text:
        raise HTTPException(status_code=400, detail="No text was provided")

    # Generate audio with HIFI GAN from the provided text
    audio_data = hifi_gan_model.generate_audio(text)

    # Create a temporary audio file to send as a response
    with NamedTemporaryFile(delete=True, suffix=".wav") as temp:
        temp.write(audio_data)

        return FileResponse(temp.name)

@app.get('/', response_class=RedirectResponse)
async def redirect_docs():
  return '/docs'
