import requests
import os

from fastapi import FastAPI, Request, File, UploadFile, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from typing import Annotated, List

import whisper
import torch
import gradio as gr

from tempfile import NamedTemporaryFile

torch.cuda.is_available()

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model = whisper.load_model('small', DEVICE)


def inference(audio):
    audio = whisper.load_audio(audio)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio, padding=480000).to(model.device)
    _, probs = model.detect_language(mel)
    options = whisper.DecodingOptions(fp16=False)
    result = whisper.decode(model, mel, options)
    print(result.text)
    return result.text, gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)


app = FastAPI(title="aitest")

# tmp_file_dir = "./"
# Path(tmp_file_dir).mkdir(parents=True, exist_ok=True)


@app.post('/whisper')
async def handler(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No files were uploaded")

    # with open(os.path.join(tmp_file_dir, file.filename), 'wb') as disk_file:
    file_bytes = await file.read()
    result = model.transcribe(file_bytes)
    print(result)
    # results = []

    # # with NamedTemporaryFile(delete=True, dir='./') as temp:
    # with open(file.filename, 'rb+') as temp_file:
    #     # temp_file.write(file.file.read())
    #     # print(temp_file.name)
    #   result = inference(file.file.read())
    #   print(result)
    # # results.append(
    # #   {
    # #     "filename": file.filename,
    # #     "transribe": result['text']
    # #   }
    # # )
    # # print(file.filename)
    # # return JSONResponse(content={
    # #   'results': results[0]['text']
    # # })


@app.get('/', response_class=RedirectResponse)
async def redirect_docs():
    return '/docs'
