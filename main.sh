#!/bin/bash
eval "$(/root/miniconda3/bin/conda shell.bash hook)"
conda activate audio-api
uvicorn openai_api:app --reload --host 0.0.0.0 --port 7998