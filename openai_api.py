import io
import sys
import time

import torchaudio
import uvicorn
from fastapi import FastAPI, Response

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav

app = FastAPI()

sys.path.append('third_party/Matcha-TTS')

cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=True, load_onnx=False, load_trt=False)

# sft usage


prompt_speech_16k = load_wav('zero_shot_prompt.wav', 16000)


@app.post("/v1/audio/speech")
async def tts(
        request: dict
):
    print("openai_api.py tts() start...")
    start = time.process_time()
    # 生成音频数据
    audio_data = cosyvoice.inference_zero_shot(
        tts_text=request["input"].strip(),
        prompt_text='希望你以后能够做的比我还好呦。',
        prompt_speech_16k=prompt_speech_16k,
        stream=False,
    )

    # 将音频数据写入 buffer
    buffer = io.BytesIO()
    for i, j in enumerate(audio_data):
        torchaudio.save(buffer, j['tts_speech'], cosyvoice.sample_rate, format="wav")

    end = time.process_time()
    print(f"Infer time: {end - start:.1f}s")

    # 将 buffer 指针重置到起始位置
    buffer.seek(0)

    # 返回音频数据
    return Response(content=buffer.read(-1), media_type="audio/wav")


if __name__ == '__main__':
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=7998,
        log_level="info"
    )
