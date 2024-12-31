from openai import OpenAI

client = OpenAI(
    api_key="sk-123456",
    base_url="http://localhost:7998/v1",
)

response = client.audio.speech.create(
    model="tts-1",
    voice="alloy",
    input="《武松打虎》讲述了武松在回家乡的途中，经过景阳冈时遇到了一只老虎。武松与老虎展开了一场激烈的搏斗，最终成功地将老虎打死，成为了民间传说中的英雄。",
)

response.write_to_file("output.wav")


# import sys
#
# sys.path.append('third_party/Matcha-TTS')
# from cosyvoice.cli.cosyvoice import CosyVoice2
# from cosyvoice.utils.file_utils import load_wav
# import torchaudio
#
# cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=True, load_onnx=False, load_trt=False)
#
# # NOTE if you want to reproduce the results on https://funaudiollm.github.io/cosyvoice2, please add text_frontend=False during inference
# # zero_shot usage
# prompt_speech_16k = load_wav('zero_shot_prompt.wav', 16000)
# for i, j in enumerate(cosyvoice.inference_zero_shot(
#         '收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。',
#         '希望你以后能够做的比我还好呦。', prompt_speech_16k, stream=False)):
#     torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
#
# # fine grained control, for supported control, check cosyvoice/tokenizer/tokenizer.py#L248
# for i, j in enumerate(cosyvoice.inference_cross_lingual(
#         '在他讲述那个荒诞故事的过程中，他突然[laughter]停下来，因为他自己也被逗笑了[laughter]。', prompt_speech_16k,
#         stream=False)):
#     torchaudio.save('fine_grained_control_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
#
# # instruct usage
# for i, j in enumerate(cosyvoice.inference_instruct2(
#         '收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。',
#         '用四川话说这句话', prompt_speech_16k, stream=False)):
#     torchaudio.save('instruct_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
