from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

pipeline = ASRInferencePipeline(model_card="omniASR_LLM_1B")

audio_files = ["data/transcribe_test.wav"]
lang = ["nob_Latn"]
transcriptions = pipeline.transcribe(audio_files, lang=lang, batch_size=1)

for file, trans in zip(audio_files, transcriptions):
    print(f"{file}: {trans}")