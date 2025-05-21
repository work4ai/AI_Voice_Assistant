import torch
import whisper
from transformers import BlipProcessor, BlipForQuestionAnswering
from PIL import Image
from gtts import gTTS
import gradio as gr
import os

# Load Whisper
whisper_model = whisper.load_model("base")

# Load BLIP
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
blip_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

def transcribe_audio(audio_path):
    if audio_path is None:
        return ""
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(whisper_model.device)
    options = whisper.DecodingOptions(fp16=False)
    result = whisper.decode(whisper_model, mel, options)
    return result.text.strip()

def analyze_image_vqa(image_path, question):
    image = Image.open(image_path).convert("RGB")
    prompt = question if question.strip() else "Describe the image in detail."

    inputs = blip_processor(image, prompt, return_tensors="pt")
    out = blip_model.generate(**inputs)
    result = blip_processor.decode(out[0], skip_special_tokens=True)
    return result

def speak_text(text, filename="response.mp3"):
    tts = gTTS(text=text, lang='en')
    tts.save(filename)
    return filename

def process(audio, image):
    question = transcribe_audio(audio)
    if not question:
        question = "Describe this image in detail."

    answer = analyze_image_vqa(image, question)
    audio_file = speak_text(answer)
    return question, answer, audio_file

iface = gr.Interface(
    fn=process,
    inputs=[
        gr.Audio(sources=["microphone"], type="filepath", label="Ask a Question"),
        gr.Image(type="filepath", label="Upload Image")
    ],
    outputs=[
        gr.Textbox(label="Transcribed Question"),
        gr.Textbox(label="Image Description"),
        gr.Audio(label="Spoken Answer")
    ],
    title="🖼️ Voice-Powered Image Description",
    description="Speak a question and upload an image. The app transcribes your voice, analyzes the image, and reads the answer aloud.",
)

iface.launch(debug=True)

