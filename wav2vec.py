import pyaudio
import speech_recognition as sr
import torch
from transformers import Wav2Vec2ForCTC ,Wav2Vec2Processor
import io
from pydub import AudioSegment
import openai

# Set your OpenAI API key
openai.api_key = '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'

# Load pre-trained model and processor
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
tokenizer = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

r = sr.Recognizer()

with sr.Microphone(sample_rate=16000) as source:
    print("Listening... Press Ctrl+C to stop.")
    while True:
        audio = r.listen(source)
        data = io.BytesIO(audio.get_wav_data())
        clip = AudioSegment.from_file(data)
        x = torch.FloatTensor(clip.get_array_of_samples())

        inputs = tokenizer(x, sampling_rate=16000, return_tensors='pt', padding='longest').input_values
        logits = model(inputs).logits
        tokens = torch.argmax(logits, axis=-1)
        text = tokenizer.batch_decode(tokens)

        print("you said : ", str(text).lower())

        response = openai.Completion.create(
            engine="babbage-002",  # Specify the engine (davinci, curie, etc.)
            prompt=text,  # Pass the user's spoken text as prompt
            max_tokens=50,  # Adjust max tokens as needed for the response length
            temperature=0.7,  # Adjust temperature for randomness in responses
            n=1  # Number of responses to generate
        )

        # Get the generated response
        chat_gpt_response = response.choices[0].text.strip()

        print("ChatGPT response:", chat_gpt_response)
