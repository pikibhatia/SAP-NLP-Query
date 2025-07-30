import os
import tempfile
import wave
import streamlit as st
from dotenv import load_dotenv
import pyaudio
from groq import Groq

# Load environment files
load_dotenv()
# Set up Groq client
api_key = st.secrets['GROQ_API_KEY']
client = Groq(api_key=api_key)

class VoiceToText:
    

    def record_audio():
        chunk = 1024  # Record in chunks of 1024 samples
        sample_format = pyaudio.paInt16  # 16 bits per sample
        channels = 1
        sample_rate=16000 # Record at 16000 samples per second
        duration = 10
        p = pyaudio.PyAudio()  # Create an interface to PortAudio

        print('Recording')

        stream = p.open(format=sample_format,
                        channels=channels,
                        rate=sample_rate,
                        frames_per_buffer=chunk,
                        input=True)
        frames = []

        for _ in range(0, int(sample_rate / chunk * duration)):
            data = stream.read(chunk)
            frames.append(data)

        # Stop and close the stream
        stream.stop_stream()
        stream.close()
        p.terminate()

        print('Finished recording')

        return frames, sample_rate

    def save_audio(frames, sample_rate):
        """
        Save recorded audio to a temporary WAV file.
        """
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            wf = wave.open(temp_audio.name, "wb")
            wf.setnchannels(1)
            wf.setsampwidth(pyaudio.PyAudio().get_sample_size(pyaudio.paInt16))
            wf.setframerate(sample_rate)
            wf.writeframes(b"".join(frames))
            wf.close()
            return temp_audio.name

    def transcribe_audio(audio_file_path):
            """
            Transcribe audio using Groq's Whisper implementation.
            """
            try:
                with open(audio_file_path, "rb") as file:
                    transcription = client.audio.transcriptions.create(
                        file=(os.path.basename(audio_file_path), file.read()),
                        model="whisper-large-v3",
                        prompt="""The audio is by a programmer discussing programming issues, the programmer mostly uses python and might mention python libraries or reference code in his speech.""",
                        response_format="text",
                        language="en",
                    )
                return transcription  # This is now directly the transcription text
            except Exception as e:
                print(f"An error occurred: {str(e)}")
                return None