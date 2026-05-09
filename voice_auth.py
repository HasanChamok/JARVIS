# voice_auth.py
from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
import numpy as np

encoder = VoiceEncoder()

# Load your voice sample
your_voice = preprocess_wav(Path("your_voice_sample.wav"))
your_embedding = encoder.embed_utterance(your_voice)

def is_my_voice(audio_input, threshold=0.71):
    try:
        processed = preprocess_wav(audio_input)
        input_embedding = encoder.embed_utterance(processed)
        similarity = np.dot(your_embedding, input_embedding)
        print(f"Voice similarity: {similarity:.2f}")
        return similarity >= threshold
    except:
        return False