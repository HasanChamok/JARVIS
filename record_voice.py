import sounddevice as sd
import soundfile as sf

def record_voice_sample(filename="your_voice_sample.wav", duration=35):
    print("Get ready... Recording starts in 3 seconds!")
    import time
    time.sleep(3)
    print("🔴 Recording NOW — speak clearly!")
    sample_rate = 16000
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    sf.write(filename, audio, sample_rate)
    print(f"✅ Saved as {filename}")

record_voice_sample()