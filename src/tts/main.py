

from transformers import AutoProcessor, AutoModel
import numpy as np
import scipy

# Load the Bark model and processor
processor = AutoProcessor.from_pretrained("suno/bark")
model = AutoModel.from_pretrained("suno/bark")

voice_preset = "v2/en_speaker_3"

# Prepare the text input
# text = "In a land far far away"
# text = "Ze Manga"
text = "Rasco Vosa"

# Generate speech with explicit attention mask
inputs = processor(text=text, voice_preset=voice_preset, return_tensors="pt")

# Generate speech
speech_output = model.generate(**inputs)

# The output is a tensor, convert it to a numpy array
speech_numpy = speech_output.cpu().numpy().squeeze()

# Save the audio to a file
sample_rate = 24000  # Bark uses 24kHz sample rate
scipy.io.wavfile.write("bark_rasco_vosa_3.wav", rate=sample_rate, data=speech_numpy)

print("Audio generated and saved as bark_output.wav")