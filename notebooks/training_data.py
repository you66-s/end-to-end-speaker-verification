import librosa, os, time
import numpy as np
import matplotlib.pyplot as plt

raw_data_path = r'data\raw_data'
actors = os.listdir(raw_data_path)

n_mels = 64
n_fft = 512
hop_length = 160
start = time.time()
for actor in actors:
    print(f"Processing actor: {actor}")
    audios = [f for f in os.listdir(os.path.join(raw_data_path, actor)) if f.endswith('.wav')]
    for audio in audios:
        audio_path = os.path.join(raw_data_path, actor, audio)
        y, sr = librosa.load(audio_path, sr=16000)
        y = y / np.max(np.abs(y))
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        plt.figure(figsize=(8, 6))
        librosa.display.specshow(mel_spec_db, sr=sr, hop_length=hop_length)
        plt.tight_layout()
        output_dir = os.path.join("data/spectogrammes", actor)
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"{audio}_mel.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
end = time.time()
print(f"Time taken: {end - start:.2f} seconds")