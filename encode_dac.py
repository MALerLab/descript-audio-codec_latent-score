import os
import dac
from tqdm import tqdm
from audiotools import AudioSignal
from pathlib import Path

# Function to find all wav files in a directory and its subdirectories
def find_wav_files(directory):
    wav_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.wav'):
                wav_files.append(os.path.join(root, file))
    return wav_files

# Function to encode wav files and save as dac files
def encode_wav_files(wav_files, model, stereo=True):
    for wav_file in tqdm(wav_files):
        try:
            # Load audio signal file
            signal = AudioSignal(wav_file)
            if stereo != True:
                signal = signal.to_mono()

            # Save compressed file
            x = model.compress(signal)

            if stereo == True:
                dac_file_path = wav_file.replace('.wav', '.dac').replace('audio_segments', 'audio_tokens_stereo')
            else:
                dac_file_path = wav_file.replace('.wav', '.dac').replace('audio_segments', 'audio_tokens_mono')
            
            Path(dac_file_path).parent.mkdir(parents=True, exist_ok=True)

            x.save(dac_file_path)

            print(f"Encoded and saved {wav_file} to {dac_file_path}")
        except Exception as e:
            print(f"Error processing {wav_file}: {e}")

# Main function to load model and process files
def main(directory):
    # Download and load model
    model_path = dac.utils.download(model_type="44khz")
    model = dac.DAC.load(model_path)
    model.to('cuda')

    # Find all wav files in the directory and its subdirectories
    wav_files = find_wav_files(directory)
    
    # Encode and save each wav file
    encode_wav_files(wav_files, model, stereo=False)

if __name__ == "__main__":
    directory = str(Path.home() / "userdata/latent_score_dataset/string_quartet/split")  # Set your directory path here
    main(directory)
