import dac
from tqdm import tqdm
from audiotools import AudioSignal
from pathlib import Path

# Function to encode wav files and save as dac files
def encode_wav_files(wav_files, model, model_id, stereo=True, drop_code_idx = None):
  for wav_file in tqdm(wav_files):
    try:
      # Load audio signal file
      signal = AudioSignal(wav_file)
      if stereo != True:
        signal = signal.to_mono()

      # Compress audio
      x = model.compress(signal, win_duration=None)

      # Drop codes
      if drop_code_idx:
        x.codes = x.codes[:,:drop_code_idx]
        n_codebook = drop_code_idx
      else:
        n_codebook = x.codes.shape[1]
        
      # Save compressed file
      if stereo == True:
        num_channels = 'stereo'
      else:
        num_channels = 'mono'

      dac_file_name = wav_file.stem
      dac_file_path = wav_file.parents[2] / 'audio_tokens' / num_channels / f'{model_id}_{n_codebook}' / f'{dac_file_name}.dac'

      Path(dac_file_path).parent.mkdir(parents=True, exist_ok=True)

      x.save(dac_file_path)

      print(f"Encoded and saved {wav_file} to {dac_file_path}")
    except Exception as e:
      print(f"Error processing {wav_file}: {e}")

# Main function to load model and process files
def main(directory, stereo=False, model_path = None, model_id = None, drop_code_idx = None):
  # Download and load model
  if model_path is None:
    model_path = dac.utils.download(model_type="44khz")
    
  model = dac.DAC.load(model_path)
  model.to('cuda')

  # Find all wav files in the directory and its subdirectories
  wav_files = list(Path(directory).rglob('*.wav'))
  
  # Encode and save each wav file
  encode_wav_files(wav_files, model, model_id = model_id, stereo=stereo, drop_code_idx = drop_code_idx)

if __name__ == "__main__":
  directory = str(Path.home() / "userdata/latent_score_dataset/string_quartet/segments")  # Set your directory path here
  
  model_path = Path.home() / 'userdata' / "sake/descript-audio-codec_latent-score/sqdac4/dac/weights.pth"
  model_id = "sqdac4"
  
  drop_code_idx = None

  main(directory, stereo=False, model_path = model_path, model_id = model_id, drop_code_idx = drop_code_idx)
