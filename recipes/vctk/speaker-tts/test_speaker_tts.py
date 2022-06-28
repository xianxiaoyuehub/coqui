import os
import torch
from trainer import Trainer, TrainerArgs

from TTS.config import BaseAudioConfig, BaseDatasetConfig
from TTS.tts.configs.speaker_tts_config import SpeakerTTSConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.speaker_tts import SpeakerTTS
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

os.environ['CUDA_VISIBLE_DEVICES'] = "2"

output_path = os.path.dirname(os.path.abspath(__file__))
dataset_config = BaseDatasetConfig(name="vctk", meta_file_train="", path=os.path.join(output_path, "../VCTK_all/"))

audio_config = BaseAudioConfig(
    sample_rate=22050,
    do_trim_silence=True,
    trim_db=23.0,
    signal_norm=False,
    mel_fmin=0.0,
    mel_fmax=8000,
    spec_gain=1.0,
    log_func="np.log",
    ref_level_db=20,
    preemphasis=0.0,
)

config = SpeakerTTSConfig(
    run_name="speaker_tts_vctk",
    audio=audio_config,
    batch_size=32,
    eval_batch_size=16,
    num_loader_workers=4,
    num_eval_loader_workers=4,
    compute_input_seq_cache=True,
    precompute_num_workers=4,
    compute_f0=True,
    f0_cache_path=os.path.join(output_path, "f0_cache"),
    run_eval=False,
    test_delay_epochs=-1,
    epochs=1000,
    text_cleaner="english_cleaners",
    use_phonemes=True,
    phoneme_language="en-us",
    phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
    print_step=50,
    print_eval=False,
    mixed_precision=False,
    min_text_len=0,
    max_text_len=500,
    min_audio_len=0,
    max_audio_len=500000,
    output_path=output_path,
    datasets=[dataset_config],
    use_speaker_embedding=True,
    device='cuda:0',

    test_sentences = [
        "I am a chinese and I love china",
        "Do not go gentle into that good night, Old age should burn and rave at close of day"
    ]
)


ap = AudioProcessor.init_from_config(config)

tokenizer, config = TTSTokenizer.init_from_config(config)
train_samples, eval_samples = load_tts_samples(dataset_config, eval_split=True)

speaker_manager = SpeakerManager()
speaker_manager.set_speaker_ids_from_data(train_samples + eval_samples)
# speaker_manager.speaker_ids
config.model_args.num_speakers = speaker_manager.num_speakers

# init model
model = SpeakerTTS(config, ap, tokenizer, speaker_manager=speaker_manager)

test_args = TrainerArgs(
    continue_path = "/workspace/tts-project/TTS/recipes/vctk/speaker_tts/speaker_tts_vctk-June-08-2022_09+29AM-0cf3265",
    skip_train_epoch = True
)

test = Trainer(
    test_args, config, output_path, model=model, train_samples=train_samples, eval_samples=eval_samples
)
output_path = '/workspace/tts-project/TTS/recipes/vctk/speaker_tts/'
test_output = test.test_run()
test.model.ap.save_wav(test_output[1]['0-audio'], f"{output_path}0-audio.wav", 16000)