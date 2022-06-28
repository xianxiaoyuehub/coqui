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

os.environ['CUDA_VISIBLE_DEVICES'] = "5"
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

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
    batch_size=24,
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
    device='cuda:0'
)

# INITIALIZE THE AUDIO PROCESSOR
# Audio processor is used for feature extraction and audio I/O.
# It mainly serves to the dataloader and the training loggers.
ap = AudioProcessor.init_from_config(config)

# INITIALIZE THE TOKENIZER
# Tokenizer is used to convert text to sequences of token IDs.
# If characters are not defined in the config, default characters are passed to the config
tokenizer, config = TTSTokenizer.init_from_config(config)

# LOAD DATA SAMPLES
# Each sample is a list of ```[text, audio_file_path, speaker_name]```
# You can define your custom sample loader returning the list of samples.
# Or define your custom formatter and pass it to the `load_tts_samples`.
# Check `TTS.tts.datasets.load_tts_samples` for more details.
train_samples, eval_samples = load_tts_samples(dataset_config, eval_split=True)

# init speaker manager for multi-speaker training
# it maps speaker-id to speaker-name in the model and data-loader
speaker_manager = SpeakerManager()
speaker_manager.set_d_vectors_from_samples(train_samples + eval_samples)
speaker_manager.set_speaker_ids_from_data(train_samples + eval_samples)
# speaker_manager.speaker_ids
config.model_args.num_speakers = speaker_manager.num_speakers

# init model
model = SpeakerTTS(config, ap, tokenizer, speaker_manager=speaker_manager)

""" train from predict model """
# model_dict = model.state_dict()
# predict_model = torch.load('/workspace/tts-project/TTS/recipes/vctk/fast_pitch/fast_pitch_vctk-June-03-2022_05+01PM-0cf3265/best_model_170811.pth')
# state_dict = {k: v for k, v in predict_model.items() if k in model_dict.keys()}
# model_dict.update(state_dict)
# model.load_state_dict(model_dict)

# INITIALIZE THE TRAINER
# Trainer provides a generic API to train all the 🐸TTS models with all its perks like mixed-precision training,
# distributed training, etc.

# init TrainerArgs
trainerargs = TrainerArgs(
    continue_path = "/workspace/tts-project/TTS/recipes/vctk/speaker_tts/speaker_tts_vctk-June-08-2022_09+29AM-0cf3265",
    skip_train_epoch = False
)
trainer = Trainer(
    trainerargs, config, output_path, model=model, train_samples=train_samples, eval_samples=eval_samples
)

# AND... 3,2,1... 🚀
trainer.fit()