import random
import sys
from importlib.resources import files

import soundfile as sf
import os
import tqdm
from cached_path import cached_path
from hydra.utils import get_class
from omegaconf import OmegaConf

from f5_tts.infer.utils_infer import (
    load_model,
    load_vocoder,
    transcribe,
    preprocess_ref_audio_text,
    infer_process,
    remove_silence_for_generated_wav,
    save_spectrogram,
)
from f5_tts.model.utils import seed_everything


class F5TTS:
    def __init__(
        self,
        model="F5TTS_v1_Base",
        ckpt_file="",
        vocab_file="",
        ode_method="euler",
        use_ema=True,
        vocoder_local_path=None,
        device=None,
        hf_cache_dir=None,
    ):
        model_cfg = OmegaConf.load(str(files("f5_tts").joinpath(f"configs/{model}.yaml")))
        model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
        model_arc = model_cfg.model.arch

        self.mel_spec_type = model_cfg.model.mel_spec.mel_spec_type
        self.target_sample_rate = model_cfg.model.mel_spec.target_sample_rate

        self.ode_method = ode_method
        self.use_ema = use_ema

        if device is not None:
            self.device = device
        else:
            import torch

            self.device = (
                "cuda"
                if torch.cuda.is_available()
                else "xpu"
                if torch.xpu.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )

        # Load models
        self.vocoder = load_vocoder(
            self.mel_spec_type, vocoder_local_path is not None, vocoder_local_path, self.device, hf_cache_dir
        )

        repo_name, ckpt_step, ckpt_type = "F5-TTS", 1250000, "safetensors"

        # override for previous models
        if model == "F5TTS_Base":
            if self.mel_spec_type == "vocos":
                ckpt_step = 1200000
            elif self.mel_spec_type == "bigvgan":
                model = "F5TTS_Base_bigvgan"
                ckpt_type = "pt"
        elif model == "E2TTS_Base":
            repo_name = "E2-TTS"
            ckpt_step = 1200000

        if not ckpt_file:
            ckpt_file = str(
                cached_path(f"hf://SWivid/{repo_name}/{model}/model_{ckpt_step}.{ckpt_type}", cache_dir=hf_cache_dir)
            )
        self.ema_model = load_model(
            model_cls, model_arc, ckpt_file, self.mel_spec_type, vocab_file, self.ode_method, self.use_ema, self.device
        )

    def transcribe(self, ref_audio, language=None):
        return transcribe(ref_audio, language)

    def export_wav(self, wav, file_wave, remove_silence=False):
        sf.write(file_wave, wav, self.target_sample_rate)

        if remove_silence:
            remove_silence_for_generated_wav(file_wave)

    def export_spectrogram(self, spec, file_spec):
        save_spectrogram(spec, file_spec)

    def infer(
        self,
        ref_file,
        ref_text,
        gen_text,
        show_info=print,
        progress=tqdm,
        target_rms=0.1,
        cross_fade_duration=0.15,
        sway_sampling_coef=-1,
        cfg_strength=2,
        nfe_step=32,
        speed=1.0,
        fix_duration=None,
        remove_silence=False,
        file_wave=None,
        file_spec=None,
        seed=None,
    ):
        if seed is None:
            seed = random.randint(0, sys.maxsize)
        seed_everything(seed)
        self.seed = seed

        ref_file, ref_text = preprocess_ref_audio_text(ref_file, ref_text)

        wav, sr, spec = infer_process(
            ref_file,
            ref_text,
            gen_text,
            self.ema_model,
            self.vocoder,
            self.mel_spec_type,
            show_info=show_info,
            progress=progress,
            target_rms=target_rms,
            cross_fade_duration=cross_fade_duration,
            nfe_step=nfe_step,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            speed=speed,
            fix_duration=fix_duration,
            device=self.device,
        )

        if file_wave is not None:
            self.export_wav(wav, file_wave, remove_silence)

        if file_spec is not None:
            self.export_spectrogram(spec, file_spec)

        return wav, sr, spec


if __name__ == "__main__":
    f5tts = F5TTS()
    wavfiles = os.listdir("/content/voicefiles/")
    quotes = [
        "Success is not final, failure is not fatal: it is the courage to continue that counts.",
        "Opportunities don't happen. You create them.",
        "Hard work beats talent when talent doesn’t work hard.",
        "Do what you can, with what you have, where you are.",
        "Dream big and dare to fail.",
        "Knowing yourself is the beginning of all wisdom.",
        "The only true wisdom is in knowing you know nothing.",
        "An investment in knowledge pays the best interest.",
        "Judge a man by his questions rather than his answers.",
        "Wisdom is not a product of schooling but of the lifelong attempt to acquire it.",
        "Happiness depends upon ourselves.",
        "Life is really simple, but we insist on making it complicated.",
        "The purpose of our lives is to be happy.",
        "Do what you love, and you’ll never work a day in your life.",
        "Life isn’t about finding yourself. It’s about creating yourself.",
        "To love and be loved is to feel the sun from both sides.",
        "Love all, trust a few, do wrong to none.",
        "We accept the love we think we deserve.",
        "A loving heart is the truest wisdom.",
        "Love is not about how much you say ‘I love you,’ but how much you prove it.",
        "Courage is resistance to fear, mastery of fear, not absence of fear.",
        "He who has a why to live can bear almost any how.",
        "Strength does not come from physical capacity. It comes from an indomitable will.",
        "Bravery is not the absence of fear but the triumph over it.",
        "Do what you fear, and the death of fear is certain.",
        "It does not matter how slowly you go as long as you do not stop.",
        "The secret of getting ahead is getting started.",
        "Don’t watch the clock; do what it does. Keep going.",
        "Fall seven times, stand up eight.",
        "Believe you can, and you’re halfway there."
    ]

    for idx, wavfile in enumerate(wavfiles):
      wav, sr, spec = f5tts.infer(
          # ref_file=str(files("f5_tts").joinpath("infer/examples/basic/basic_ref_en.wav")),
          ref_file = f"/content/voicefiles/{wavfile}",
          ref_text = "Come to a concert surrounded by a thousand candles.",
          gen_text=quotes[idx],
          file_wave=str(files("f5_tts").joinpath(f'../../tests/api_out{idx}.wav')),
          file_spec=str(files("f5_tts").joinpath(f'../../tests/api_out{idx}.png')),
          seed=None,
          nfe_step = 60
      )

      print("seed :", f5tts.seed)
