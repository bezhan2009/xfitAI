"""Voice: text-to-speech for replies and speech-to-text for incoming voice notes.

TTS engines (all MIT-licensed, commercial use allowed):
- "silero"     Silero v5 CIS (CPU, ~90 MB): Russian, Tajik, Uzbek female voices. Russian stress by silero-stress.
- "chatterbox" Chatterbox Multilingual (GPU, ~3 GB): Russian, English, Chinese and others, optional voice cloning.
Default: Silero for ru/tg/uz, Chatterbox (if installed) for other languages.
STT: faster-whisper. Heavy libraries are imported lazily so the bot still runs text-only without them.
"""
from __future__ import annotations

import asyncio
import io
import logging
import re
import threading
from pathlib import Path

from .agent import user_language
from .config import ROOT, settings

log = logging.getLogger(__name__)

URL_RE = re.compile(r"(https?://\S+|t\.me/\S+|\b[\w.-]+\.(?:tj|com|ru)\S*)", re.I)
EMOJI_RE = re.compile("[\U0001F000-\U0001FAFF☀-➿️‍]+")

SILERO_URL = "https://models.silero.ai/models/tts/ru/v5_cis_base_nostress.pt"
SILERO_PATH = ROOT / "models" / "v5_cis_base_nostress.pt"
SILERO_SPEAKERS = {"ru": settings.tts_speaker_ru, "tg": settings.tts_speaker_tg, "uz": settings.tts_speaker_uz}
SILERO_SR = 48000


def basic_clean(text: str) -> str:
    text = URL_RE.sub("", text)
    text = EMOJI_RE.sub("", text)
    text = re.sub(r"[*_`#>|~«»\"]", "", text)
    text = re.sub(r"\(\s*\)", "", text)
    text = text.replace("—", ",").replace("–", ",")
    return re.sub(r"\s+", " ", text).strip()


def digits_to_words(text: str, lang: str) -> str:
    """Safety net for numbers the spoken rewrite missed (Silero can't read digits)."""
    try:
        from num2words import num2words
    except ImportError:
        return text
    nlang = {"ru": "ru", "tg": "ru", "uz": "ru", "en": "en"}.get(lang, "en")
    text = re.sub(r"(\d+)\s*%", lambda m: f"{m.group(1)} " + ("процентов" if nlang == "ru" else "percent"), text)
    return re.sub(r"\d+", lambda m: num2words(int(m.group()), lang=nlang), text)


def split_sentences(text: str, max_len: int = 220) -> list[str]:
    out: list[str] = []
    for p in re.split(r"(?<=[.!?…])\s+", text):
        p = p.strip()
        while len(p) > max_len:  # split long sentences on commas
            cut = p.rfind(",", 0, max_len)
            cut = cut if cut > 40 else max_len
            out.append(p[: cut + 1].strip())
            p = p[cut + 1:].strip()
        if p:
            out.append(p)
    return out


def _encode_ogg(pieces: list, sr: int) -> bytes:
    import numpy as np
    import soundfile as sf

    audio = np.concatenate(pieces).astype("float32") if pieces else np.zeros(sr // 2, dtype="float32")
    peak = float(np.abs(audio).max() or 1.0)
    audio = audio / peak * 0.92
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="OGG", subtype="OPUS")  # Telegram voice notes need OGG/Opus
    return buf.getvalue()


def _pause(sr: int, i: int):
    import numpy as np

    return np.zeros(int(sr * (0.22 if i % 2 else 0.3)), dtype="float32")


class SileroEngine:
    languages = {"ru", "tg", "uz"}

    def __init__(self) -> None:
        self._model = None
        self._accentor = None

    def _load(self):
        if self._model is None:
            import torch

            if not SILERO_PATH.exists():
                SILERO_PATH.parent.mkdir(exist_ok=True)
                log.info("Downloading Silero TTS model…")
                torch.hub.download_url_to_file(SILERO_URL, str(SILERO_PATH))
            torch.set_num_threads(4)
            model = torch.package.PackageImporter(str(SILERO_PATH)).load_pickle("tts_models", "model")
            model.to(torch.device("cpu"))
            from silero_stress import load_accentor

            self._accentor = load_accentor()
            self._model = model  # set last, so a failed load is retried next time
            log.info("Silero TTS loaded")
        return self._model

    def synth(self, text: str, lang: str) -> bytes:
        model = self._load()
        pieces = []
        for i, sentence in enumerate(split_sentences(text)):
            if lang == "ru":
                sentence = self._accentor(sentence)
            wav = model.apply_tts(text=sentence, speaker=SILERO_SPEAKERS[lang], sample_rate=SILERO_SR)
            pieces += [wav.numpy(), _pause(SILERO_SR, i)]
        return _encode_ogg(pieces, SILERO_SR)


class ChatterboxEngine:
    languages = {"ru", "en", "zh", "ar", "da", "de", "el", "es", "fi", "fr", "he", "hi", "it", "ja", "ko", "ms",
                 "nl", "no", "pl", "pt", "sv", "sw", "tr"}

    def __init__(self) -> None:
        self._model = None

    @staticmethod
    def available() -> bool:
        try:
            import chatterbox  # noqa: F401
            return True
        except ImportError:
            return False

    def synth(self, text: str, lang: str) -> bytes:
        import torch

        if self._model is None:
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS

            device = settings.tts_device if torch.cuda.is_available() else "cpu"
            log.info("Loading Chatterbox Multilingual on %s…", device)
            self._model = ChatterboxMultilingualTTS.from_pretrained(device=device)
        sr = self._model.sr
        kwargs = dict(language_id=lang, exaggeration=settings.tts_exaggeration, cfg_weight=settings.tts_cfg_weight)
        if settings.tts_voice_ref:
            kwargs["audio_prompt_path"] = settings.tts_voice_ref
        pieces = []
        for i, sentence in enumerate(split_sentences(text)):
            wav = self._model.generate(sentence, **kwargs).squeeze(0).detach().cpu().numpy()
            pieces += [wav, _pause(sr, i)]
        return _encode_ogg(pieces, sr)


class TTS:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.silero = SileroEngine()
        self.chatterbox = ChatterboxEngine() if ChatterboxEngine.available() else None

    def _engine(self, lang: str):
        prefer_cb = settings.tts_engine == "chatterbox"
        if self.chatterbox and (prefer_cb or lang not in SileroEngine.languages) and lang in ChatterboxEngine.languages:
            return self.chatterbox
        if lang in SileroEngine.languages:
            return self.silero
        return None

    def warmup(self) -> None:
        try:
            self._synth_sync("Привет!", "ru")
        except Exception:  # noqa: BLE001
            log.exception("TTS warmup failed (voice replies disabled until fixed)")

    def _synth_sync(self, text: str, lang: str) -> bytes | None:
        engine = self._engine(lang)
        if engine is None:
            log.info("no TTS voice for language %s, skipping voice reply", lang)
            return None
        with self._lock:
            return engine.synth(text, lang)

    async def synthesize(self, spoken_text: str) -> bytes | None:
        lang = user_language(spoken_text)
        text = digits_to_words(basic_clean(spoken_text), lang)
        if not text:
            return None
        return await asyncio.to_thread(self._synth_sync, text, lang)


class STT:
    def __init__(self) -> None:
        self._model = None
        self._lock = threading.Lock()

    def _transcribe_sync(self, data: bytes) -> str:
        with self._lock:
            if self._model is None:
                from faster_whisper import WhisperModel

                device = settings.stt_device
                compute = "float16" if device == "cuda" else "int8"
                log.info("Loading faster-whisper %s on %s (%s)…", settings.stt_model, device, compute)
                self._model = WhisperModel(settings.stt_model, device=device, compute_type=compute)
            segments, _info = self._model.transcribe(io.BytesIO(data), beam_size=5, vad_filter=True)
            return " ".join(s.text.strip() for s in segments).strip()

    async def transcribe(self, data: bytes) -> str:
        return await asyncio.to_thread(self._transcribe_sync, data)
