"""Voice: text-to-speech for replies and speech-to-text for incoming voice notes / audio files.

TTS engines (all MIT-licensed, commercial use allowed):
- "silero"     Silero v5 CIS (CPU, ~90 MB): Russian, Tajik, Uzbek female voices. Russian stress by silero-stress.
               Emotions are rendered with SSML prosody (speed, pitch) and natural pauses.
- "chatterbox" Chatterbox Multilingual (GPU, ~3 GB): Russian, English, Chinese and others, optional voice cloning.
Default: Silero for ru/tg/uz, Chatterbox (if installed) for other languages.
STT: faster-whisper. Heavy libraries are imported lazily so the bot still runs text-only without them.
"""
from __future__ import annotations

import asyncio
import io
import logging
import random
import threading
from xml.sax.saxutils import escape

from .config import ROOT, settings

log = logging.getLogger(__name__)

SILERO_URL = "https://models.silero.ai/models/tts/ru/v5_cis_base_nostress.pt"
SILERO_PATH = ROOT / "models" / "v5_cis_base_nostress.pt"
SILERO_SPEAKERS = {"ru": settings.tts_speaker_ru, "tg": settings.tts_speaker_tg, "uz": settings.tts_speaker_uz}
SILERO_SR = 48000

# emotion -> (speech rate, pitch, pause after in seconds)
PROSODY = {
    "радость": ("fast", "high", 0.28),
    "воодушевление": ("medium", "high", 0.3),
    "тепло": ("medium", "medium", 0.32),
    "сочувствие": ("slow", "low", 0.45),
    "вопрос": ("medium", "high", 0.4),
    "спокойно": ("medium", "medium", 0.32),
}
# emotion -> Chatterbox exaggeration
EXAGGERATION = {"радость": 0.8, "воодушевление": 0.75, "тепло": 0.55, "сочувствие": 0.35, "вопрос": 0.55,
                "спокойно": 0.45}

Segments = list[tuple[str, str]]  # [(emotion, sentence)]


def _encode_ogg(pieces: list, sr: int) -> bytes:
    import numpy as np
    import soundfile as sf

    audio = np.concatenate(pieces).astype("float32") if pieces else np.zeros(sr // 2, dtype="float32")
    peak = float(np.abs(audio).max() or 1.0)
    audio = audio / peak * 0.92
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="OGG", subtype="OPUS")  # Telegram voice notes need OGG/Opus
    return buf.getvalue()


def _silence(sr: int, seconds: float):
    import numpy as np

    return np.zeros(int(sr * seconds * random.uniform(0.85, 1.2)), dtype="float32")


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

    def synth(self, segments: Segments, lang: str) -> bytes:
        model = self._load()
        pieces = [_silence(SILERO_SR, 0.12)]
        for emotion, sentence in segments:
            text = self._accentor(sentence) if lang == "ru" else sentence
            rate, pitch, pause = PROSODY.get(emotion, PROSODY["тепло"])
            ssml = f"<speak><prosody rate='{rate}' pitch='{pitch}'>{escape(text)}</prosody></speak>"
            try:
                wav = model.apply_tts(ssml_text=ssml, speaker=SILERO_SPEAKERS[lang], sample_rate=SILERO_SR)
            except Exception:  # noqa: BLE001 - fall back to plain text if SSML is rejected
                wav = model.apply_tts(text=text, speaker=SILERO_SPEAKERS[lang], sample_rate=SILERO_SR)
            pieces += [wav.numpy(), _silence(SILERO_SR, pause)]
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

    def synth(self, segments: Segments, lang: str) -> bytes:
        import torch

        if self._model is None:
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS

            device = settings.tts_device if torch.cuda.is_available() else "cpu"
            log.info("Loading Chatterbox Multilingual on %s…", device)
            self._model = ChatterboxMultilingualTTS.from_pretrained(device=device)
        sr = self._model.sr
        pieces = []
        for emotion, sentence in segments:
            kwargs = dict(language_id=lang, exaggeration=EXAGGERATION.get(emotion, settings.tts_exaggeration),
                          cfg_weight=settings.tts_cfg_weight)
            if settings.tts_voice_ref:
                kwargs["audio_prompt_path"] = settings.tts_voice_ref
            wav = self._model.generate(sentence, **kwargs).squeeze(0).detach().cpu().numpy()
            pieces += [wav, _silence(sr, PROSODY.get(emotion, PROSODY["тепло"])[2])]
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

    def supports(self, lang: str) -> bool:
        return self._engine(lang) is not None

    def warmup(self) -> None:
        try:
            self._synth_sync([("радость", "Привет!")], "ru")
        except Exception:  # noqa: BLE001
            log.exception("TTS warmup failed (voice replies disabled until fixed)")

    def _synth_sync(self, segments: Segments, lang: str) -> bytes | None:
        engine = self._engine(lang)
        if engine is None:
            log.info("no TTS voice for language %s, skipping voice reply", lang)
            return None
        with self._lock:
            return engine.synth(segments, lang)

    async def synthesize(self, segments: Segments, lang: str) -> bytes | None:
        segments = [(e, s) for e, s in segments if s.strip()]
        if not segments:
            return None
        return await asyncio.to_thread(self._synth_sync, segments, lang)


class STT:
    def __init__(self) -> None:
        self._model = None
        self._lock = threading.Lock()

    def _transcribe_sync(self, data: bytes) -> tuple[str, str]:
        with self._lock:
            if self._model is None:
                from faster_whisper import WhisperModel

                device = settings.stt_device
                compute = "float16" if device == "cuda" else "int8"
                log.info("Loading faster-whisper %s on %s (%s)…", settings.stt_model, device, compute)
                self._model = WhisperModel(settings.stt_model, device=device, compute_type=compute)
            segments, info = self._model.transcribe(io.BytesIO(data), beam_size=5, vad_filter=True)
            text = " ".join(s.text.strip() for s in segments).strip()
            return text, info.language

    async def transcribe(self, data: bytes) -> tuple[str, str]:
        """Returns (text, detected language code)."""
        return await asyncio.to_thread(self._transcribe_sync, data)
