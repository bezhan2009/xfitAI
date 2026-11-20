"""Voice: text-to-speech for replies (Chatterbox Multilingual, MIT) and speech-to-text for incoming voice
notes (faster-whisper). Heavy libraries are imported lazily so the bot still runs text-only without them."""
from __future__ import annotations

import asyncio
import io
import logging
import re
import threading

from .config import settings

log = logging.getLogger(__name__)

URL_RE = re.compile(r"(https?://\S+|t\.me/\S+|\b[\w.-]+\.(?:tj|com|ru)\S*)", re.I)
EMOJI_RE = re.compile("[\U0001F000-\U0001FAFF☀-➿️‍]+")
TAJIK_RE = re.compile(r"[ғӣқӯҳҷҒӢҚӮҲҶ]")
CJK_RE = re.compile(r"[一-鿿]")


def detect_lang(text: str) -> str:
    """Language id for Chatterbox. Tajik isn't supported, so Cyrillic Tajik is voiced with the Russian model."""
    if CJK_RE.search(text):
        return "zh"
    cyr = len(re.findall(r"[А-Яа-яЁё]", text))
    lat = len(re.findall(r"[A-Za-z]", text))
    return "ru" if cyr >= lat else "en"


def basic_clean(text: str) -> str:
    text = URL_RE.sub("", text)
    text = EMOJI_RE.sub("", text)
    text = re.sub(r"[*_`#>|~]", "", text)
    text = re.sub(r"\(\s*\)", "", text)
    return re.sub(r"\s+", " ", text).strip()


def digits_to_words(text: str, lang: str) -> str:
    """Safety net for numbers the rewrite step missed."""
    try:
        from num2words import num2words
    except ImportError:
        return text
    nlang = "ru" if lang == "ru" else "en"
    text = re.sub(r"(\d+)\s*%", lambda m: f"{m.group(1)} процентов" if nlang == "ru" else f"{m.group(1)} percent", text)
    text = re.sub(r"\b(\d{1,2}):(\d{2})\b",
                  lambda m: num2words(int(m.group(1)), lang=nlang) + ("" if m.group(2) == "00" else " " + num2words(int(m.group(2)), lang=nlang)),
                  text)
    return re.sub(r"\d+", lambda m: num2words(int(m.group()), lang=nlang), text)


def split_sentences(text: str, max_len: int = 220) -> list[str]:
    parts = re.split(r"(?<=[.!?…])\s+", text)
    out: list[str] = []
    for p in parts:
        p = p.strip()
        while len(p) > max_len:  # split long sentences on commas
            cut = p.rfind(",", 0, max_len)
            cut = cut if cut > 40 else max_len
            out.append(p[: cut + 1].strip())
            p = p[cut + 1:].strip()
        if p:
            out.append(p)
    return out


class TTS:
    def __init__(self) -> None:
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        if self._model is None:
            import torch
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS

            device = settings.tts_device if torch.cuda.is_available() else "cpu"
            log.info("Loading Chatterbox Multilingual on %s…", device)
            self._model = ChatterboxMultilingualTTS.from_pretrained(device=device)
        return self._model

    def warmup(self) -> None:
        try:
            self._synth_sync("Привет!", "ru")
        except Exception:  # noqa: BLE001
            log.exception("TTS warmup failed (voice replies disabled until fixed)")

    def _synth_sync(self, text: str, lang: str) -> bytes:
        import numpy as np
        import soundfile as sf

        with self._lock:
            model = self._load()
            sr = model.sr
            pieces = []
            kwargs = dict(language_id=lang, exaggeration=settings.tts_exaggeration, cfg_weight=settings.tts_cfg_weight)
            if settings.tts_voice_ref:
                kwargs["audio_prompt_path"] = settings.tts_voice_ref
            for i, sentence in enumerate(split_sentences(text)):
                wav = model.generate(sentence, **kwargs).squeeze(0).detach().cpu().numpy()
                pieces.append(wav)
                pieces.append(np.zeros(int(sr * (0.18 if i % 2 else 0.25)), dtype=wav.dtype))  # natural pauses
            audio = np.concatenate(pieces) if pieces else np.zeros(sr // 2, dtype="float32")
        peak = float(np.abs(audio).max() or 1.0)
        audio = (audio / peak * 0.92).astype("float32")
        buf = io.BytesIO()
        sf.write(buf, audio, sr, format="OGG", subtype="OPUS")  # Telegram voice notes need OGG/Opus
        return buf.getvalue()

    async def synthesize(self, spoken_text: str) -> bytes | None:
        lang = detect_lang(spoken_text)
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
