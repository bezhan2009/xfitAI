"""Local machine translation for languages the LLM can't write well (Tajik, Uzbek).

Model: google/madlad400-3b-mt (Apache-2.0), CTranslate2 int8 build, runs on CPU. The LLM thinks and answers in
Russian (with the 1C facts), and the answer is translated for the client. Incoming messages are translated to
Russian first so the LLM understands them.
"""
from __future__ import annotations

import asyncio
import logging
import re
import threading

from .config import ROOT, settings

log = logging.getLogger(__name__)

MODEL_DIR = ROOT / "models" / "madlad400-3b-ct2-int8"
# keep these untouched by the translator: phones, times, dates, numbers with %, links, brand names
PROTECT_RE = re.compile(
    r"(https?://\S+|t\.me/\S+|\+?\d[\d\s\-()]{6,18}\d|\d{1,2}:\d{2}|\d+\s*%|\d+|X-Fit Premium Dushanbe|"
    r"Family Solo Premium|Family Premium|X-DRIVE|X-Fit|VIP|«Салом»|Алиф Банк)")


class Translator:
    def __init__(self) -> None:
        self._model = None
        self._sp = None
        self._lock = threading.Lock()

    @staticmethod
    def available() -> bool:
        return (MODEL_DIR / "model.bin").exists() and (MODEL_DIR / "spiece.model").exists()

    def _load(self):
        if self._model is None:
            import ctranslate2
            import sentencepiece as spm

            threads = settings.mt_threads
            log.info("Loading MADLAD-400 translator (CPU, %s threads)…", threads)
            self._sp = spm.SentencePieceProcessor(model_file=str(MODEL_DIR / "spiece.model"))
            self._model = ctranslate2.Translator(str(MODEL_DIR), device="cpu", compute_type="int8",
                                                 intra_threads=threads)
        return self._model

    def _translate_sync(self, text: str, target: str) -> str:
        # protect values so the translator can't alter them
        keep: list[str] = []

        def hold(m: re.Match) -> str:
            keep.append(m.group())
            return f"⟦{len(keep) - 1}⟧"

        masked = PROTECT_RE.sub(hold, text)
        paragraphs = [p for p in masked.split("\n") if p.strip()]
        with self._lock:
            model = self._load()
            batch = [[f"<2{target}>"] + self._sp.encode(p, out_type=str) for p in paragraphs]
            results = model.translate_batch(batch, beam_size=2, max_decoding_length=400,
                                            repetition_penalty=1.1)
        out = "\n".join(self._sp.decode(r.hypotheses[0]) for r in results)

        def restore(m: re.Match) -> str:
            i = int(m.group(1))
            return keep[i] if i < len(keep) else ""

        out = re.sub(r"⟦\s*(\d+)\s*⟧", restore, out)
        # if a placeholder got lost, append nothing but log it
        missing = [k for k in keep if k not in out]
        if missing:
            log.warning("translator dropped protected values: %s", missing)
        return out.strip()

    async def translate(self, text: str, target: str) -> str:
        return await asyncio.to_thread(self._translate_sync, text, target)

    def warmup(self) -> None:
        try:
            self._translate_sync("Здравствуйте!", "tg")
            log.info("Translator ready")
        except Exception:  # noqa: BLE001
            log.exception("translator warmup failed")
