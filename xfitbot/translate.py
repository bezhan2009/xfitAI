"""Local machine translation for languages the main LLM can't write well (Tajik, Uzbek).

Model: TranslateGemma 4B (Google, Gemma 3 based, 55 languages incl. Tajik and Uzbek) via Ollama, run on the CPU so
it doesn't push the main model off the GPU. The main LLM thinks and answers in Russian with the 1C facts; the answer
is translated for the client, and incoming messages are translated to Russian first so the LLM understands them.
"""
from __future__ import annotations

import logging
import re

import httpx

from .config import settings

log = logging.getLogger(__name__)

LANGS = {"ru": "Russian", "tg": "Tajik", "uz": "Uzbek", "en": "English"}
PROMPT = (
    "You are a professional {src} ({sc}) to {tgt} ({tc}) translator. Your goal is to accurately convey the meaning "
    "and nuances of the original {src} text while adhering to {tgt} grammar, vocabulary, and cultural sensitivities. "
    "Produce only the {tgt} translation, without any additional explanations or commentary. Please translate the "
    "following {src} text into {tgt}:\n\n\n{text}"
)
# values that must survive translation unchanged
KEEP_RE = re.compile(r"(https?://\S+|t\.me/\S+|\+?\d[\d\s\-()]{6,18}\d|\d{1,2}:\d{2}|\d+\s*%)")


class Translator:
    def __init__(self) -> None:
        self.http = httpx.AsyncClient(base_url=settings.ollama_url, timeout=httpx.Timeout(180, connect=10))

    @staticmethod
    def available() -> bool:
        try:
            r = httpx.get(f"{settings.ollama_url}/api/tags", timeout=5)
            names = {m["name"] for m in r.json().get("models", [])}
            return settings.mt_model in names or f"{settings.mt_model}:latest" in names
        except Exception:  # noqa: BLE001
            return False

    async def translate(self, text: str, target: str, source: str = "ru") -> str:
        if not text.strip() or source == target:
            return text
        prompt = PROMPT.format(src=LANGS.get(source, source), sc=source, tgt=LANGS.get(target, target), tc=target,
                               text=text.strip())
        r = await self.http.post("/api/chat", json={
            "model": settings.mt_model, "stream": False, "keep_alive": "60m",
            "messages": [{"role": "user", "content": prompt}],
            # run on CPU (num_gpu 0) so qwen stays on the GPU; low temperature for faithful translation
            "options": {"temperature": 0.1, "num_gpu": settings.mt_num_gpu, "num_ctx": 2048},
        })
        r.raise_for_status()
        out = (r.json()["message"].get("content") or "").strip()
        lost = [k.strip() for k in KEEP_RE.findall(text) if re.sub(r"\D", "", k) and
                re.sub(r"\D", "", k) not in re.sub(r"\D", "", out)]
        if lost:
            log.warning("translation lost values %s; keeping them appended", lost)
        return out or text

    async def warmup(self) -> None:
        try:
            out = await self.translate("Здравствуйте!", "tg")
            log.info("Translator ready (%s): %s", settings.mt_model, out)
        except Exception:  # noqa: BLE001
            log.exception("translator warmup failed")
