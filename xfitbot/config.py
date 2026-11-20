"""Settings loaded from .env (never commit real secrets)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    return default if v is None else v.strip().lower() in {"1", "true", "yes", "on"}


def _ids(name: str) -> set[int]:
    return {int(x) for x in os.getenv(name, "").replace(";", ",").split(",") if x.strip().lstrip("-").isdigit()}


@dataclass(frozen=True)
class Settings:
    telegram_token: str = os.getenv("TELEGRAM_TOKEN", "")
    admin_ids: set[int] = field(default_factory=lambda: _ids("ADMIN_IDS"))
    manager_chat_id: int | None = int(os.getenv("MANAGER_CHAT_ID")) if os.getenv("MANAGER_CHAT_ID", "").lstrip("-").isdigit() else None

    onec_base_url: str = os.getenv("ONEC_BASE_URL", "").rstrip("/")
    onec_user: str = os.getenv("ONEC_USER", "")
    onec_password: str = os.getenv("ONEC_PASSWORD", "")

    ollama_url: str = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
    llm_model: str = os.getenv("LLM_MODEL", "qwen3:4b-instruct-2507-q4_K_M")
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.4"))

    tts_enabled: bool = _bool("TTS_ENABLED", True)
    tts_voice_ref: str = os.getenv("TTS_VOICE_REF", "")  # optional wav of a real voice (with consent)
    tts_device: str = os.getenv("TTS_DEVICE", "cuda")
    tts_exaggeration: float = float(os.getenv("TTS_EXAGGERATION", "0.55"))
    tts_cfg_weight: float = float(os.getenv("TTS_CFG_WEIGHT", "0.45"))

    stt_enabled: bool = _bool("STT_ENABLED", True)
    stt_model: str = os.getenv("STT_MODEL", "large-v3-turbo")
    stt_device: str = os.getenv("STT_DEVICE", "cpu")

    data_dir: Path = ROOT / "data"
    timezone: str = os.getenv("TZ_NAME", "Asia/Dushanbe")


settings = Settings()
settings.data_dir.mkdir(exist_ok=True)
