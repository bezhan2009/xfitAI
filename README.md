# X-Fit AI assistant (Telegram + 1C:Fitness, fully local)

Telegram bot for **X-Fit Premium Dushanbe** that answers clients and staff using live data from
**1C:Fitness КОРП** (read-only) and a club knowledge base. Every reply is sent twice: as a text message,
then as a separate **voice message** with a natural human voice.

Everything runs on the local machine (GPU): no OpenAI or other cloud APIs.

| Part | Model / library | License | Why |
|---|---|---|---|
| Answers + tool calling | [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) via [Ollama](https://ollama.com) (Q4_K_M, ~2.6 GB VRAM) | Apache-2.0 | Best Russian tool-calling score among small Qwen3 models in an independent Habr test; fits a 6 GB GPU next to TTS |
| Voice replies (TTS) | [Chatterbox Multilingual](https://huggingface.co/ResembleAI/chatterbox) | MIT | Russian/English/Chinese, emotion control, optional voice cloning from a short sample; commercial use allowed |
| Incoming voice notes (STT) | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) `large-v3-turbo` | MIT | Accurate Russian/Tajik/English recognition, runs on CPU (int8) to leave the GPU for LLM + TTS |
| Telegram | aiogram 3 | MIT | |

## What it does

- **Clients**: ask anything about the club → answers only from `xfitbot/knowledge.md` (rules from the ТЗ: no prices,
  family discounts, installments, free tour, etc.). Never invents prices or promotions.
- **Club members**: press **📱 Поделиться номером** → the bot finds them in 1C by phone (Telegram-verified number only,
  typed numbers never unlock personal data) and answers about their cards: tariff, status, activation date, valid
  until, days left, freeze days, guest visits.
- **Hand-off**: tour booking, purchase, complaint, refund, discount, medical questions, photos/files → summary posted
  to the managers' chat (`MANAGER_CHAT_ID`). The bot never books or changes anything itself.
- **Staff** (Telegram IDs in `ADMIN_IDS`): analytics in plain language: "сколько карт продали в августе?",
  "сколько новых клиентов за неделю?", "найди клиента +992…", "найди Маматзода".
- **Voice in, voice out**: incoming voice notes are transcribed; every answer is also voiced. Before voicing, the LLM
  rewrites the text the way a person would say it aloud (numbers, dates and times as words in the right case,
  links replaced with "ссылку я прислала в сообщении"), then it is synthesised sentence by sentence with natural pauses.
- Human feel: typing / recording indicators, length-dependent pauses, short conversational replies, persona "Алина".

## 1C integration (read-only)

Uses the standard OData interface: `<base>/odata/standard.odata/`. Only `GET` requests are made.

Currently published in 1C and used:
- `Catalog_Контрагенты`: clients, phone search via `КонтактнаяИнформация/any(...)`
- `Document_ЧленствоПакетУслуг`: memberships / packages

**Ask 1C IT to also publish** (Администрирование → Общие настройки → Интеграция с другими программами →
Настройка стандартного интерфейса OData):
- `Справочник Номенклатура`: tariff **names** (without it the bot says "клубная карта на 12 месяцев" instead of
  "VIP Premium 12"). Picked up automatically, no code change. Until then you can fill `data/plans.json`
  by hand from `data/plans.example.json` (198 tariffs sold in 2026, sorted by sales).
- Optional: `Справочник Пользователи` (manager names), `Справочник Помещения`, the registers with real card
  end dates / freezes, which would make "действует до" exact instead of computed (activation date + term).

Known limits of this 1C build: `any()` filters on `ДополнительныеВладельцы` return HTTP 500, so family co-owner
cards are not looked up. The test base (`Fitness_test`) is a snapshot whose latest document is dated 2026-09-08.

## Setup (Windows, NVIDIA GPU)

Disk: ~15 GB free (PyTorch+CUDA ~5 GB, Qwen ~2.6 GB, Chatterbox ~3 GB, Whisper ~1.6 GB).

```powershell
# 1. Ollama + model
#    install from https://ollama.com/download, then:
ollama pull qwen3:4b-instruct-2507-q4_K_M

# 2. Python 3.11 venv
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
.venv\Scripts\python -m pip install -r requirements.txt

# 3. Config
copy .env.example .env   # fill TELEGRAM_TOKEN, ONEC_*, ADMIN_IDS, MANAGER_CHAT_ID

# 4. Run
.venv\Scripts\python run.py
```

Telegram allows one consumer per bot: if the bot token has a webhook set (e.g. by another platform), polling fails
with *409 Conflict* until that webhook is removed. Removing it disconnects the other platform.

### Options (`.env`)
- `TTS_VOICE_REF=path\to\voice.wav`: 7-15 s clean recording of a real person (with their consent) to clone a
  specific voice. Without it, the built-in Chatterbox voice is used.
- `TTS_EXAGGERATION` (0.25-1.0): more expressive / emotional speech. `TTS_CFG_WEIGHT`: lower = slower, calmer pace.
- `STT_DEVICE=cuda`: faster transcription if the GPU has spare memory.
- `TTS_ENABLED=false` / `STT_ENABLED=false`: text-only mode.

### Bot commands
`/start`, `/reset` (clear conversation), `/forget` (remove stored phone).

## Layout
```
run.py                 entry point
xfitbot/config.py      settings from .env
xfitbot/onec.py        read-only 1C OData client (clients, memberships, sales stats)
xfitbot/agent.py       LLM agent: system prompt, tools, history, spoken rewrite
xfitbot/voice.py       TTS (Chatterbox) + STT (faster-whisper), OGG/Opus encoding
xfitbot/bot.py         Telegram handlers
xfitbot/knowledge.md   club knowledge base (edit to add facts)
data/                  plans.json (optional tariff names), users.json (verified phones, not in git)
```

## Privacy
- Personal card data is shown only to the owner of a Telegram-verified phone number found in 1C.
- `.env` and `data/users.json` are git-ignored. Chat history lives in memory only.
