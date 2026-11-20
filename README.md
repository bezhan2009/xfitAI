# X-Fit AI assistant (Telegram + 1C:Fitness, fully local)

Telegram bot for **X-Fit Premium Dushanbe** that answers clients and staff using live data from
**1C:Fitness КОРП** (read-only) and a club knowledge base. Every reply is sent twice: as a text message,
then as a separate **voice message** with a natural human voice.

Everything runs on the local machine (GPU): no OpenAI or other cloud APIs.

| Part | Model / library | License | Notes |
|---|---|---|---|
| Answers + tool calling | **qwen2.5:7b** (Q4_K_M) via [Ollama](https://ollama.com), already installed on this PC | Apache-2.0 | GPU. Good Russian/English, follows facts placed next to the question. Can't write Tajik/Uzbek well, so those messages get a Russian reply (see `REPLY_LANGUAGES`) |
| Voice replies (TTS) | [Silero v5 CIS](https://github.com/snakers4/silero-models) `v5_cis_base_nostress` + [silero-stress](https://github.com/snakers4/silero-stress) | MIT | CPU, ~90 MB. Female voices: Russian `ru_ekaterina`, Tajik `tgk_onaoy`, Uzbek `uzb_saida` |
| Voice replies, other languages (optional) | [Chatterbox Multilingual](https://huggingface.co/ResembleAI/chatterbox) | MIT | GPU, ~3 GB + CUDA PyTorch. Used for English/Chinese if installed, or for everything with `TTS_ENGINE=chatterbox`; can clone a real voice |
| Incoming voice notes (STT) | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) `small` | MIT | CPU int8. Use `large-v3-turbo` for better Tajik if disk allows |
| Telegram | aiogram 3 | MIT | |

Upgrade path when there is ~15 GB free disk: `ollama pull qwen3:4b-instruct-2507-q4_K_M` (or a Russian-tuned 7-8B),
CUDA PyTorch + `pip install chatterbox-tts`, `STT_MODEL=large-v3-turbo`.

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

```powershell
# 1. Ollama with the model (on this PC the models live in D:\RelocatedData\Ollama\models)
$env:OLLAMA_MODELS = "D:\RelocatedData\Ollama\models"; ollama serve
ollama pull qwen2.5:7b        # only if not installed yet

# 2. Python 3.11 venv
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m pip install -r requirements.txt

# 3. Config
copy .env.example .env   # fill TELEGRAM_TOKEN, ONEC_*, ADMIN_IDS, MANAGER_CHAT_ID

# 4. Run (the Silero model downloads to models/ on first start)
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
`/start`, `/reset` (clear conversation), `/forget` (remove stored phone), `/mode` (admins: switch staff / customer view).

## Layout
```
run.py                 entry point
xfitbot/config.py      settings from .env
xfitbot/onec.py        read-only 1C OData client (clients, memberships, sales stats)
xfitbot/agent.py       LLM agent: system prompt, tools, history, spoken rewrite
xfitbot/voice.py       TTS (Silero / Chatterbox) + STT (faster-whisper), OGG/Opus encoding
xfitbot/bot.py         Telegram handlers
xfitbot/knowledge.md   club knowledge base (edit to add facts)
data/                  plans.json (optional tariff names), users.json (verified phones, not in git)
```

## Privacy
- Personal card data is shown only to the owner of a Telegram-verified phone number found in 1C.
- `.env` and `data/users.json` are git-ignored. Chat history lives in memory only.
