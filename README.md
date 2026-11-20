# X-Fit AI assistant (Telegram + 1C:Fitness, fully local)

Telegram bot for **X-Fit Premium Dushanbe** that answers clients and staff using live data from
**1C:Fitness КОРП** (read-only) and a club knowledge base. Every reply is sent twice: as a text message,
then as a separate **voice message** with a natural, emotional human voice. Clients can also write or speak
in **Russian, Tajik, Uzbek or English** (Tajik/Uzbek are understood and, for now, answered in Russian).

Everything runs on the local machine: no OpenAI or other cloud APIs.

| Part | Model / library | License | Notes |
|---|---|---|---|
| Answers + tool calling | **qwen2.5:7b** (Q4_K_M) via [Ollama](https://ollama.com) | Apache-2.0 | GPU (~4 GB VRAM). Thinks and answers in Russian/English, follows facts placed next to the question |
| Tajik / Uzbek | [TranslateGemma 4B](https://ollama.com/library/translategemma) via Ollama | Gemma terms (commercial use allowed) | CPU (`MT_NUM_GPU=0`), 3.3 GB. Tajik/Uzbek → English is accurate, so clients are understood well (also informal spelling without ҳ/ҷ/ӣ). Its *written* Tajik is not good enough yet, so by default replies are in Russian (`TRANSLATE_REPLIES=false`) |
| Voice replies (TTS) | [Silero v5 CIS](https://github.com/snakers4/silero-models) `v5_cis_base_nostress` + [silero-stress](https://github.com/snakers4/silero-stress) | MIT | CPU, ~90 MB. Female voices: Russian `ru_ekaterina`, Tajik `tgk_onaoy`, Uzbek `uzb_saida`. Emotions via SSML |
| Voice replies, other languages (optional) | [Chatterbox Multilingual](https://huggingface.co/ResembleAI/chatterbox) | MIT | GPU, ~3 GB + CUDA PyTorch. English/Chinese if installed; can clone a real voice |
| Incoming voice (STT) | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) `small` | MIT | CPU int8, detects the language. `large-v3-turbo` is more accurate for Tajik if disk allows |
| Telegram | aiogram 3 | MIT | |

## What it does

- **Clients** ask anything about the club → answers only from `xfitbot/knowledge.md` (rules from the ТЗ: no prices,
  family discounts, installments, free tour…). Never invents prices or promotions.
- **Club members**: write their phone number (`+992…`, any format) or press **📱 Поделиться номером** → the bot
  looks them up in 1C *in code, before the AI answers* (the model can't skip the lookup or claim "not found"),
  then answers about their cards: tariff, status, activation date, valid until, days left, freeze days, guest visits.
- **Staff** (Telegram IDs in `ADMIN_IDS`): "сколько карт продали в августе?", "сколько новых клиентов за неделю?",
  "найди клиента +992…", "найди Маматзода". `/mode` switches an admin to the customer view and back.
- **Hand-off**: tour booking, purchase, complaint, refund, discount, medical questions, photos/files → summary posted
  to the managers' chat (`MANAGER_CHAT_ID`). The bot never books or changes anything in 1C.
- **Voice in**: voice notes, round videos, audio tracks and audio files sent as attachments (ogg, mp3, m4a, wav, …,
  up to 5 minutes) are transcribed and answered; the spoken language is detected.
- **Voice out**: every answer is also sent as a voice message:
  1. `speech_text.py` turns numbers into words deterministically: phones digit-group by digit-group
     ("плюс девятьсот девяносто два, девятьсот восемь, восемьдесят два…"), dates and years in the right case
     ("до седьмого октября две тысячи двадцать шестого года"), times ("с девяти утра до девяти вечера"),
     percentages, links dropped, Latin brand names transliterated (X-Fit → «Икс Фит»).
  2. The LLM only *tags the emotion* of each sentence (joy, enthusiasm, warmth, sympathy, question, calm) and may add
     up to two short interjections («Ой,», «Так,», «Хм…»). It never rewrites the facts.
  3. Each emotion maps to speech rate, pitch and pause length (SSML) in the Silero voice.
- **Human feel**: typing / recording indicators, length-dependent pauses, short conversational replies, persona "Алина".

## 1C integration (read-only)

Uses the standard OData interface: `<base>/odata/standard.odata/`. Only `GET` requests are made.

Everything currently published in 1C is used:
- `Catalog_Контрагенты`: clients; phone search across `НомерТелефона`, `НомерТелефонаБезКодов` and `Телефон`
  (1C stores numbers inconsistently), name search for staff.
- `Document_ЧленствоПакетУслуг`: memberships / packages: term, activation, computed end date, freeze days,
  guest visits, time restrictions, sales statistics.

**To integrate more of 1C, publish these objects** (1C: Администрирование → Общие настройки → Интеграция с другими
программами → Настройка стандартного интерфейса OData → tick the object → Сохранить). The bot picks most of them
up without code changes:
- `Справочник Номенклатура`: tariff **names** (now the bot says "клубная карта на 12 месяцев" instead of the real name).
  Until then you can fill `data/plans.json` from `data/plans.example.json` (198 tariffs sold in 2026).
- `Справочник Пользователи` / `Сотрудники`: names of the assigned managers.
- Registers with real card end dates, freezes and visits (e.g. `РегистрСведений СтатусыЧленств`, visits register):
  makes "действует до" exact instead of computed, and enables "когда я последний раз был в клубе".
- `Справочник Помещения`, schedule documents: zones and class schedule.

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

# 3. Tajik/Uzbek translator (3.3 GB), optional
ollama pull translategemma:4b

# 4. Config
copy .env.example .env   # fill TELEGRAM_TOKEN, ONEC_*, ADMIN_IDS, MANAGER_CHAT_ID

# 5. Run (the Silero voice model downloads to models/ on first start)
.venv\Scripts\python run.py
```

Telegram allows one consumer per bot: if the bot token has a webhook set (e.g. by another platform), polling fails
with *409 Conflict* until that webhook is removed. Removing it disconnects the other platform.

### Options (`.env`)
- `TTS_SPEAKER_RU`: female `ru_ekaterina` (default), `ru_vika`, `ru_oksana`, `ru_zara`, `ru_onaoy`, `ru_alfia`,
  `ru_karina`; male `ru_igor`, `ru_roman`, `ru_dmitriy`. `TTS_SPEAKER_TG` (`tgk_onaoy`, `tgk_safarhuja`), `TTS_SPEAKER_UZ`.
- `TRANSLATE_LANGUAGES=tg,uz`: languages understood through the translator. `REPLY_LANGUAGES=ru,en,zh`: languages the
  LLM writes itself. Anything else gets Russian.
- `TRANSLATE_REPLIES=true`: answer Tajik/Uzbek clients in their language (machine translation + Tajik/Uzbek voice).
  Off by default: `translategemma:4b` writes Tajik with errors. With ~8 GB free disk try `MT_MODEL=translategemma:12b`
  (slow on this GPU, ~30-60 s per reply) and review the output with a native speaker first.
- `MT_MODEL`, `MT_NUM_GPU` (0 = CPU).
- `TTS_ENGINE=chatterbox`, `TTS_VOICE_REF=voice.wav` (Chatterbox only): clone a real person's voice (with consent).
- `STT_MODEL` (`small`, `medium`, `large-v3-turbo`), `STT_DEVICE=cuda`.
- `TTS_ENABLED=false` / `STT_ENABLED=false`: text-only mode.

### Bot commands
`/start`, `/reset` (clear conversation), `/forget` (remove stored phone), `/mode` (admins: staff / customer view).

## Layout
```
run.py                   entry point
xfitbot/config.py        settings from .env
xfitbot/onec.py          read-only 1C OData client (clients, memberships, sales stats)
xfitbot/agent.py         LLM agent: prompt, 1C lookups, tools, hand-off, translation, emotion tagging
xfitbot/speech_text.py   numbers/dates/phones/brands -> speakable words; emotion helpers
xfitbot/voice.py         TTS (Silero with SSML emotions / Chatterbox) + STT (faster-whisper), OGG/Opus
xfitbot/translate.py     TranslateGemma (Ollama) translator for Tajik/Uzbek
xfitbot/bot.py           Telegram handlers
xfitbot/knowledge.md     club knowledge base (edit to add facts)
data/                    plans.json (optional tariff names), users.json (linked phones, not in git), bot.log
```

## Privacy
- Personal card data is only given for a phone number found in 1C; no full names or other clients' data to clients.
- `.env`, `data/users.json`, logs and models are git-ignored. Chat history lives in memory only.
