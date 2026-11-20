"""Telegram bot: text reply first, then the same reply as a voice note."""
from __future__ import annotations

import asyncio
import logging
import random

from aiogram import Bot, Dispatcher, F
from aiogram.enums import ChatAction
from aiogram.filters import Command, CommandStart
from aiogram.types import (BufferedInputFile, KeyboardButton, Message, ReplyKeyboardMarkup,
                           ReplyKeyboardRemove)
from aiogram.utils.chat_action import ChatActionSender

from .agent import Agent, UserState, UserStore
from .config import settings
from .onec import OneC
from .translate import Translator
from .voice import STT, TTS

log = logging.getLogger(__name__)

SHARE_KB = ReplyKeyboardMarkup(
    keyboard=[[KeyboardButton(text="📱 Поделиться номером", request_contact=True)]],
    resize_keyboard=True, one_time_keyboard=True,
)

onec = OneC()
store = UserStore(settings.data_dir / "users.json")
tts = TTS() if settings.tts_enabled else None
stt = STT() if settings.stt_enabled else None
translator = Translator() if settings.translate_languages and Translator.available() else None
AUDIO_EXT = (".ogg", ".oga", ".opus", ".mp3", ".m4a", ".wav", ".aac", ".amr", ".flac", ".webm", ".wma", ".3gp")
MAX_AUDIO_SECONDS = 300
bot = Bot(settings.telegram_token)
dp = Dispatcher()
_user_locks: dict[int, asyncio.Lock] = {}


async def notify_manager(st: UserState, reason: str, summary: str) -> None:
    if not settings.manager_chat_id:
        log.info("handoff (no MANAGER_CHAT_ID set): %s | %s", reason, summary)
        return
    who = st.client_name or st.first_name or "клиент"
    phone = f"+{st.phone}" if st.phone else "номер не подтверждён"
    text = (f"🔔 Передача менеджеру\nКлиент: {who} ({phone}), tg id {st.user_id}\n"
            f"Причина: {reason}\nSummary: {summary}\nНаписать: tg://user?id={st.user_id}")
    try:
        await bot.send_message(settings.manager_chat_id, text)
    except Exception:  # noqa: BLE001
        log.exception("could not notify manager chat")


agent = Agent(onec, store, notify_manager, translator)


_client_view: set[int] = set()  # admins who switched to the customer view with /mode


def is_staff(user_id: int) -> bool:
    return user_id in settings.admin_ids and user_id not in _client_view


@dp.message(Command("mode"))
async def on_mode(message: Message) -> None:
    uid = message.from_user.id
    if uid not in settings.admin_ids:
        return
    if uid in _client_view:
        _client_view.discard(uid)
        await message.answer("Режим сотрудника: доступна аналитика 1С (продажи, новые клиенты, поиск клиента).")
    else:
        _client_view.add(uid)
        await message.answer("Режим клиента: бот отвечает вам как обычному клиенту. /mode — вернуться.")
    store.get(uid).history.clear()


async def human_pause(text: str) -> None:
    """A short, length-dependent 'typing' pause so replies don't feel instant and robotic."""
    await asyncio.sleep(min(0.6 + len(text) / 90, 3.0) + random.uniform(0, 0.5))


async def answer(message: Message, user_text: str, lang_hint: str | None = None) -> None:
    uid = message.from_user.id
    st = store.get(uid, message.from_user.first_name or "")
    lock = _user_locks.setdefault(uid, asyncio.Lock())
    async with lock:
        async with ChatActionSender.typing(bot=bot, chat_id=message.chat.id):
            try:
                reply = await agent.reply(st, user_text, is_staff(uid), lang_hint)
                log.info("dialog %s | Q: %s | A: %s", uid, user_text[:300].replace("\n", " "), reply.replace("\n", " "))
            except Exception:  # noqa: BLE001
                log.exception("agent failed")
                await message.answer("Ой, у меня небольшая техническая заминка. Напишите, пожалуйста, ещё раз через минуту 🙏")
                return
            await human_pause(reply)
        kb = SHARE_KB if not st.phone and "Поделиться номером" in reply else None
        await message.answer(reply, reply_markup=kb, disable_web_page_preview=True)
        if tts:
            asyncio.create_task(send_voice(message, reply, st.reply_lang))


async def send_voice(message: Message, reply: str, lang: str = "ru") -> None:
    if not tts.supports(lang):
        log.info("no voice for language %s", lang)
        return
    try:
        async with ChatActionSender.record_voice(bot=bot, chat_id=message.chat.id):
            segments = await agent.spoken_version(reply, lang)
            log.info("voice %s: %s", lang, " | ".join(f"[{e}] {t}" for e, t in segments)[:500])
            audio = await tts.synthesize(segments, lang)
        if audio:
            await message.answer_voice(BufferedInputFile(audio, filename="reply.ogg"))
    except Exception:  # noqa: BLE001
        log.exception("voice reply failed")


@dp.message(CommandStart())
async def on_start(message: Message) -> None:
    st = store.get(message.from_user.id, message.from_user.first_name or "")
    st.history.clear()
    name = message.from_user.first_name or ""
    greet = (f"Здравствуйте{', ' + name if name else ''}! Я Алина, ассистент клуба X-Fit Premium Dushanbe 😊 "
             "Расскажу о клубе, картах и помогу записаться на бесплатный тур. А если вы уже член клуба — "
             "поделитесь номером, и я посмотрю вашу карту в базе.")
    await message.answer(greet, reply_markup=None if st.phone else SHARE_KB)
    if tts:
        asyncio.create_task(send_voice(message, greet))


@dp.message(Command("reset"))
async def on_reset(message: Message) -> None:
    store.get(message.from_user.id).history.clear()
    await message.answer("Начнём сначала 🙂 Чем могу помочь?")


@dp.message(Command("forget"))
async def on_forget(message: Message) -> None:
    st = store.get(message.from_user.id)
    st.phone = st.client_key = st.client_name = st.client_data = None
    st.history.clear()
    store.save(st)
    await message.answer("Удалила ваш номер из бота. Чтобы снова посмотреть карту, поделитесь номером.",
                         reply_markup=SHARE_KB)


@dp.message(F.contact)
async def on_contact(message: Message) -> None:
    c = message.contact
    if c.user_id != message.from_user.id:
        await message.answer("Пожалуйста, поделитесь именно своим номером — кнопкой ниже 🙏", reply_markup=SHARE_KB)
        return
    st = store.get(message.from_user.id, message.from_user.first_name or "")
    async with ChatActionSender.typing(bot=bot, chat_id=message.chat.id):
        try:
            res = await agent.verify_contact(st, c.phone_number)
        except Exception:  # noqa: BLE001
            log.exception("1C lookup failed")
            await message.answer("Спасибо! База клуба сейчас не отвечает, попробую чуть позже.",
                                 reply_markup=ReplyKeyboardRemove())
            return
    await message.answer("Спасибо, номер получила ✅", reply_markup=ReplyKeyboardRemove())
    prompt = ("[Системное событие: клиент только что подтвердил номер. "
              + ("Он найден в 1С — коротко поприветствуй по имени и одним-двумя предложениями скажи главное "
                 "по его картам из блока ДАННЫЕ КЛИЕНТА ИЗ 1С (есть ли действующая и до какого числа).]" if res == "found" else
                 "В 1С его нет — значит, он у нас новенький. Тепло скажи об этом и спроси, что его интересует.]"))
    await answer(message, prompt)


@dp.message(F.voice | F.audio | F.video_note | F.document.func(lambda d: d is not None and ((d.mime_type or "").startswith("audio/")
                                                               or (d.file_name or "").lower().endswith(AUDIO_EXT))))
async def on_voice(message: Message) -> None:
    """Voice notes, round videos and audio files (also sent as attachments): transcribe and answer them."""
    if not stt:
        await message.answer("Голосовые пока не слушаю — напишите, пожалуйста, текстом 🙏")
        return
    media = message.voice or message.audio or message.video_note or message.document
    duration = getattr(media, "duration", None) or 0
    if duration > MAX_AUDIO_SECONDS:
        await message.answer("Ой, запись очень длинная 🙈 Можете коротко сказать или написать, что вас интересует?")
        return
    async with ChatActionSender.typing(bot=bot, chat_id=message.chat.id):
        buf = await bot.download(media)
        try:
            text, lang = await stt.transcribe(buf.read())
        except Exception:  # noqa: BLE001
            log.exception("transcription failed")
            text, lang = "", ""
    if not text:
        await message.answer("Не расслышала 🙈 Можете повторить или написать текстом?")
        return
    if message.caption:
        text = f"{message.caption}\n{text}"
    log.info("voice from %s (%s, %ss): %s", message.from_user.id, lang, duration, text)
    await answer(message, text, lang_hint=lang)


@dp.message(F.photo | F.document | F.video | F.sticker)
async def on_media(message: Message) -> None:
    st = store.get(message.from_user.id, message.from_user.first_name or "")
    await notify_manager(st, "клиент прислал фото/файл", message.caption or "(без подписи)")
    await message.answer("Файлы и фото я, к сожалению, не открываю — передала менеджеру, он посмотрит и ответит здесь.")


@dp.message(F.text)
async def on_text(message: Message) -> None:
    await answer(message, message.text)


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log.info("1C reachable: %s", await onec.ping())
    if tts:
        await asyncio.to_thread(tts.warmup)
    if translator:
        await asyncio.to_thread(translator.warmup)
    else:
        log.info("Translator not installed: Tajik/Uzbek messages get Russian replies")
    me = await bot.get_me()
    log.info("Bot @%s started", me.username)
    try:
        await dp.start_polling(bot)
    finally:
        await agent.close()
        await onec.close()
