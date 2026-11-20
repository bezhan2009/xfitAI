"""LLM agent: local Qwen via Ollama with read-only 1C tools."""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Awaitable, Callable
from zoneinfo import ZoneInfo

import httpx

from .config import settings
from .onec import OneC, normalize_phone

log = logging.getLogger(__name__)

KNOWLEDGE = (Path(__file__).parent / "knowledge.md").read_text("utf-8")
MAX_HISTORY = 16
MONTHS = ["января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа",
          "сентября", "октября", "ноября", "декабря"]
WEEKDAYS = ["понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье"]

SYSTEM_PROMPT = """Ты — Алина, ассистент фитнес-клуба X-Fit Premium Dushanbe в Telegram. Ты девушка: о себе говоришь только в женском роде («я посмотрела», «я передала», «я могла бы»). Общаешься как живой, тёплый и внимательный администратор клуба: просто, по-человечески, без канцелярита.

Как писать:
- ЯЗЫК ОТВЕТА: {language}. Отвечай только на этом языке.
- Коротко: 1–3 предложения. Без списков, заголовков, markdown — это переписка в мессенджере.
- Сначала ответ по сути, потом не больше одного вопроса. Не веди анкету, не переспрашивай то, что уже известно.
- Естественно, как человек: «Смотрю… да, вижу вашу карту», «Понимаю», «Хороший вопрос». Максимум одно эмодзи. Не здоровайся повторно, если уже здоровалась в этом диалоге. Обращайся по имени, если оно известно.
- Даты пиши как «7 октября». Никогда не показывай GUID, ключи, служебные поля.
- Только когда новый клиент спрашивает о покупке карты или цене — закончи ОДНИМ вопросом с двумя вариантами времени тура («Вам удобнее сегодня вечером или завтра днём?»). Действующему члену клуба, который спрашивает про свою карту, тур НЕ предлагай.

Правила по данным:
- Отвечай ТОЛЬКО по базе знаний и по блоку «ДАННЫЕ КЛИЕНТА ИЗ 1С». Ничего не выдумывай: ни цен, ни дат, ни тарифов, ни акций, ни расписания, ни условий. Если в данных чего-то нет — так и скажи и предложи уточнить у менеджера.
- Про абонемент клиента говори только то, что есть в блоке «ДАННЫЕ КЛИЕНТА ИЗ 1С». Если блока нет — у тебя НЕТ данных о его карте: попроси написать номер телефона, на который оформлена карта (+992…), или нажать кнопку «📱 Поделиться номером».
- Поиск по номеру делает система, результат придёт в пометке [Поиск в 1С …]. НИКОГДА не говори, что клиента нет в базе, если такой пометки нет.
- Заморозка: возможна только в пределах «дней_заморозки_по_тарифу». Если там 0 — по тарифу заморозка не предусмотрена. Саму заморозку, продление, переоформление, запись на тур оформляет менеджер, не ты.
- Тур: когда клиент назвал день и примерное время (с 9:00 до 21:00) — вызови handoff_to_manager с summary и скажи, что передала менеджеру, он подтвердит время в этом чате. Если назвал только день — спроси время двумя вариантами.
- Жалоба, возврат, скидка, мед. вопрос, просьба позвать человека, нет ответа в базе — вызови handoff_to_manager.

Примеры тона:
Клиент: Сколько стоит безлимит?
Алина: Цена зависит от срока и формата карты, точный расчёт сделает менеджер на бесплатном туре по клубу — заодно всё покажет. Вам удобнее заглянуть сегодня вечером или завтра днём?
Клиент: Завтра вечером
Алина: Отлично! Во сколько удобнее — к 18:00 или ближе к 20:00?
Клиент: Можно заморозить карту?
Алина: Смотрю… по вашей карте заморозка, к сожалению, не предусмотрена. Если хотите, передам вопрос менеджеру — он подскажет варианты.

Сегодня: {today}. Часовой пояс: Душанбе.
Клиент: {client}
{client_data}{staff}
=== БАЗА ЗНАНИЙ ===
{knowledge}
"""

STAFF_NOTE = """Собеседник — сотрудник клуба (режим персонала). Ему можно отвечать на аналитические вопросы по 1С \
(продажи, новые клиенты, поиск клиента по телефону или ФИО) через инструменты staff_*. Если сотрудник называет ФИО \
или фамилию — обязательно вызови staff_find_client. Поиск по телефону система делает сама (пометка [Поиск в 1С …]). \
Отвечай точно цифрами, кратко, можно списком. Правила про тур и вопрос «сегодня или завтра» к сотруднику не применяй."""


HANDOFF_RE = re.compile(r"(передала|передаю|уже передал|менеджер (свяжется|подтвердит|подключится|ответит)|"
                        r"passed (it|this|your).{0,30} manager|manager will (contact|confirm|reply))", re.I)

LANG_NAMES = {"ru": "русский", "en": "английский (English)", "tg": "таджикский (тоҷикӣ)",
              "uz": "узбекский (o'zbek)", "zh": "китайский (中文)"}


def user_language(text: str) -> str:
    if re.search(r"[一-鿿]", text):
        return "zh"
    if re.search(r"[ғӣқӯҳҷҒӢҚӮҲҶ]", text) or re.search(r"\b(салом|ташаккур|чӣ|чанд|ман|шумо|ҳаст|аст)\b", text, re.I):
        return "tg"
    lat = len(re.findall(r"[A-Za-z]", text))
    cyr = len(re.findall(r"[А-Яа-яЁё]", text))
    if lat > cyr:
        if re.search(r"(o'|g'|\bsalom\b|\brahmat\b|\bqancha\b|\bmen\b|\bsiz\b)", text, re.I):
            return "uz"
        return "en"
    return "ru"


PHONE_RUN_RE = re.compile(r"\+?\d[\d\s\-()]{7,20}\d")


def find_phone(text: str) -> str | None:
    """First phone-like number in a message (9 local or 12 digits with 992), normalised to 992XXXXXXXXX."""
    for m in PHONE_RUN_RE.finditer(text):
        digits = re.sub(r"\D", "", m.group())
        if len(digits) == 9 or (len(digits) == 12 and digits.startswith("992")):
            return normalize_phone(digits)
    return None


def _kb_sections() -> list[tuple[str, str]]:
    parts = re.split(r"^## ", KNOWLEDGE, flags=re.M)[1:]
    return [(p.split("\n", 1)[0].strip(), p.split("\n", 1)[1].strip()) for p in parts if "\n" in p]


KB_SECTIONS = _kb_sections()
KB_KEYWORDS = {
    "Цены": r"цен|стоит|стоимост|сколько.*(карт|абонем)|price|cost|нарх|narx|qancha|чанд пул|чанд сомон",
    "Что можно называть про деньги": r"сем|family|дет|ребён|ребен|жен|муж|скидк|рассроч|алиф|салом.*карт|discount|"
                                     r"installment|оил|kids|child",
    "Визиты": r"тур|визит|посмотр|попроб|прийти|приду|гост|бесплат|trial|visit|tour|guest",
    "Дети": r"дет|ребён|ребен|сын|доч|kid|child|бача|фарзанд",
    "Здоровье": r"спин|бол|здоров|врач|травм|беремен|давлен|сердц|health|doctor|pain",
    "Клуб": r"адрес|где|наход|время работ|часы|режим|телефон|контакт|расписан|инстаграм|address|where|hours|schedule|"
            r"куҷо|кучо|суроға|ҷойгир|қаерда|manzil|qayer|соат|vaqt|jadval",
    "Продукты (виды карт)": r"карт|абонем|тариф|vip|безлимит|x-drive|выходн|дневн|card|membership",
}


def relevant_facts(text: str) -> str:
    low = text.lower()
    picked = [f"{title}: {body}" for title, body in KB_SECTIONS
              if title in KB_KEYWORDS and re.search(KB_KEYWORDS[title], low)]
    return "\n".join(picked)[:2500]


TOUR_Q_RE = re.compile(r"\s*Вам удобнее (заглянуть )?сегодня вечером или завтра днём[^?]*\?\s*$")


def tidy_answer(answer: str, st: "UserState") -> str:
    """Remove the stock tour question when it is redundant (members, or another question already asked)."""
    m = TOUR_Q_RE.search(answer)
    if m:
        before = answer[: m.start()]
        if st.client_key or "?" in before:
            answer = before.rstrip()
    answer = re.sub(r"(?i)(?<=\w)assistant\s*:\s*", "", answer)  # stray role token glued into text by the model
    return re.sub(r"\n{3,}", "\n\n", answer).strip()


def times_to_words(text: str) -> str:
    """19:00 -> «семь вечера», 9:30 -> «девять тридцать утра» (so the rewrite step can't change the value)."""
    try:
        from num2words import num2words
    except ImportError:
        return text

    def conv(m: re.Match) -> str:
        h, mi = int(m.group(1)), int(m.group(2))
        if h > 23 or mi > 59:
            return m.group(0)
        part = "ночи" if h < 5 else "утра" if h < 12 else "дня" if h < 17 else "вечера"
        h12 = h % 12 or 12
        hw = "час" if h12 == 1 else num2words(h12, lang="ru")
        return f"{hw}{' ' + num2words(mi, lang='ru') if mi else ''} {part}"

    return re.sub(r"\b(\d{1,2}):(\d{2})\b", conv, text)


def human_date(iso: str | None) -> str | None:
    if not iso:
        return None
    d = date.fromisoformat(iso[:10])
    return f"{d.day} {MONTHS[d.month - 1]} {d.year}"


def _humanize(obj: Any) -> Any:
    """Make tool output easier for a small model: human dates, no internal keys."""
    if isinstance(obj, list):
        return [_humanize(x) for x in obj]
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k.startswith("_"):
                continue
            if isinstance(v, str) and len(v) == 10 and v[4] == "-" and v[7] == "-" and k != "период":
                v = human_date(v)
            out[k] = _humanize(v)
        return out
    return obj


@dataclass
class UserState:
    user_id: int
    first_name: str = ""
    phone: str | None = None  # verified via Telegram contact
    client_key: str | None = None
    client_name: str | None = None
    history: list[dict] = field(default_factory=list)
    language: str = "ru"
    greet_salom: bool = False
    client_data: str | None = None
    client_data_at: float = 0.0


class UserStore:
    """Tiny JSON store for verified phones (no chat history on disk)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._data: dict[str, dict] = json.loads(path.read_text("utf-8")) if path.exists() else {}
        self._live: dict[int, UserState] = {}

    def get(self, user_id: int, first_name: str = "") -> UserState:
        st = self._live.get(user_id)
        if st is None:
            saved = self._data.get(str(user_id), {})
            st = UserState(user_id=user_id, first_name=first_name, phone=saved.get("phone"),
                           client_key=saved.get("client_key"), client_name=saved.get("client_name"))
            self._live[user_id] = st
        if first_name:
            st.first_name = first_name
        return st

    def save(self, st: UserState) -> None:
        self._data[str(st.user_id)] = {"phone": st.phone, "client_key": st.client_key, "client_name": st.client_name}
        self.path.write_text(json.dumps(self._data, ensure_ascii=False, indent=1), "utf-8")


def _fn(name: str, desc: str, props: dict | None = None, required: list[str] | None = None) -> dict:
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": props or {}, "required": required or []}}}


DATE_PROPS = {
    "date_from": {"type": "string", "description": "Начало периода, YYYY-MM-DD"},
    "date_to": {"type": "string", "description": "Конец периода включительно, YYYY-MM-DD"},
}

CLIENT_TOOLS = [
    _fn("handoff_to_manager",
        "Передать диалог живому менеджеру (тур, покупка, жалоба, возврат, скидка, мед. вопрос, нет ответа в базе, "
        "просьба позвать человека).",
        {"reason": {"type": "string", "description": "Причина передачи"},
         "summary": {"type": "string", "description": "Кратко: кто клиент, что хочет, договорённости (дата/время тура)"}},
        ["reason", "summary"]),
]

STAFF_TOOLS = [
    _fn("staff_sales_stats", "Сколько абонементов продано за период, по видам и тарифам, новые/продления.",
        DATE_PROPS, ["date_from", "date_to"]),
    _fn("staff_new_clients", "Сколько новых клиентов зарегистрировано в 1С за период.", DATE_PROPS,
        ["date_from", "date_to"]),
    _fn("staff_totals", "Общее число клиентов и абонементов в 1С и дата актуальности данных."),
    _fn("staff_find_client", "Найти клиента в 1С по номеру телефона или части ФИО и показать его абонементы.",
        {"query": {"type": "string", "description": "Телефон (+992...) или фамилия/ФИО"}}, ["query"]),
]


class Agent:
    def __init__(self, onec: OneC, store: UserStore,
                 notify_manager: Callable[[UserState, str, str], Awaitable[None]] | None = None) -> None:
        self.onec = onec
        self.store = store
        self.notify_manager = notify_manager
        self.http = httpx.AsyncClient(base_url=settings.ollama_url, timeout=httpx.Timeout(300, connect=10))

    async def close(self) -> None:
        await self.http.aclose()

    # ------------------------------------------------------------------ verification
    async def verify_contact(self, st: UserState, phone: str) -> str:
        norm = normalize_phone(phone)
        st.phone = norm
        st.client_data = None
        clients = await self.onec.find_clients_by_phone(norm or "")
        if clients:
            c = clients[0]
            st.client_key = c["Ref_Key"]
            st.client_name = c.get("Имя") or c["Description"]
        else:
            st.client_key = st.client_name = None
        self.store.save(st)
        return "found" if clients else "not_found"

    # ------------------------------------------------------------------ tools
    async def _run_tool(self, st: UserState, name: str, args: dict, is_staff: bool) -> Any:
        if name.startswith("staff_") and not is_staff:
            return {"ошибка": "Нет доступа: это инструмент для сотрудников."}
        try:
            if name == "handoff_to_manager":
                if self.notify_manager:
                    await self.notify_manager(st, args.get("reason", ""), args.get("summary", ""))
                return {"ok": True, "сказать_клиенту": "Менеджер подключится и ответит в этом чате в рабочее время (9:00–21:00)."}
            if name == "staff_sales_stats":
                return await self.onec.sales_stats(date.fromisoformat(args["date_from"]), date.fromisoformat(args["date_to"]))
            if name == "staff_new_clients":
                return await self.onec.new_clients_count(date.fromisoformat(args["date_from"]),
                                                         date.fromisoformat(args["date_to"]))
            if name == "staff_totals":
                return await self.onec.totals()
            if name == "staff_find_client":
                q = args.get("query", "")
                found = await (self.onec.find_clients_by_phone(q) if normalize_phone(q) and sum(ch.isdigit() for ch in q) >= 9
                               else self.onec.search_clients_by_name(q))
                out = []
                for c in found[:5]:
                    out.append({"ФИО": c["Description"], "зарегистрирован": (c.get("ДатаРегистрации") or "")[:10],
                                "абонементы": (await self.onec.memberships(c["Ref_Key"]))[:5]})
                if not found:
                    return {"результат": f"по запросу «{q}» в 1С клиентов НЕ найдено"}
                for c in out:
                    if not c["абонементы"]:
                        c["абонементы"] = "абонементов в 1С НЕТ (ни одного)"
                return {"результат": f"НАЙДЕНО клиентов: {len(found)}", "клиенты": out}
        except Exception as e:  # noqa: BLE001
            log.exception("tool %s failed", name)
            return {"ошибка": f"1С сейчас недоступна или вернула ошибку ({type(e).__name__}). Скажи, что уточнишь позже."}
        return {"ошибка": f"неизвестный инструмент {name}"}

    # ------------------------------------------------------------------ chat
    async def _client_block(self, st: UserState) -> str:
        """1C data of the verified client, injected into the prompt (refreshed every 5 minutes)."""
        if not st.client_key:
            return ""
        if st.client_data is None or time.time() - st.client_data_at > 300:
            try:
                ms = await self.onec.memberships(st.client_key)
                data = {"имя_в_1С": st.client_name, "всего_абонементов": len(ms),
                        "абонементы_последние": ms[:5] if ms else "у клиента в 1С НЕТ ни одного абонемента"}
                st.client_data = json.dumps(_humanize(data), ensure_ascii=False, default=str)
                st.client_data_at = time.time()
            except Exception:  # noqa: BLE001
                log.exception("1C memberships fetch failed")
                return "ДАННЫЕ КЛИЕНТА ИЗ 1С: база временно недоступна — скажи, что уточнишь позже.\n"
        return f"ДАННЫЕ КЛИЕНТА ИЗ 1С (статус «действует» = карта активна сейчас):\n{st.client_data}\n"

    def _system(self, st: UserState, is_staff: bool, client_data: str) -> str:
        now = datetime.now(ZoneInfo(settings.timezone))
        today = f"{WEEKDAYS[now.weekday()]}, {now.day} {MONTHS[now.month - 1]} {now.year}, {now:%H:%M} ({now.date().isoformat()})"
        if is_staff:
            client = (f"сотрудник клуба ({st.first_name or 'без имени'}). Телефоны, которые он пишет, — это номера "
                      "клиентов для поиска, а не его собственный номер.")
            client_data = ""
        elif st.client_key:
            client = f"номер подтверждён, найден в 1С: {st.client_name}."
        elif st.phone:
            client = "номер подтверждён, но в 1С клиента с таким номером нет (новый клиент, данных о карте нет)."
        else:
            client = f"номер НЕ подтверждён, данных о карте нет. Имя в Telegram: {st.first_name or 'неизвестно'}."
        return SYSTEM_PROMPT.format(today=today, client=client, client_data=client_data,
                                    language=LANG_NAMES.get(st.language, "русский"),
                                    staff=STAFF_NOTE if is_staff else "", knowledge=KNOWLEDGE)

    async def _chat(self, messages: list[dict], tools: list[dict]) -> dict:
        r = await self.http.post("/api/chat", json={
            "model": settings.llm_model, "messages": messages, "tools": tools, "stream": False,
            "keep_alive": "60m",
            "options": {"temperature": settings.llm_temperature, "num_ctx": 8192, "top_p": 0.9},
        })
        r.raise_for_status()
        return r.json()["message"]

    async def spoken_version(self, text: str) -> str:
        """Rewrite a chat reply so it sounds natural when read aloud by TTS."""
        prompt = (
            "Перепиши сообщение так, как его произнёс бы вслух живой администратор фитнес-клуба в голосовом "
            "сообщении. Правила: тот же язык и тот же смысл, ничего не добавляй по фактам; все числа, даты, время, "
            "проценты и телефоны напиши словами в правильном падеже (например «до седьмого октября», «десять процентов»); "
            "время уже записано словами — НЕ меняй его значение, можно только поправить падеж; ссылки не читай — скажи «ссылку я прислала в сообщении»; "
            "убери эмодзи и спецсимволы; можно добавить лёгкие разговорные связки («так», «смотрите»), но коротко. "
            "Верни только текст для озвучки.\n\nСообщение:\n" + times_to_words(text)
        )
        try:
            r = await self.http.post("/api/chat", json={
                "model": settings.llm_model, "stream": False, "keep_alive": "60m",
                "messages": [{"role": "user", "content": prompt}],
                "options": {"temperature": 0.3, "num_ctx": 4096},
            })
            r.raise_for_status()
            out = (r.json()["message"].get("content") or "").strip().strip('"«»')
            return out or text
        except Exception:  # noqa: BLE001
            log.exception("spoken rewrite failed, using raw text")
            return text

    async def reply(self, st: UserState, text: str, is_staff: bool = False) -> str:
        tools = CLIENT_TOOLS + (STAFF_TOOLS if is_staff else [])
        if not text.startswith("[Системное"):
            lang = user_language(text)
            # languages the local model writes badly (e.g. Tajik for qwen2.5:7b) are answered in Russian
            st.language = lang if lang in settings.reply_languages else "ru"
            st.greet_salom = lang in ("tg", "uz")
        st.history.append({"role": "user", "content": text})
        st.history[:] = st.history[-MAX_HISTORY:]
        client_data = await self._client_block(st)
        messages = [{"role": "system", "content": self._system(st, is_staff, client_data)}, *st.history]
        notes = []
        phone = find_phone(text)
        if phone:  # deterministic 1C lookup: small models skip tool calls too often
            if is_staff:
                found = _humanize(await self._run_tool(st, "staff_find_client", {"query": phone}, True))
                log.info("staff phone lookup +%s -> %s", phone, str(found)[:300])
                notes.append(f"Поиск в 1С по номеру +{phone} (режим сотрудника): "
                             f"{json.dumps(found, ensure_ascii=False, default=str)}")
            else:
                try:
                    res = await self.verify_contact(st, phone)
                except Exception:  # noqa: BLE001
                    log.exception("1C phone lookup failed")
                    res = "error"
                if res == "found":
                    client_data = await self._client_block(st)
                    messages[0] = {"role": "system", "content": self._system(st, is_staff, client_data)}
                    notes.append(f"Поиск в 1С по номеру +{phone}: клиент НАЙДЕН ({st.client_name}). "
                                 "Его карты — в блоке ДАННЫЕ КЛИЕНТА ИЗ 1С. Не называй полное ФИО, только имя.")
                elif res == "not_found":
                    notes.append(f"Поиск в 1С по номеру +{phone}: клиента с таким номером НЕТ. "
                                 "Предложи проверить номер или нажать кнопку «📱 Поделиться номером»; "
                                 "если клиент уверен — передай менеджеру.")
                else:
                    notes.append("Поиск в 1С: база сейчас недоступна, скажи, что проверишь позже.")
        facts = relevant_facts(text)
        if facts:  # small models follow facts placed next to the question much better than a long system prompt
            notes.append(f"Справка из базы знаний — используй дословно, не меняй цифры:\n{facts}")
        if st.language != "ru":
            notes.append(f"Ответь на языке: {LANG_NAMES[st.language]}. Переведи справку на этот язык.")
        elif st.greet_salom:
            notes.append("Клиент пишет по-таджикски/узбекски. Ответь по-русски, простыми словами; "
                         "первое приветствие в диалоге начни с «Салом!».")
        if notes:
            messages[-1] = {"role": "user", "content": text + "\n\n[" + "\n".join(notes) + "]"}
        handed_off = False
        for _ in range(5):
            msg = await self._chat(messages, tools)
            calls = msg.get("tool_calls") or []
            if not calls:
                answer = (msg.get("content") or "").strip()
                break
            messages.append({"role": "assistant", "content": msg.get("content", ""), "tool_calls": calls})
            for call in calls:
                fn = call.get("function", {})
                args = fn.get("arguments") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                handed_off |= fn.get("name") == "handoff_to_manager"
                result = _humanize(await self._run_tool(st, fn.get("name", ""), args, is_staff))
                log.info("tool %s(%s) -> %s", fn.get("name"), args, str(result)[:300])
                messages.append({"role": "tool", "tool_name": fn.get("name", ""),
                                 "content": json.dumps(result, ensure_ascii=False, default=str)})
        else:
            answer = "Секунду, уточню у менеджера и вернусь с ответом."
        if not answer:
            answer = "Извините, не совсем поняла. Можете переформулировать?"
        if st.language != "zh" and re.search(r"[\u4e00-\u9fff]", answer):
            log.warning("model drifted into Chinese, retrying")
            messages.append({"role": "assistant", "content": answer})
            messages.append({"role": "user", "content": f"[Ты перешла на китайский. Перепиши ответ полностью на языке: "
                                                        f"{LANG_NAMES.get(st.language, 'русский')}, без китайских символов.]"})
            answer = ((await self._chat(messages, []))).get("content", "").strip() or answer
            answer = re.sub(r"[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]+", "", answer).strip()
        answer = tidy_answer(answer, st)
        st.history.append({"role": "assistant", "content": answer})
        if not handed_off and self.notify_manager and HANDOFF_RE.search(answer):
            # the model said it passed the dialog on but didn't call the tool: do it anyway
            convo = "\n".join(f"{'Клиент' if m['role'] == 'user' else 'Алина'}: {m['content']}"
                              for m in st.history[-6:] if not m["content"].startswith("[Системное"))
            await self.notify_manager(st, "ассистент передал диалог", convo)
        return answer
