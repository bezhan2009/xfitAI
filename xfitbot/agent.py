"""LLM agent: local Qwen via Ollama with read-only 1C tools."""
from __future__ import annotations

import json
import logging
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

SYSTEM_PROMPT = """Ты — Алина, ассистент фитнес-клуба X-Fit Premium Dushanbe в Telegram. Ты общаешься с людьми \
как живой, тёплый и внимательный администратор клуба: просто, по-человечески, без канцелярита.

Как писать:
- Отвечай на языке собеседника (русский, таджикский, английский; узбекский и китайский — по возможности).
- Коротко: обычно 1–4 предложения. Без списков, заголовков, markdown и таблиц — это переписка в мессенджере.
- Сначала ответ по сути, потом максимум один-два уточняющих вопроса. Не веди анкету.
- Пиши естественно, как человек: «Смотрю… да, нашла вашу карту», «Понимаю», «Отличный вопрос». Можно одно \
уместное эмодзи, не больше. Не повторяй одно и то же приветствие. Обращайся по имени, если оно известно.
- Даты пиши словами месяца: «7 октября», а не 2026-10-07. Не показывай служебные поля, ключи, GUID и номера документов, \
если о них не спросили.
- Если разговор про покупку — заверши вопросом с двумя вариантами: «Вам удобнее сегодня вечером или завтра днём?»

Правила по данным:
- Отвечай ТОЛЬКО по базе знаний ниже и по данным из 1С, полученным через инструменты. Ничего не выдумывай: \
ни цен, ни акций, ни расписания, ни условий.
- Если ответа нет ни в базе знаний, ни в 1С — честно скажи, что уточнишь у менеджера, и вызови handoff_to_manager.
- Личные данные (абонемент, срок, статус) можно сообщать только когда номер телефона клиента подтверждён \
(поле «Клиент» ниже). Если номер не подтверждён — попроси нажать кнопку «📱 Поделиться номером». \
Номер, набранный текстом, для личных данных не подходит.
- 1С ты только читаешь. Записать на тур, заморозить, продлить, оформить — это делает менеджер: \
собери пожелание (дата/время) и передай менеджеру через handoff_to_manager.
- Данные 1С бывают с задержкой: если инструмент говорит, на какую дату актуальны данные, учитывай это.

Сегодня: {today}. Часовой пояс: Душанбе.
Клиент: {client}
{staff}
=== БАЗА ЗНАНИЙ ===
{knowledge}
"""

STAFF_NOTE = """Собеседник — сотрудник клуба (режим персонала). Ему можно отвечать на аналитические вопросы по 1С \
(продажи, новые клиенты, поиск клиента по телефону или ФИО) через инструменты staff_*. Отвечай точно цифрами, \
кратко, можно списком."""


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
    _fn("get_my_memberships",
        "Абонементы/клубные карты подтверждённого клиента из 1С: тариф, статус, дата активации, действует до, "
        "сколько дней осталось, дни заморозки, гостевые посещения."),
    _fn("get_my_profile", "Карточка подтверждённого клиента в 1С: имя, дата регистрации."),
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
        if name in ("get_my_memberships", "get_my_profile") and not st.client_key:
            return {"ошибка": "Номер телефона не подтверждён или клиент не найден в 1С. "
                              "Попроси поделиться номером кнопкой, не раскрывай никаких данных."}
        try:
            if name == "get_my_memberships":
                ms = await self.onec.memberships(st.client_key)
                return {"абонементы": ms[:8], "всего": len(ms)} if ms else {"абонементы": [], "примечание": "Абонементов в 1С нет."}
            if name == "get_my_profile":
                rows = await self.onec._list("Catalog_Контрагенты", filter=f"Ref_Key eq guid'{st.client_key}'",
                                             select="Description,Имя,ДатаРегистрации,ПолКлиента")
                return rows[0] if rows else {}
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
                return {"найдено": len(found), "клиенты": out}
        except Exception as e:  # noqa: BLE001
            log.exception("tool %s failed", name)
            return {"ошибка": f"1С сейчас недоступна или вернула ошибку ({type(e).__name__}). Скажи, что уточнишь позже."}
        return {"ошибка": f"неизвестный инструмент {name}"}

    # ------------------------------------------------------------------ chat
    def _system(self, st: UserState, is_staff: bool) -> str:
        now = datetime.now(ZoneInfo(settings.timezone))
        today = f"{WEEKDAYS[now.weekday()]}, {now.day} {MONTHS[now.month - 1]} {now.year}, {now:%H:%M} ({now.date().isoformat()})"
        if st.client_key:
            client = f"номер подтверждён, найден в 1С: {st.client_name}."
        elif st.phone:
            client = "номер подтверждён, но в 1С клиента с таким номером нет (новый клиент)."
        else:
            client = f"номер не подтверждён. Имя в Telegram: {st.first_name or 'неизвестно'}."
        return SYSTEM_PROMPT.format(today=today, client=client, staff=STAFF_NOTE if is_staff else "", knowledge=KNOWLEDGE)

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
            "проценты и телефоны напиши словами в правильном падеже (например «до седьмого октября», «с девяти утра "
            "до девяти вечера», «десять процентов»); ссылки не читай — скажи «ссылку я прислала в сообщении»; "
            "убери эмодзи и спецсимволы; можно добавить лёгкие разговорные связки («так», «смотрите»), но коротко. "
            "Верни только текст для озвучки.\n\nСообщение:\n" + text
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
        st.history.append({"role": "user", "content": text})
        st.history[:] = st.history[-MAX_HISTORY:]
        messages = [{"role": "system", "content": self._system(st, is_staff)}, *st.history]
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
                result = _humanize(await self._run_tool(st, fn.get("name", ""), args, is_staff))
                log.info("tool %s(%s) -> %s", fn.get("name"), args, str(result)[:300])
                messages.append({"role": "tool", "tool_name": fn.get("name", ""),
                                 "content": json.dumps(result, ensure_ascii=False, default=str)})
        else:
            answer = "Секунду, уточню у менеджера и вернусь с ответом."
        if not answer:
            answer = "Извините, не совсем поняла. Можете переформулировать?"
        st.history.append({"role": "assistant", "content": answer})
        return answer
