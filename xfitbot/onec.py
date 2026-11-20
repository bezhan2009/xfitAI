"""Read-only client for 1C:Fitness (standard OData interface).

Only HTTP GET is ever issued: the bot never creates or changes anything in 1C.
"""
from __future__ import annotations

import base64
import calendar
import json
import logging
import re
import asyncio
import time
from datetime import date, datetime, timedelta
from typing import Any
from urllib.parse import quote

import httpx

from .config import settings

log = logging.getLogger(__name__)

EMPTY_GUID = "00000000-0000-0000-0000-000000000000"
MEMBERSHIPS = "Document_ЧленствоПакетУслуг"
CLIENTS = "Catalog_Контрагенты"

TYPE_NAMES = {
    "АбонементЧленство": "клубная карта",
    "АбонементПакетУслуг": "пакет услуг",
}
UNIT_NAMES = {
    "День": ("день", "дня", "дней"),
    "Неделя": ("неделя", "недели", "недель"),
    "Месяц": ("месяц", "месяца", "месяцев"),
    "Год": ("год", "года", "лет"),
}


def plural(n: int, forms: tuple[str, str, str]) -> str:
    n = abs(n)
    if n % 10 == 1 and n % 100 != 11:
        return forms[0]
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return forms[1]
    return forms[2]


def normalize_phone(raw: str) -> str | None:
    """Return 992XXXXXXXXX (Tajikistan) or the full international digits, else None."""
    digits = re.sub(r"\D", "", raw or "")
    if len(digits) == 9:  # local Tajik number without the country code
        digits = "992" + digits
    if digits.startswith("8") and len(digits) == 10:
        digits = "992" + digits[1:]
    return digits if 10 <= len(digits) <= 15 else None


def _parse_dt(v: str | None) -> datetime | None:
    if not v or v.startswith("0001-01-01"):
        return None
    try:
        return datetime.fromisoformat(v)
    except ValueError:
        return None


def _add_period(start: date, n: int, unit: str) -> date:
    if unit == "День":
        return start + timedelta(days=n - 1 if n > 0 else 0)
    if unit == "Неделя":
        return start + timedelta(weeks=n) - timedelta(days=1)
    months = n * 12 if unit == "Год" else n
    if unit not in ("Месяц", "Год"):
        return start
    y, m = divmod(start.month - 1 + months, 12)
    y, m = start.year + y, m + 1
    d = min(start.day, calendar.monthrange(y, m)[1])
    return date(y, m, d) - timedelta(days=1)


class OneC:
    def __init__(self) -> None:
        token = base64.b64encode(f"{settings.onec_user}:{settings.onec_password}".encode("utf-8")).decode()
        self._client = httpx.AsyncClient(
            base_url=settings.onec_base_url + "/odata/standard.odata/",
            headers={"Authorization": f"Basic {token}", "Accept": "application/json"},
            timeout=httpx.Timeout(60, connect=15),
        )
        self._plan_names: dict[str, str] | None = None
        self._plan_names_loaded_at = 0.0

    async def close(self) -> None:
        await self._client.aclose()

    # ------------------------------------------------------------------ low level
    async def _get(self, entity: str, **params: str) -> Any:
        safe = "',()=:/"
        query = "&".join(f"${k}={quote(str(v), safe=safe)}" for k, v in params.items())
        url = quote(entity, safe="()'/,=$") + ("?" + query if query else "")
        for attempt in range(3):
            try:
                r = await self._client.get(url)
                if r.status_code == 404:
                    return None
                r.raise_for_status()
                return r.json() if "json" in r.headers.get("content-type", "") else r.text
            except httpx.TransportError as e:
                log.warning("1C request failed (%s): %s", attempt + 1, e)
                if attempt == 2:
                    raise
                await asyncio.sleep(1.5 * (attempt + 1))

    async def _list(self, entity: str, **params: str) -> list[dict]:
        params.setdefault("format", "json")
        data = await self._get(entity, **params)
        return (data or {}).get("value", []) if isinstance(data, dict) else []

    async def _count(self, entity: str, flt: str | None = None) -> int:
        params = {"filter": flt} if flt else {}
        data = await self._get(f"{entity}/$count", **params)
        return int(str(data).strip() or 0) if data is not None else 0

    async def ping(self) -> bool:
        try:
            await self._get("", format="json")
            return True
        except Exception:  # noqa: BLE001
            return False

    # ------------------------------------------------------------------ plans
    async def plan_names(self) -> dict[str, str]:
        """GUID -> human name for products (tariffs).

        Uses Catalog_Номенклатура when 1C publishes it via OData; otherwise data/plans.json,
        which staff can fill in by hand. Refreshed every 6 hours.
        """
        if self._plan_names is not None and time.time() - self._plan_names_loaded_at < 6 * 3600:
            return self._plan_names
        names: dict[str, str] = {}
        try:
            for row in await self._list("Catalog_Номенклатура", select="Ref_Key,Description"):
                names[row["Ref_Key"]] = row["Description"]
        except Exception:  # noqa: BLE001 - catalog not published yet
            pass
        manual = settings.data_dir / "plans.json"
        if manual.exists():
            names.update({k: v for k, v in json.loads(manual.read_text("utf-8")).items() if v})
        self._plan_names, self._plan_names_loaded_at = names, time.time()
        return names

    # ------------------------------------------------------------------ clients
    async def find_clients_by_phone(self, phone: str) -> list[dict]:
        norm = normalize_phone(phone)
        if not norm:
            return []
        sel = "Ref_Key,Description,Имя,Фамилия,ПолКлиента,ДатаРегистрации,ДатаРожденияКлиента,ТипКонтрагента,DeletionMark"
        rows = await self._list(
            CLIENTS, select=sel, top="10",
            filter=f"КонтактнаяИнформация/any(d: d/НомерТелефона eq '{norm}')",
        )
        # 1C stores numbers inconsistently ("992XXXXXXXXX", "XXXXXXXXX", "+992 XX…"), so fall back step by step
        for flt in (f"КонтактнаяИнформация/any(d: d/НомерТелефонаБезКодов eq '{norm[3:]}')",
                    f"КонтактнаяИнформация/any(d: d/НомерТелефонаБезКодов eq '{norm}')",
                    f"substringof('{norm[-9:]}', Телефон) eq true"):
            if rows:
                break
            rows = await self._list(CLIENTS, select=sel, top="10", filter=flt)
        return [r for r in rows if not r.get("DeletionMark")]

    async def search_clients_by_name(self, text: str, limit: int = 10) -> list[dict]:
        safe = text.replace("'", "''").strip()
        if len(safe) < 3:
            return []
        return await self._list(
            CLIENTS, select="Ref_Key,Description,ДатаРегистрации,ТипКонтрагента", top=str(limit),
            filter=f"substringof('{safe}', Description) eq true and DeletionMark eq false",
        )

    async def client_phones(self, client_key: str) -> list[str]:
        row = await self._get(f"{CLIENTS}(guid'{client_key}')", format="json", select="КонтактнаяИнформация")
        return [c.get("Представление") for c in (row or {}).get("КонтактнаяИнформация", []) if c.get("Тип") == "Телефон"]

    # ------------------------------------------------------------------ memberships
    async def memberships(self, client_key: str) -> list[dict]:
        own = await self._list(
            MEMBERSHIPS,
            filter=f"Контрагент eq cast(guid'{client_key}', '{CLIENTS}') and DeletionMark eq false and Posted eq true",
        )
        # NB: this 1C build rejects any() filters on ДополнительныеВладельцы (HTTP 500),
        # so family co-owner cards are not looked up here.
        extra: list[dict] = []
        own_keys = {r["Ref_Key"] for r in own}
        seen, out = set(), []
        names = await self.plan_names()
        for row in own + extra:
            if row["Ref_Key"] in seen:
                continue
            seen.add(row["Ref_Key"])
            m = self._describe(row, names)
            m["вы_дополнительный_владелец"] = row["Ref_Key"] not in own_keys
            out.append(m)
        out.sort(key=lambda m: m["_sort"], reverse=True)
        for m in out:
            m.pop("_sort", None)
        return out

    def _describe(self, row: dict, names: dict[str, str]) -> dict:
        today = date.today()
        n = int(row.get("СрокДействия") or 0)
        unit = row.get("СрокДействияТип") or ""
        period = f"{n} {plural(n, UNIT_NAMES[unit])}" if unit in UNIT_NAMES and n else None

        start_dt = _parse_dt(row.get("ДатаРучнойАктивации")) or _parse_dt(row.get("СоДоступДатаПервойАктивации"))
        end_dt = _parse_dt(row.get("СрокДействияДатой")) or _parse_dt(row.get("СоДоступДатаОкончанияСрокаДействия"))
        start = start_dt.date() if start_dt else None
        end = end_dt.date() if end_dt else (_add_period(start, n, unit) if start and n else None)

        if row.get("Архивный"):
            status = "архивная (закрыта)"
        elif not start:
            status = "ещё не активирована"
        elif end and end < today:
            status = "истекла"
        elif start > today:
            status = "активируется позже"
        else:
            status = "действует"

        kind = TYPE_NAMES.get(row.get("ТипЧленстваПакетаУслуг"), row.get("ТипЧленстваПакетаУслуг"))
        plan = names.get(row.get("Номенклатура_Key", ""))
        info: dict[str, Any] = {
            "номер_документа": row.get("Number"),
            "дата_продажи": (_parse_dt(row.get("Date")) or datetime.min).date().isoformat(),
            "тариф": plan or f"{kind}{' на ' + period if period else ''}",
            "вид": kind,
            "срок": period,
            "статус": status,
            "дата_активации": start.isoformat() if start else None,
            "действует_до": end.isoformat() if end else None,
            "осталось_дней": (end - today).days + 1 if end and status == "действует" else None,
            "дней_заморозки_по_тарифу": row.get("КоличествоДнейЗаморозок") or 0,
            "гостевых_посещений": row.get("КоличествоГостевыхПосещений") or 0,
            "с_полотенцем": bool(row.get("XFIT_СПолотенцем")),
            "_sort": (row.get("Date") or ""),
        }
        visits = int(row.get("КоличествоПосещений") or 0)
        if visits:
            info["посещений_по_тарифу"] = visits
        if row.get("ВремяДействия"):
            info["время_действия"] = [
                f"{r.get('ДеньНедели')}: {r.get('ВремяС', '')[11:16]}–{r.get('ВремяПо', '')[11:16]}"
                for r in row["ВремяДействия"]
            ]
        if row.get("ПричинаОтказаОтПродления"):
            info["причина_отказа_от_продления"] = row["ПричинаОтказаОтПродления"]
        if not plan:
            info["_plan_key"] = row.get("Номенклатура_Key")
        return info

    # ------------------------------------------------------------------ staff analytics
    async def sales_stats(self, date_from: date, date_to: date) -> dict:
        """Memberships sold (posted documents) in [date_from, date_to]."""
        flt = (
            f"Date ge datetime'{date_from.isoformat()}T00:00:00' and Date le datetime'{date_to.isoformat()}T23:59:59' "
            f"and Posted eq true and DeletionMark eq false"
        )
        rows = await self._list(
            MEMBERSHIPS, filter=flt,
            select="Номенклатура_Key,ТипЧленстваПакетаУслуг,СрокДействия,СрокДействияТип,ТипПродления,ПробныйПакетУслуг",
        )
        names = await self.plan_names()
        by_plan: dict[str, int] = {}
        by_kind: dict[str, int] = {}
        by_renewal: dict[str, int] = {}
        for r in rows:
            n = int(r.get("СрокДействия") or 0)
            unit = r.get("СрокДействияТип") or ""
            kind = TYPE_NAMES.get(r["ТипЧленстваПакетаУслуг"], r["ТипЧленстваПакетаУслуг"])
            label = names.get(r["Номенклатура_Key"]) or (
                f"{kind} {n} {plural(n, UNIT_NAMES[unit])}" if unit in UNIT_NAMES else kind)
            by_plan[label] = by_plan.get(label, 0) + 1
            by_kind[kind] = by_kind.get(kind, 0) + 1
            ren = {"Новое": "новые", "Продление": "продления"}.get(r.get("ТипПродления"), r.get("ТипПродления") or "—")
            by_renewal[ren] = by_renewal.get(ren, 0) + 1
        top = dict(sorted(by_plan.items(), key=lambda kv: -kv[1])[:15])
        return {
            "период": f"{date_from.isoformat()} — {date_to.isoformat()}",
            "всего_продано": len(rows),
            "по_виду": by_kind,
            "новые_или_продления": by_renewal,
            "топ_тарифов": top,
            "примечание": None if names else "Названия тарифов не опубликованы в OData 1С, показаны вид и срок.",
        }

    async def new_clients_count(self, date_from: date, date_to: date) -> dict:
        flt = (
            f"ДатаРегистрации ge datetime'{date_from.isoformat()}T00:00:00' and "
            f"ДатаРегистрации le datetime'{date_to.isoformat()}T23:59:59' and DeletionMark eq false"
        )
        return {"период": f"{date_from.isoformat()} — {date_to.isoformat()}",
                "новых_клиентов": await self._count(CLIENTS, flt)}

    async def totals(self) -> dict:
        last = await self._list(MEMBERSHIPS, top="1", select="Date", orderby="Date desc")
        return {
            "данные_1С_актуальны_на": (last[0]["Date"][:10] if last else None),
            "клиентов_в_базе": await self._count(CLIENTS, "DeletionMark eq false"),
            "документов_членства_всего": await self._count(MEMBERSHIPS, "Posted eq true"),
        }
