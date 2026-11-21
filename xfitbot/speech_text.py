"""Turn a chat reply into text a TTS voice can read correctly.

Numbers are the main problem for TTS: "+992111553535" must not become "nine hundred ninety-two billion…".
Everything here is deterministic (no LLM), so values can't change on the way to the voice.
"""
from __future__ import annotations

import re

from num2words import num2words

MONTHS_RU = ["января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа", "сентября", "октября",
             "ноября", "декабря"]
MONTH_RE = "|".join(MONTHS_RU)

# Latin words that a Cyrillic voice would otherwise skip or mangle
BRANDS_RU = [
    (r"X[\s-]?Fit", "Икс Фит"), (r"X[\s-]?DRIVE", "Икс Драйв"), (r"Family\s+Solo\s+Premium", "Фэмили Соло Премиум"),
    (r"Family\s+Premium", "Фэмили Премиум"), (r"Family", "Фэмили"), (r"Premium", "Премиум"), (r"Solo", "Соло"),
    (r"Dushanbe", "Душанбе"), (r"VIP", "вип"), (r"Telegram", "Телеграм"), (r"Instagram", "Инстаграм"),
    (r"Facebook", "Фейсбук"), (r"WhatsApp", "Ватсап"), (r"Hyrox", "Хайрокс"), (r"CrossFit", "Кроссфит"),
    (r"\bSPA\b", "спа"), (r"\bPT\b", "пэ тэ"), (r"\bOK\b", "окей"),
]
URL_RE = re.compile(r"(https?://\S+|t\.me/\S+|www\.\S+|\b[\w.-]+\.(?:tj|com|ru|org)\b\S*)", re.I)
EMOJI_RE = re.compile("[\U0001F000-\U0001FAFF☀-➿️‍]+")

UZ_ONES = ["nol", "bir", "ikki", "uch", "to'rt", "besh", "olti", "yetti", "sakkiz", "to'qqiz"]
UZ_TENS = ["", "o'n", "yigirma", "o'ttiz", "qirq", "ellik", "oltmish", "yetmish", "sakson", "to'qson"]


def _uz_number(n: int) -> str:
    if n < 10:
        return UZ_ONES[n]
    parts = []
    for value, word in ((10 ** 9, "milliard"), (10 ** 6, "million"), (1000, "ming"), (100, "yuz")):
        if n >= value:
            q, n = divmod(n, value)
            parts.append(("" if q == 1 and value in (100, 1000) else _uz_number(q) + " ") + word)
    if n >= 10:
        parts.append(UZ_TENS[n // 10])
        n %= 10
    if n:
        parts.append(UZ_ONES[n])
    return " ".join(p.strip() for p in parts)


def cardinal(n: int, lang: str) -> str:
    if lang == "uz":
        return _uz_number(n)
    try:
        return num2words(n, lang="tg" if lang == "tg" else "ru" if lang == "ru" else "en")
    except Exception:  # noqa: BLE001
        return num2words(n, lang="ru")


def _ordinal_case(n: int, case: str) -> str:
    """Russian ordinal (masc/neuter) in genitive ('gen': седьмого) or prepositional ('prep': седьмом)."""
    words = num2words(n, lang="ru", to="ordinal").split()
    last = words[-1]
    if last.endswith("ий") and last not in ("третий",):
        last = last[:-2] + ("его" if case == "gen" else "ем")
    elif last == "третий":
        last = "третьего" if case == "gen" else "третьем"
    elif last.endswith(("ой", "ый")):
        last = last[:-2] + ("ого" if case == "gen" else "ом")
    return " ".join(words[:-1] + [last])


def _plural_ru(n: int, one: str, few: str, many: str) -> str:
    if n % 10 == 1 and n % 100 != 11:
        return one
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return few
    return many


def _digits_grouped(digits: str, lang: str, original: str = "") -> str:
    """Read an ID/phone as small groups: 992 11 155 35 35 -> 'девятьсот девяносто два, одиннадцать, …'.
    Keeps the writer's own grouping (+992 908 82 7777) when there is one."""
    own = [g for g in re.split(r"[\s\-()+]+", original) if g]
    if len(own) > 1 and all(g.isdigit() for g in own) and max(len(g) for g in own) <= 4:
        groups = []
        for g in own:
            groups += [g[:2], g[2:]] if len(g) == 4 else [g]
    elif len(digits) == 12 and digits.startswith("992"):
        groups = [digits[:3], digits[3:5], digits[5:8], digits[8:10], digits[10:]]
    elif len(digits) == 9:
        groups = [digits[:2], digits[2:5], digits[5:7], digits[7:]]
    else:
        groups = [digits[i:i + 2] for i in range(0, len(digits), 2)]
    out = []
    for g in groups:
        if g.startswith("0"):  # "05" -> "ноль пять"
            out.append(" ".join(cardinal(int(d), lang) for d in g))
        else:
            out.append(cardinal(int(g), lang))
    return ", ".join(out)


HOURS_GEN = ["", "часа", "двух", "трёх", "четырёх", "пяти", "шести", "семи", "восьми", "девяти", "десяти",
             "одиннадцати", "двенадцати"]
# dative, after "к": "к шести вечера", "к часу дня"
HOURS_DAT = ["", "часу", "двум", "трём", "четырём"] + HOURS_GEN[5:]


def _time_words(h: int, mi: int, lang: str, genitive: bool | str = False) -> str:
    if lang != "ru":
        return cardinal(h, lang) + (" " + cardinal(mi, lang) if mi else "")
    part = "ночи" if h < 5 else "утра" if h < 12 else "дня" if h < 17 else "вечера"
    h12 = h % 12 or 12
    if genitive and not mi:  # "с девяти утра до девяти вечера", "к шести вечера"
        return f"{(HOURS_DAT if genitive == 'dat' else HOURS_GEN)[h12]} {part}"
    hw = "час" if h12 == 1 else cardinal(h12, "ru")
    return f"{hw}{' ' + cardinal(mi, 'ru') if mi else ''} {part}"


def speakable(text: str, lang: str = "ru") -> str:
    """Make every number, link and Latin brand name in `text` readable by the voice."""
    text = URL_RE.sub("", text)
    text = EMOJI_RE.sub("", text)
    text = re.sub(r"[*_`#>|~«»\"]", "", text)
    if lang == "ru":
        for pat, rep in BRANDS_RU:
            text = re.sub(pat, rep, text, flags=re.I)

    # ISO dates 2026-09-08 -> "8 сентября 2026 года" (spelled out below)
    if lang == "ru":
        text = re.sub(r"\b(\d{4})-(\d{2})-(\d{2})\b",
                      lambda m: f"{int(m.group(3))} {MONTHS_RU[int(m.group(2)) - 1]} {m.group(1)} года"
                      if 1 <= int(m.group(2)) <= 12 else m.group(), text)
        # ranges "9:00–21:00" / "(9:00-21:00)" -> "с девяти утра до девяти вечера"
        text = re.sub(r"\(?\b(?:с\s+)?([01]?\d|2[0-3]):([0-5]\d)\s*(?:[–—-]|до)\s*([01]?\d|2[0-3]):([0-5]\d)\)?",
                      lambda m: f"с {_time_words(int(m.group(1)), int(m.group(2)), 'ru', True)} "
                                f"до {_time_words(int(m.group(3)), int(m.group(4)), 'ru', True)}", text)
        # time after "с/до/от/после/около" takes the genitive
        text = re.sub(r"\b(с|до|от|после|около)\s+([01]?\d|2[0-3]):([0-5]\d)\b",
                      lambda m: f"{m.group(1)} {_time_words(int(m.group(2)), int(m.group(3)), 'ru', True)}", text)
        # after "к" the dative: "к 18:00" -> "к шести вечера"
        text = re.sub(r"\b([кК])\s+([01]?\d|2[0-3]):([0-5]\d)\b",
                      lambda m: f"{m.group(1)} {_time_words(int(m.group(2)), int(m.group(3)), 'ru', 'dat')}", text)

    # phones and long IDs: +992 90 882 77 77, 992111553535, 111 55 35 35
    def phone(m: re.Match) -> str:
        digits = re.sub(r"\D", "", m.group())
        plus = ("плюс " if lang == "ru" else "плюс " if lang == "tg" else "plyus ") if m.group().lstrip().startswith("+") else ""
        return " " + plus + _digits_grouped(digits, lang, m.group()) + " "

    text = re.sub(r"\+?\d[\d\s\-()]{6,18}\d", lambda m: phone(m) if len(re.sub(r"\D", "", m.group())) >= 7 else m.group(), text)

    # times 19:00 / 9:30
    text = re.sub(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", lambda m: _time_words(int(m.group(1)), int(m.group(2)), lang), text)

    if lang == "ru":
        # dates: "7 октября 2026 года" / "7 октября"
        def date(m: re.Match) -> str:
            s = f"{_ordinal_case(int(m.group(1)), 'gen')} {m.group(2)}"
            if m.group(3):
                s += f" {_ordinal_case(int(m.group(3)), 'gen')} года"
            return s

        text = re.sub(rf"\b(\d{{1,2}})\s+({MONTH_RE})(?:\s+(\d{{4}})(?:\s*(?:года|г\.?))?)?", date, text)
        # years: "в 2026 году", "2026 года", "2026 год"
        text = re.sub(r"\b(\d{4})\s+(году)\b", lambda m: f"{_ordinal_case(int(m.group(1)), 'prep')} году", text)
        text = re.sub(r"\b(\d{4})\s+(года)\b", lambda m: f"{_ordinal_case(int(m.group(1)), 'gen')} года", text)
        text = re.sub(r"\b(\d{4})\s+(год)\b", lambda m: f"{num2words(int(m.group(1)), lang='ru', to='ordinal')} год", text)
        # percentages
        text = re.sub(r"(\d+)\s*%", lambda m: f"{cardinal(int(m.group(1)), 'ru')} "
                                               f"{_plural_ru(int(m.group(1)), 'процент', 'процента', 'процентов')}", text)
        text = re.sub(r"(\d+)\s*(?:сомони|сом\.?)\b", lambda m: f"{cardinal(int(m.group(1)), 'ru')} сомони", text)
    else:
        text = re.sub(r"(\d+)\s*%", lambda m: f"{cardinal(int(m.group(1)), lang)} "
                                               f"{'фоиз' if lang == 'tg' else 'foiz' if lang == 'uz' else 'percent'}", text)

    # anything left: short numbers as words, long ones in groups
    def rest(m: re.Match) -> str:
        d = m.group()
        return _digits_grouped(d, lang) if len(d) >= 7 else cardinal(int(d), lang)

    text = re.sub(r"\d+", rest, text)
    text = re.sub(r"\bодна тысяча\b", "тысяча", text)
    text = text.replace("—", ",").replace("–", ",").replace("+", " плюс ")
    text = re.sub(r"\s+([,.!?…])", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


EMOTIONS = ("радость", "воодушевление", "тепло", "сочувствие", "вопрос", "спокойно")
EMOTION_LINE_RE = re.compile(r"^\s*[-*\d.)]*\s*\[(\w+)\]\s*(.+?)\s*$")
SAD_RE = re.compile(r"(к сожалению|увы|извин|не предусмотр|истекл|нельзя|не получится|понимаю|afsus|мутаассифона)", re.I)
HAPPY_RE = re.compile(r"(отлично|здорово|рада|супер|прекрасно|ура|здравствуйте|привет|салом|добро пожаловать|great)", re.I)


def split_sentences(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"(?<=[.!?…])\s+", text) if p.strip()]


def guess_emotion(sentence: str) -> str:
    if sentence.rstrip().endswith("?"):
        return "вопрос"
    if SAD_RE.search(sentence):
        return "сочувствие"
    if sentence.rstrip().endswith("!") or HAPPY_RE.search(sentence):
        return "радость"
    return "тепло"


def heuristic_emotions(text: str) -> list[tuple[str, str]]:
    return [(guess_emotion(s), s) for s in split_sentences(text)]


def parse_emotion_lines(out: str) -> list[tuple[str, str]]:
    """Parse '[радость] Ой, как здорово!' lines from the LLM; untagged lines get a guessed emotion."""
    segs: list[tuple[str, str]] = []
    for line in out.splitlines():
        line = line.strip().strip('"«»')
        if not line:
            continue
        m = EMOTION_LINE_RE.match(line)
        if m:
            emo, sent = m.group(1).lower(), m.group(2).strip()
            if emo not in EMOTIONS:
                emo = guess_emotion(sent)
        else:
            emo, sent = guess_emotion(line), line
        if sent.endswith("?"):
            emo = "вопрос"
        segs.append((emo, sent))
    return segs
