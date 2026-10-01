"""What a file the owner sent actually is, and whether it is safe to open.

Until Oct 1, 2026 the hub had no handler for files at all: the owner sent his
university calendar (.ics), nothing answered, nothing was logged, and on
«че это за файл?» the bot could only ask «какой файл?».

The type is decided by the CONTENT, not by the name: a «.ics» that is really
a zip or an executable is refused. Nothing here ever executes or unpacks a
file; text formats are decoded as text and parsed as data. Executables and
scripts are refused outright, archives and office files are not unpacked,
and anything over MAX_BYTES is not downloaded in the first place.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Optional

# Telegram lets bots download up to 20 MB; a calendar or a document of
# interest is far below this, and the VM has 1 GB of memory.
MAX_BYTES = 5 * 1024 * 1024

# Never opened, whatever is inside: a script is text and would otherwise pass
# as «text».
_EXECUTABLE_EXT = {
    ".exe", ".msi", ".bat", ".cmd", ".com", ".scr", ".pif", ".ps1", ".psm1", ".vbs",
    ".vbe", ".js", ".jse", ".wsf", ".wsh", ".hta", ".jar", ".apk", ".app", ".dmg",
    ".sh", ".bash", ".py", ".pyw", ".lnk", ".reg", ".dll", ".so", ".deb", ".rpm",
}
_OFFICE_EXT = {".docx", ".xlsx", ".pptx", ".odt", ".ods", ".odp", ".doc", ".xls", ".ppt"}

_MAGIC = [
    (b"MZ", "executable"),
    (b"\x7fELF", "executable"),
    (b"\xca\xfe\xba\xbe", "executable"),
    (b"%PDF", "pdf"),
    (b"PK\x03\x04", "zip"),
    (b"Rar!", "archive"),
    (b"7z\xbc\xaf\x27\x1c", "archive"),
    (b"\x1f\x8b", "archive"),
    (b"\x89PNG", "image"),
    (b"\xff\xd8\xff", "image"),
    (b"GIF8", "image"),
    (b"\xd0\xcf\x11\xe0", "office"),  # old binary Office (.doc/.xls)
]

# Kinds with a reader behind them. The rest get an honest «not yet».
READABLE = {"calendar"}


@dataclass
class FileVerdict:
    kind: str     # calendar | text | pdf | office | image | archive | executable | unknown
    ok: bool      # there is a reader for it and it is safe to read
    reason: str   # for the owner, when not ok
    text: str = ""


def _ext(filename: str) -> str:
    return PurePosixPath(str(filename or "").lower()).suffix


def decode_text(raw: bytes) -> Optional[str]:
    """Text in any of the encodings calendars come in, or None for binary.

    CampusNet exports UTF-16 LE without a BOM (01.10.2026: «B\\x00E\\x00G\\x00…»),
    so a missing BOM is detected by where the zero bytes are.
    """
    if not raw:
        return ""
    if raw.startswith(b"\xef\xbb\xbf"):
        candidates = ["utf-8-sig"]
    elif raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        candidates = ["utf-16"]
    else:
        sample = raw[:4096]
        even_zeros = sample[0::2].count(0)
        odd_zeros = sample[1::2].count(0)
        half = max(1, len(sample) // 2)
        if odd_zeros > half * 0.3 and even_zeros < half * 0.05:
            candidates = ["utf-16-le"]
        elif even_zeros > half * 0.3 and odd_zeros < half * 0.05:
            candidates = ["utf-16-be"]
        else:
            candidates = ["utf-8", "cp1252"]
    for enc in candidates:
        try:
            text = raw.decode(enc)
        except UnicodeDecodeError:
            continue
        controls = sum(1 for ch in text[:4096]
                       if ord(ch) < 32 and ch not in "\r\n\t\x0c")
        if controls > max(8, len(text[:4096]) * 0.01):
            return None  # decodes, but it is binary
        return text
    return None


def inspect(raw: bytes, filename: str = "") -> FileVerdict:
    """Decide what the file is from its bytes. Never executes or unpacks it."""
    if len(raw) > MAX_BYTES:
        return FileVerdict("unknown", False,
                           f"файл больше {MAX_BYTES // (1024 * 1024)} МБ — не открываю")
    ext = _ext(filename)
    if ext in _EXECUTABLE_EXT:
        return FileVerdict("executable", False,
                           f"это исполняемый файл или скрипт ({ext}) — такие не открываю")
    for magic, kind in _MAGIC:
        if raw.startswith(magic):
            if kind == "zip":
                kind = "office" if ext in _OFFICE_EXT else "archive"
            return _not_readable(kind, ext)

    text = decode_text(raw)
    if text is None:
        return FileVerdict("unknown", False, "это двоичный файл неизвестного формата — не открываю")
    if re.match(r"\s*BEGIN:VCALENDAR\b", text, re.IGNORECASE):
        return FileVerdict("calendar", True, "", text)
    if ext in (".ics", ".ical", ".ifb"):
        return FileVerdict("unknown", False,
                           "файл назван календарём, но внутри не календарь — не открываю")
    return _not_readable("text", ext)


def _not_readable(kind: str, ext: str) -> FileVerdict:
    reasons = {
        "executable": "это исполняемый файл — такие не открываю",
        "archive": "это архив — архивы не распаковываю",
        "image": "картинку пришли как фото, не файлом — тогда разберу",
        "pdf": "PDF пока не читаю — умею только календари (.ics)",
        "office": "документы Office пока не читаю — умею только календари (.ics)",
        "text": "текстовые файлы пока не разбираю — умею только календари (.ics)",
    }
    label = f" ({ext})" if ext else ""
    return FileVerdict(kind, False, reasons.get(kind, "такой формат не читаю") + label)
