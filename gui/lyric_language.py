"""Infer the singing language from pasted reference lyrics."""
from __future__ import annotations

import re

_KANA_RE = re.compile(r"[\u3040-\u309f\u30a0-\u30ff]")
_HAN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_LATIN_RE = re.compile(r"[A-Za-z]")

# The same LRC shapes stripped before matching: [ti:…] values and credit
# lines ("作詞：…") are not sung and must not decide the language.
_LRC_TIMESTAMP_RE = re.compile(r"\[\d{1,3}:\d{1,2}(?:[.:]\d{1,3})?\]")
_LRC_METADATA_RE = re.compile(
    r"^\s*\[(?:ti|ar|al|by|offset|re|ve|length|au|tool|encoding):([^\]]*)\]\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_LRC_CREDIT_RE = re.compile(
    r"^\s*(?:作词|作曲|编曲|词曲|词|曲|歌|演唱|原唱|翻唱|和声|混音|母带|制作人|制作|策划|出品|录音|配唱|监制"
    r"|作詞|編曲|歌手|唄|artist|title)(?:\s*[:：]|\s+by\b).*$",
    re.IGNORECASE | re.MULTILINE,
)


def _lyric_body(text: str) -> str:
    if _LRC_TIMESTAMP_RE.search(text):
        text = _LRC_METADATA_RE.sub(" ", text)
        text = _LRC_TIMESTAMP_RE.sub(" ", text)
        text = _LRC_CREDIT_RE.sub(" ", text)
    return text


def detect_lyric_language(text: str | None) -> str | None:
    """Return "ja", "zh" or "en" for the lyrics text, else None.

    Kana are unique to Japanese, while Han-only text is treated as Chinese
    (Cantonese lyrics are written the same way and need an explicit choice).
    Latin-only text is English; romaji Japanese is indistinguishable from it
    without a lexicon, so it also reports "en".
    """
    body = _lyric_body(text or "")
    if _KANA_RE.search(body):
        return "ja"
    if _HAN_RE.search(body):
        return "zh"
    if len(_LATIN_RE.findall(body)) >= 3:
        return "en"
    return None
