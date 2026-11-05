from __future__ import annotations

from gui.lyric_language import detect_lyric_language


def test_kana_lyrics_are_japanese():
    assert detect_lyric_language("夢を見ていたいわ\nいつまでも終わらず") == "ja"
    assert detect_lyric_language("きみのこえ") == "ja"


def test_kanji_lyrics_without_kana_are_chinese():
    # Kanji-only Japanese (e.g. a title) is indistinguishable from Chinese;
    # Cantonese is written the same way, so both report "zh".
    assert detect_lyric_language("月亮代表我的心") == "zh"
    assert detect_lyric_language("梦想 白百合") == "zh"


def test_latin_lyrics_are_english():
    assert detect_lyric_language("I dreamed of you again") == "en"
    # Romaji is indistinguishable from English without a lexicon.
    assert detect_lyric_language("kimi no koe") == "en"


def test_short_or_symbolic_text_is_not_classified():
    assert detect_lyric_language("") is None
    assert detect_lyric_language("   ") is None
    assert detect_lyric_language("hi") is None  # fewer than 3 latin letters
    assert detect_lyric_language("...!?") is None


def test_lrc_metadata_does_not_decide_the_language():
    lrc = "\n".join(
        [
            "[ti:Dreaming of You]",
            "[ar:Unknown Artist]",
            "[00:07.700]夢を見ていたいわ",
            "[00:11.200]いつまでも終わらず",
        ]
    )
    assert detect_lyric_language(lrc) == "ja"


def test_lrc_credits_are_ignored():
    lrc = "\n".join(
        [
            "[00:07.700]作詞：KAB.",
            "[00:12.000]夢を見ていたいわ",
        ]
    )
    assert detect_lyric_language(lrc) == "ja"
    plain = "作詞：KAB. 月亮"
    assert detect_lyric_language(plain) == "zh"
