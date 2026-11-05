from inference.LyricFA.tools.lyric_matcher import strip_lrc_markup


def test_lrc_metadata_timestamps_and_credits_removed():
    raw = (
        "\ufeff[ti:月夜の君と空の上]\n"
        "[ar:風華]\n"
        "[al:いつか、届く、あの空に。]\n"
        "[by:]\n"
        "[offset:0]\n"
        "[00:00.000]月夜の君と空の上 - 風華\n"
        "[00:07.700]词：KAB.\n"
        "[00:15.410]曲：高井ウララ\n"
        "[00:30.839]長い髪を結わいて風を\n"
        "[00:33.767]感じてみるの\n"
    )

    assert strip_lrc_markup(raw) == "長い髪を結わいて風を\n感じてみるの"


def test_plain_lyrics_are_left_alone():
    text = "長い髪を\n感じてみるの"

    assert strip_lrc_markup(text) == text


def test_repeated_timestamps_on_one_line_are_removed():
    assert strip_lrc_markup("[00:12.34][01:02.50]同じ歌詞\n") == "同じ歌詞"


def test_credit_line_without_lrc_markers_is_kept():
    assert strip_lrc_markup("歌：誰か") == "歌：誰か"
