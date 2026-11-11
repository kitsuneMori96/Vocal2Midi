import yaml

from application.sentence_jobs import add_reference_take_with_handshake, splice_take_with_handshake
from inference.io.note_io import NoteInfo
from inference.API.ustx_api import save_ustx


def _write_ustx(path, notes):
    save_ustx(notes, path, 120.0)
    return path


def test_in_place_patch_keeps_backup_and_splices(tmp_path, monkeypatch):
    monkeypatch.setattr("application.openutau_bridge._running_openutau_dirs", lambda: [])
    base = _write_ustx(tmp_path / "song.ustx", [
        NoteInfo(1.0, 1.5, 60.0, "ど"),
        NoteInfo(2.2, 2.6, 62.0, "old"),
    ])
    take = _write_ustx(tmp_path / "take.ustx", [NoteInfo(2.1, 2.4, 65.0, "new")])

    removed, inserted = splice_take_with_handshake(base, take, base, (2.0, 4.0))
    assert (removed, inserted) == (1, 1)

    patched = yaml.safe_load(base.read_text(encoding="utf-8"))
    assert [n["lyric"] for n in patched["voice_parts"][0]["notes"]] == ["ど", "new"]
    backup = yaml.safe_load((tmp_path / "song_backup.ustx").read_text(encoding="utf-8"))
    assert [n["lyric"] for n in backup["voice_parts"][0]["notes"]] == ["ど", "old"]
    assert not (tmp_path / "song.v2m-tmp.ustx").exists()


def test_in_place_patch_aborts_without_save_ack(tmp_path, monkeypatch):
    from application.openutau_bridge import OpenUtauSaveError

    def refuse(_path, **_kwargs):
        raise OpenUtauSaveError("timeout")

    import application.openutau_bridge as bridge
    monkeypatch.setattr(bridge, "request_save", refuse)
    base = _write_ustx(tmp_path / "song.ustx", [NoteInfo(2.2, 2.6, 62.0, "old")])
    take = _write_ustx(tmp_path / "take.ustx", [NoteInfo(2.1, 2.4, 65.0, "new")])
    before = base.read_text(encoding="utf-8")

    try:
        splice_take_with_handshake(base, take, base, (2.0, 4.0))
        raise AssertionError("expected OpenUtauSaveError")
    except OpenUtauSaveError:
        pass
    assert base.read_text(encoding="utf-8") == before
    assert not (tmp_path / "song_backup.ustx").exists()


def test_reference_track_in_place_keeps_backup(tmp_path, monkeypatch):
    monkeypatch.setattr("application.openutau_bridge._running_openutau_dirs", lambda: [])
    base = _write_ustx(tmp_path / "song.ustx", [
        NoteInfo(1.0, 1.5, 60.0, "ど"),
        NoteInfo(2.2, 2.6, 62.0, "my"),
    ])
    take = _write_ustx(tmp_path / "take.ustx", [NoteInfo(2.1, 2.4, 65.0, "ref")])

    inserted = add_reference_take_with_handshake(base, take, base, (2.0, 4.0), "V2M参考 tifa-01")
    assert inserted == 1

    patched = yaml.safe_load(base.read_text(encoding="utf-8"))
    assert [n["lyric"] for n in patched["voice_parts"][0]["notes"]] == ["ど", "my"]
    assert patched["tracks"][0]["mute"] is True
    assert patched["tracks"][1]["mute"] is False
    assert patched["tracks"][1]["track_name"] == "V2M参考 tifa-01"
    backup = yaml.safe_load((tmp_path / "song_backup.ustx").read_text(encoding="utf-8"))
    assert len(backup["tracks"]) == 1


def test_backup_history_and_restore(tmp_path, monkeypatch):
    from application.sentence_jobs import list_backups, restore_backup_with_handshake

    monkeypatch.setattr("application.openutau_bridge._running_openutau_dirs", lambda: [])
    base = _write_ustx(tmp_path / "song.ustx", [NoteInfo(2.2, 2.6, 62.0, "v1")])
    take = _write_ustx(tmp_path / "take.ustx", [NoteInfo(2.1, 2.4, 65.0, "v2")])

    splice_take_with_handshake(base, take, base, (2.0, 4.0))
    history = list_backups(base)
    assert len(history) == 1
    assert "v1" in history[0].read_text(encoding="utf-8")

    restore_backup_with_handshake(base, history[0])
    restored = yaml.safe_load(base.read_text(encoding="utf-8"))
    assert [n["lyric"] for n in restored["voice_parts"][0]["notes"]] == ["v1"]
    # The pre-restore state (v2) is itself backed up, so restore is reversible.
    assert any("v2" in p.read_text(encoding="utf-8") for p in list_backups(base))


def test_restore_mix_with_handshake(tmp_path, monkeypatch):
    from application.sentence_jobs import restore_mix_with_handshake
    from inference.pipeline.sentence_gacha import snapshot_mix

    monkeypatch.setattr("application.openutau_bridge._running_openutau_dirs", lambda: [])
    base = _write_ustx(tmp_path / "song.ustx", [NoteInfo(2.2, 2.6, 62.0, "な")])
    take = _write_ustx(tmp_path / "take.ustx", [NoteInfo(2.1, 2.4, 65.0, "な")])
    snap = snapshot_mix(base)
    add_reference_take_with_handshake(base, take, base, (2.0, 4.0), "V2M参考 a")
    restore_mix_with_handshake(base, snap)
    patched = yaml.safe_load(base.read_text(encoding="utf-8"))
    assert patched["tracks"][0]["mute"] is False
    assert patched["tracks"][1]["mute"] is True


def test_separate_output_skips_handshake(tmp_path, monkeypatch):
    import application.openutau_bridge as bridge

    def fail(_path, **_kwargs):
        raise AssertionError("handshake must not run for a separate output")

    monkeypatch.setattr(bridge, "request_save", fail)
    base = _write_ustx(tmp_path / "song.ustx", [NoteInfo(2.2, 2.6, 62.0, "old")])
    take = _write_ustx(tmp_path / "take.ustx", [NoteInfo(2.1, 2.4, 65.0, "new")])
    out = tmp_path / "patched.ustx"

    removed, inserted = splice_take_with_handshake(base, take, out, (2.0, 4.0))
    assert (removed, inserted) == (1, 1)
    assert out.is_file()
    assert "old" in base.read_text(encoding="utf-8")


def test_rapid_patches_keep_unique_backups(tmp_path, monkeypatch):
    from application.sentence_jobs import list_backups

    monkeypatch.setattr("application.openutau_bridge._running_openutau_dirs", lambda: [])
    base = _write_ustx(tmp_path / "song.ustx", [NoteInfo(2.2, 2.6, 62.0, "v1")])
    take = _write_ustx(tmp_path / "take.ustx", [NoteInfo(2.1, 2.4, 65.0, "v2")])

    # Two patches within the same second must not collide on the backup name.
    splice_take_with_handshake(base, take, base, (2.0, 4.0))
    splice_take_with_handshake(base, take, base, (2.0, 4.0))
    history = list_backups(base)
    assert len(history) == 2
    assert len({p.name for p in history}) == 2


def test_expected_hash_guard_aborts_on_concurrent_edit(tmp_path, monkeypatch):
    from application.sentence_jobs import BaseChangedError, file_sha256

    monkeypatch.setattr("application.openutau_bridge._running_openutau_dirs", lambda: [])
    base = _write_ustx(tmp_path / "song.ustx", [NoteInfo(2.2, 2.6, 62.0, "old")])
    take = _write_ustx(tmp_path / "take.ustx", [NoteInfo(2.1, 2.4, 65.0, "new")])
    stale_hash = file_sha256(base)
    # Simulate an edit landing after confirmation but before the patch.
    base.write_text(base.read_text(encoding="utf-8").replace("old", "edited"), encoding="utf-8")
    before = base.read_text(encoding="utf-8")

    try:
        splice_take_with_handshake(base, take, base, (2.0, 4.0), expected_hash=stale_hash)
        raise AssertionError("expected BaseChangedError")
    except BaseChangedError:
        pass
    assert base.read_text(encoding="utf-8") == before
    assert not (tmp_path / "song_backup.ustx").exists()