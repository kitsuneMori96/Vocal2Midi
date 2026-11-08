import json
import threading
import time

import pytest

from application.openutau_bridge import (
    BRIDGE_DIR_NAME,
    OpenUtauSaveError,
    instance_dir_for_project,
    request_save,
)


def _write_prefs(data_dir, recent):
    payload = json.dumps({"RecentFiles": recent}, ensure_ascii=False)
    (data_dir / "prefs.json").write_text(payload, encoding="utf-8-sig")


def test_instance_dir_matches_recent_project(tmp_path, monkeypatch):
    data_dir = tmp_path / "OpenUtau"
    data_dir.mkdir()
    project = tmp_path / "song.ustx"
    project.write_text("{}", encoding="utf-8")
    _write_prefs(data_dir, [str(tmp_path / "other.ustx"), str(project)])
    monkeypatch.setattr("application.openutau_bridge._running_openutau_dirs", lambda: [data_dir])

    assert instance_dir_for_project(project) == data_dir
    assert instance_dir_for_project(tmp_path / "missing.ustx") is None


def test_request_save_skips_when_not_running(tmp_path, monkeypatch):
    monkeypatch.setattr("application.openutau_bridge._running_openutau_dirs", lambda: [])
    request_save(tmp_path / "song.ustx", timeout=0.05)


def test_request_save_without_bridge_raises(tmp_path, monkeypatch):
    data_dir = tmp_path / "OpenUtau"
    data_dir.mkdir()
    monkeypatch.setattr(
        "application.openutau_bridge.instance_dir_for_project", lambda _path: data_dir
    )
    with pytest.raises(OpenUtauSaveError) as exc:
        request_save(tmp_path / "song.ustx", timeout=0.05)
    assert exc.value.reason == "no-bridge"


def _make_bridge(tmp_path, monkeypatch):
    bridge = tmp_path / "OpenUtau" / BRIDGE_DIR_NAME
    bridge.mkdir(parents=True)
    (bridge / "ready.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "application.openutau_bridge.instance_dir_for_project", lambda _path: bridge.parent
    )
    return bridge


def test_request_save_waits_for_ack(tmp_path, monkeypatch):
    bridge = _make_bridge(tmp_path, monkeypatch)
    request_path = bridge / "save-request.json"
    response_path = bridge / "save-response.json"

    def responder():
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if request_path.is_file():
                payload = json.loads(request_path.read_text(encoding="utf-8"))
                response_path.write_text(
                    json.dumps({"id": payload["id"], "saved": True}), encoding="utf-8-sig"
                )
                return
            time.sleep(0.01)

    thread = threading.Thread(target=responder)
    thread.start()
    try:
        request_save(tmp_path / "song.ustx", timeout=2.0, poll_interval=0.01)
    finally:
        thread.join()
    assert not request_path.exists()


def test_request_save_timeout_aborts(tmp_path, monkeypatch):
    bridge = _make_bridge(tmp_path, monkeypatch)
    with pytest.raises(OpenUtauSaveError) as exc:
        request_save(tmp_path / "song.ustx", timeout=0.05, poll_interval=0.01)
    assert exc.value.reason == "timeout"
    assert not (bridge / "save-request.json").exists()


def test_request_save_declined_aborts_early(tmp_path, monkeypatch):
    bridge = _make_bridge(tmp_path, monkeypatch)
    request_path = bridge / "save-request.json"
    response_path = bridge / "save-response.json"

    def responder():
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if request_path.is_file():
                payload = json.loads(request_path.read_text(encoding="utf-8"))
                response_path.write_text(
                    json.dumps({"id": payload["id"], "saved": False}), encoding="utf-8"
                )
                return
            time.sleep(0.01)

    thread = threading.Thread(target=responder)
    thread.start()
    try:
        with pytest.raises(OpenUtauSaveError) as exc:
            request_save(tmp_path / "song.ustx", timeout=2.0, poll_interval=0.01)
        assert exc.value.reason == "declined"
    finally:
        thread.join()


def test_request_save_ignores_stale_ack(tmp_path, monkeypatch):
    bridge = _make_bridge(tmp_path, monkeypatch)
    (bridge / "save-response.json").write_text(
        json.dumps({"id": "stale", "saved": True}), encoding="utf-8"
    )
    with pytest.raises(OpenUtauSaveError) as exc:
        request_save(tmp_path / "song.ustx", timeout=0.05, poll_interval=0.01)
    assert exc.value.reason == "timeout"