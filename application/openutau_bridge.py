"""Locate the OpenUtau instance the user is editing in.

OpenUtau keeps its preferences (including the recent-file list, whose first
entry is the project it opened last) in its data directory: the executable's
own folder in portable mode, or a per-user folder when installed. Reading that
list lets Vocal2Midi default the sentence-repair base file to the project the
piano roll is actually showing.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _running_openutau_dirs() -> list[Path]:
    """Data-dir candidates taken from live OpenUtau processes (Windows)."""
    if sys.platform != "win32":
        return []
    import ctypes
    from ctypes import wintypes

    class ProcessEntry32W(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD),
            ("th32DefaultHeapID", ctypes.POINTER(ctypes.c_ulong)),
            ("th32ModuleID", wintypes.DWORD),
            ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD),
            ("pcPriClassBase", ctypes.c_long),
            ("dwFlags", wintypes.DWORD),
            ("szExeFile", ctypes.c_wchar * 260),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel32.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessEntry32W)]
    kernel32.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessEntry32W)]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.QueryFullProcessImageNameW.argtypes = [
        wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)
    ]
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]

    dirs: list[Path] = []
    snapshot = kernel32.CreateToolhelp32Snapshot(0x00000002, 0)
    if not snapshot or snapshot == wintypes.HANDLE(-1).value:
        return dirs
    try:
        entry = ProcessEntry32W()
        entry.dwSize = ctypes.sizeof(ProcessEntry32W)
        ok = kernel32.Process32FirstW(snapshot, ctypes.byref(entry))
        while ok:
            if entry.szExeFile.lower() == "openutau.exe":
                handle = kernel32.OpenProcess(0x1000, False, entry.th32ProcessID)
                if handle:
                    try:
                        buf = ctypes.create_unicode_buffer(32768)
                        size = wintypes.DWORD(len(buf))
                        if kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
                            dirs.append(Path(buf.value).parent)
                    finally:
                        kernel32.CloseHandle(handle)
            ok = kernel32.Process32NextW(snapshot, ctypes.byref(entry))
    finally:
        kernel32.CloseHandle(snapshot)
    return dirs


def openutau_data_dirs() -> list[Path]:
    """Candidate data directories, running instances first."""
    home = Path.home()
    dirs = _running_openutau_dirs()
    if sys.platform == "win32":
        dirs.append(home / "Documents" / "OpenUtau")
    elif sys.platform == "darwin":
        dirs.append(home / "Library" / "OpenUtau")
    else:
        dirs.append(Path(os.environ.get("XDG_DATA_HOME", str(home / ".local" / "share"))) / "OpenUtau")
    seen: set[Path] = set()
    unique: list[Path] = []
    for directory in dirs:
        resolved = directory.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(directory)
    return unique


def read_recent_projects(data_dir: Path) -> list[str]:
    prefs = data_dir / "prefs.json"
    try:
        # OpenUtau writes prefs.json with a UTF-8 BOM.
        payload = json.loads(prefs.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return []
    recent = payload.get("RecentFiles") or []
    return [str(item) for item in recent if isinstance(item, str) and item]


def current_project_path() -> Path | None:
    """The project OpenUtau opened most recently, if it still exists."""
    for directory in openutau_data_dirs():
        for recent in read_recent_projects(directory):
            path = Path(recent)
            if path.is_file():
                return path
    return None