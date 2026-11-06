import logging
import pathlib
import sys
import traceback
from contextlib import contextmanager

from PySide6.QtCore import QThread, Signal

from application.config import PipelineConfig
from application.exceptions import CancellationError
from gui.i18n import tr


class _QtLogHandler(logging.Handler):
    """Forwards logging records from the pipeline into the GUI log window."""

    def __init__(self, signal):
        super().__init__(level=logging.INFO)
        self.signal = signal

    def emit(self, record):
        try:
            self.signal.emit(record.getMessage())
        except Exception:
            self.handleError()


class _Streams:
    def __init__(self, stream, signal):
        self.stream = stream
        self.signal = signal

    def write(self, text):
        if text.strip():
            self.signal.emit(text.strip())
        self.stream.write(text)

    def flush(self):
        self.stream.flush()


@contextmanager
def _capture_output(signal):
    """Mirror stdout/stderr and logging records into ``signal``."""
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = _Streams(old_stdout, signal)
    sys.stderr = _Streams(old_stderr, signal)
    root_logger = logging.getLogger()
    old_root_level = root_logger.level
    handler = _QtLogHandler(signal)
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)
    try:
        yield
    finally:
        root_logger.removeHandler(handler)
        root_logger.setLevel(old_root_level)
        sys.stdout = old_stdout
        sys.stderr = old_stderr


try:
    from application.pipeline import run_auto_lyric_job, open_asr_session
    HYBRID_AVAILABLE = True
except ImportError as e:
    print(f"WARNING: Hybrid pipeline not available. Error: {e}")
    HYBRID_AVAILABLE = False


class WorkerThread(QThread):
    log_signal = Signal(str)
    finished_signal = Signal(str)
    error_signal = Signal(str)
    progress_signal = Signal(int, int, str)  # (current file index, total files, filename)

    def __init__(self, tasks: list[tuple[PipelineConfig, str]]):
        """Initialize the worker with a list of ``(config, filename)`` tasks.

        Each task carries its own PipelineConfig so files can be customized
        individually in batch mode.
        """
        super().__init__()
        self.tasks = tasks
        self._is_running = True

    def run(self):
        total = len(self.tasks)
        session = None
        try:
            # One lazily-spawned ASR subprocess serves the whole batch; the
            # multi-GB Qwen model loads once instead of per file.
            session = open_asr_session() if HYBRID_AVAILABLE else None
            with _capture_output(self.log_signal):
                save_dirs = {str(task_config.output_dir) for task_config, _ in self.tasks}
                save_dir = save_dirs.pop() if len(save_dirs) == 1 else "; ".join(sorted(save_dirs))

                for file_index, (config, filename) in enumerate(self.tasks, start=1):
                    if not self._is_running:
                        break
                    self.log_signal.emit(tr("worker_processing", f=filename))

                    config.audio_path = str(pathlib.Path(filename))
                    config.output_filename = filename
                    config.asr_session = session
                    config.cancel_checker = lambda: (
                        not self._is_running
                    ) or self.isInterruptionRequested()

                    self.progress_signal.emit(file_index, total, filename)
                    run_auto_lyric_job(config)

                if self._is_running:
                    self.finished_signal.emit(tr("worker_success", d=save_dir))
                else:
                    self.error_signal.emit(tr("worker_cancelled"))

        except (InterruptedError, CancellationError):
            self.error_signal.emit(tr("worker_stopped"))
        except Exception:
            self.error_signal.emit(tr("worker_error", tb=traceback.format_exc()))
        finally:
            if session is not None:
                session.close()

    def stop(self):
        self._is_running = False
        self.requestInterruption()


class FunctionWorker(QThread):
    """Run one callable off the UI thread with the pipeline's log plumbing.

    The callable gets ``session`` (a shared ASR subprocess when requested) and
    ``cancel_checker``; whatever it returns is left in ``self.result`` for the
    page to pick up in its finished handler.
    """

    log_signal = Signal(str)
    finished_signal = Signal(str)
    error_signal = Signal(str)

    def __init__(self, fn, *, needs_asr_session: bool = False):
        super().__init__()
        self._fn = fn
        self._needs_asr_session = needs_asr_session
        self._is_running = True
        self.result = None

    def _cancelled(self) -> bool:
        return not self._is_running or self.isInterruptionRequested()

    def run(self):
        session = None
        try:
            session = open_asr_session() if (self._needs_asr_session and HYBRID_AVAILABLE) else None
            with _capture_output(self.log_signal):
                self.result = self._fn(session=session, cancel_checker=self._cancelled)
            if self._cancelled():
                self.error_signal.emit(tr("worker_cancelled"))
            else:
                self.finished_signal.emit("")
        except (InterruptedError, CancellationError):
            self.error_signal.emit(tr("worker_stopped"))
        except Exception:
            self.error_signal.emit(tr("worker_error", tb=traceback.format_exc()))
        finally:
            if session is not None:
                session.close()

    def stop(self):
        self._is_running = False
        self.requestInterruption()
