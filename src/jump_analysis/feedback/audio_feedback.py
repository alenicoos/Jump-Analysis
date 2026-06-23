from __future__ import annotations

"""Small audio/text feedback helper.

During setup we can give spoken guidance. On macOS we use `say`; on other
systems we keep terminal text and a bell.
"""

import platform
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Literal


SpeechKind = Literal["step", "error", "warning"]


@dataclass
class AudioFeedback:
    """Prints messages and optionally speaks them."""

    enabled: bool = True
    min_interval_seconds: float = 4.0

    def __post_init__(self) -> None:
        """Internal state used to avoid repeating the same message too often."""

        self._last_spoken_by_message: dict[str, float] = {}
        self._active_process: subprocess.Popen | None = None
        self._active_kind: SpeechKind | None = None
        self._pending_speech: tuple[str, SpeechKind] | None = None
        self._lock = threading.Lock()

    def warn(self, message: str) -> None:
        """Warning message."""

        self.speak(message, kind="warning")

    def error(self, message: str) -> None:
        """Error message."""

        self.speak(message, kind="error")

    def speak(self, message: str, force: bool = False, kind: SpeechKind = "step") -> None:
        """Always print the message and speak it when audio is enabled.

        The rate limit is per message. Repeating the same sentence too soon is
        suppressed, but a different sentence can be queued. Speech never
        overlaps. A new step may interrupt a still-playing error because the
        next step means that error condition has cleared.
        """

        print(message, flush=True)
        if not self.enabled:
            return

        now = time.monotonic()
        last_spoken_at = self._last_spoken_by_message.get(message, 0.0)
        if not force and now - last_spoken_at < self.min_interval_seconds:
            return

        self._last_spoken_by_message[message] = now
        with self._lock:
            active_running = self._active_process is not None and self._active_process.poll() is None
            if active_running:
                if self._active_kind == "error" and kind == "step":
                    self._stop_active_speech_locked()
                else:
                    self._pending_speech = (message, kind)
                    return
            self._start_speech_locked(message, kind)

    def _start_speech_locked(self, message: str, kind: SpeechKind) -> None:
        """Start speaking while holding the internal lock."""

        self._active_kind = kind
        # Darwin is the system name for macOS. `say` is a built-in text-to-speech command.
        if platform.system() == "Darwin":
            self._active_process = subprocess.Popen(["say", message], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            threading.Thread(target=self._monitor_active_speech, daemon=True).start()
        else:
            print("\a", end="", flush=True)
            self._active_process = None
            self._active_kind = None

    def _stop_active_speech(self) -> None:
        """Stop the previous spoken message before starting a new one."""

        with self._lock:
            self._stop_active_speech_locked()

    def _stop_active_speech_locked(self) -> None:
        """Stop the previous spoken message while holding the internal lock."""

        if self._active_process is None:
            return
        if self._active_process.poll() is None:
            self._active_process.terminate()
        self._active_process = None
        self._active_kind = None

    def _monitor_active_speech(self) -> None:
        """Start the latest pending message after the current one finishes."""

        process = self._active_process
        if process is None:
            return
        process.wait()
        with self._lock:
            if self._active_process is not process:
                return
            self._active_process = None
            self._active_kind = None
            pending = self._pending_speech
            self._pending_speech = None
            if pending is not None:
                self._start_speech_locked(*pending)
