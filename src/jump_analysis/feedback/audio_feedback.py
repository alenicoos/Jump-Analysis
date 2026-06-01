from __future__ import annotations

"""Small audio/text feedback helper.

During setup we can give spoken guidance. On macOS we use `say`; on other
systems we keep terminal text and a bell.
"""

import platform
import subprocess
import time
from dataclasses import dataclass


@dataclass
class AudioFeedback:
    """Prints messages and optionally speaks them."""

    enabled: bool = True
    min_interval_seconds: float = 4.0

    def __post_init__(self) -> None:
        """Internal state used to avoid repeating the same message too often."""

        self._last_spoken_by_message: dict[str, float] = {}
        self._active_process: subprocess.Popen | None = None

    def warn(self, message: str) -> None:
        """Warning message."""

        self.speak(f"Warning. {message}")

    def error(self, message: str) -> None:
        """Error message."""

        self.speak(f"Error. {message}")

    def speak(self, message: str, force: bool = False) -> None:
        """Always print the message and speak it when audio is enabled.

        The rate limit is per message. Repeating the same sentence too soon is
        suppressed, but a different sentence is spoken immediately.
        """

        print(message, flush=True)
        if not self.enabled:
            return

        now = time.monotonic()
        last_spoken_at = self._last_spoken_by_message.get(message, 0.0)
        if not force and now - last_spoken_at < self.min_interval_seconds:
            return

        self._last_spoken_by_message[message] = now
        self._stop_active_speech()

        # Darwin is the system name for macOS. `say` is a built-in text-to-speech command.
        if platform.system() == "Darwin":
            self._active_process = subprocess.Popen(["say", message], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            print("\a", end="", flush=True)

    def _stop_active_speech(self) -> None:
        """Stop the previous spoken message before starting a new one."""

        if self._active_process is None:
            return
        if self._active_process.poll() is None:
            self._active_process.terminate()
        self._active_process = None
