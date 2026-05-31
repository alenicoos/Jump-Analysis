from __future__ import annotations

"""Small audio/text feedback helper.

Durante il setup possiamo dare indicazioni vocali. Su macOS usiamo `say`; sugli
altri sistemi lasciamo il print e un beep di terminale.
"""

import platform
import subprocess
import time
from dataclasses import dataclass


@dataclass
class AudioFeedback:
    """Gestisce messaggi stampati e, opzionalmente, pronunciati."""

    enabled: bool = True
    min_interval_seconds: float = 4.0

    def __post_init__(self) -> None:
        """Stato interno per evitare di ripetere lo stesso messaggio troppo spesso."""

        self._last_message: str | None = None
        self._last_spoken_at = 0.0

    def warn(self, message: str) -> None:
        """Messaggio di warning."""

        self.speak(f"Attenzione. {message}")

    def error(self, message: str) -> None:
        """Messaggio di errore."""

        self.speak(f"Errore. {message}")

    def speak(self, message: str, force: bool = False) -> None:
        """Stampa sempre il messaggio e, se abilitato, lo pronuncia."""

        print(message, flush=True)
        if not self.enabled:
            return

        # Rate limit: durante la webcam arrivano molti frame al secondo, quindi
        # senza questo controllo la voce ripeterebbe lo stesso warning di continuo.
        now = time.monotonic()
        if not force and message == self._last_message and now - self._last_spoken_at < self.min_interval_seconds:
            return
        if not force and now - self._last_spoken_at < self.min_interval_seconds:
            return

        self._last_message = message
        self._last_spoken_at = now

        # Darwin è il nome di sistema per macOS. `say` è un comando built-in che pronuncia il testo.
        if platform.system() == "Darwin":
            # `say` e' disponibile su macOS e non blocca il processo principale.
            subprocess.Popen(["say", message], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            print("\a", end="", flush=True)
