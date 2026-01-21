import os
import time
import requests
from typing import Optional
from dotenv import load_dotenv


class TelegramNotifier:
    def __init__(self, token: Optional[str] = None, chat_id: Optional[str] = None):
        # Load environment variables if not provided
        if not token or not chat_id:
            load_dotenv()

        self.token = token or os.environ.get("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID")
        self.api_base = (
            f"https://api.telegram.org/bot{self.token}" if self.token else None
        )

        self.last_sent_time = 0
        self.throttle_seconds = 30

    @property
    def is_enabled(self) -> bool:
        return bool(self.token and self.chat_id)

    def send(self, message: str, force: bool = False) -> bool:
        """Send message to Telegram with throttling."""
        if not self.is_enabled:
            return False

        now = time.time()
        if not force and (now - self.last_sent_time) < self.throttle_seconds:
            return False

        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "HTML",
            "disable_notification": False,
        }

        try:
            response = requests.post(
                f"{self.api_base}/sendMessage", json=payload, timeout=10
            )
            if response.status_code == 200:
                self.last_sent_time = now
                return True
        except Exception:
            pass  # Fail silently to not disrupt the main process

        return False

    def send_test(self, source: Optional[str] = None) -> bool:
        """Send a test message."""
        src_info = f" from <b>{source}</b>" if source else ""
        return self.send(
            f"🔔 <b>MLArena Notification Test</b>{src_info}\n\nTelegram bot is successfully connected!",
            force=True,
        )
