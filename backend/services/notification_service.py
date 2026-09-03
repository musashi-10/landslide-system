"""
Notification provider abstraction.

Hierarchy
---------
NotificationProvider (ABC)
    MockNotificationProvider   – logs to stdout; default & test
    EmailNotificationProvider  – SMTP stub (sends real email when configured)

New providers (SMS, webhook, push) can be added by implementing the ABC.
"""
from __future__ import annotations

import logging
import smtplib
from abc import ABC, abstractmethod
from email.message import EmailMessage

from alerts.models import Alert

logger = logging.getLogger(__name__)


class NotificationProvider(ABC):
    """Abstract notification provider."""

    @abstractmethod
    def send(self, alert: Alert) -> None:
        """Dispatch a notification for the given alert."""


class MockNotificationProvider(NotificationProvider):
    """Logs alert details; records sent alerts for test inspection."""

    def __init__(self) -> None:
        self.sent: list[Alert] = []

    def send(self, alert: Alert) -> None:
        self.sent.append(alert)
        logger.info(
            "[MOCK NOTIFICATION] %s | %s | %s | %s",
            alert.alert_id,
            alert.location_id,
            alert.risk_level,
            alert.message,
        )


class EmailNotificationProvider(NotificationProvider):
    """
    SMTP email provider stub.

    Sends a plain-text email when SMTP credentials are configured.
    Not scientifically validated — for operational use only.
    """

    def __init__(
        self,
        smtp_host: str,
        smtp_port: int,
        smtp_user: str,
        smtp_password: str,
        recipients: list[str],
    ) -> None:
        self._host = smtp_host
        self._port = smtp_port
        self._user = smtp_user
        self._password = smtp_password
        self._recipients = recipients

    def send(self, alert: Alert) -> None:
        if not self._recipients:
            logger.warning("EmailNotificationProvider: no recipients configured.")
            return

        msg = EmailMessage()
        msg["Subject"] = f"[LANDSLIDE ALERT] {alert.risk_level} — {alert.location_id}"
        msg["From"] = self._user or "alerts@landslide-system"
        msg["To"] = ", ".join(self._recipients)
        msg.set_content(
            f"Alert ID   : {alert.alert_id}\n"
            f"Location   : {alert.location_id}\n"
            f"Risk Level : {alert.risk_level}\n"
            f"Status     : {alert.status}\n"
            f"Timestamp  : {alert.timestamp_utc}\n\n"
            f"{alert.message}\n\n"
            "This is an automated alert from the Landslide Early Warning System.\n"
            "Do not reply to this email."
        )

        try:
            with smtplib.SMTP(self._host, self._port, timeout=10) as server:
                server.starttls()
                if self._user and self._password:
                    server.login(self._user, self._password)
                server.send_message(msg)
            logger.info("Email alert sent for %s to %s.", alert.alert_id, self._recipients)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to send email alert: %s", exc)
            raise
