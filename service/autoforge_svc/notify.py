"""Reaching outward: saying something when there is nobody watching.

Everything else in this framework is a request the agent answers. This module
is the one place the agent starts a conversation, and that changes what
"honest" has to mean:

1. **A notification that did not go out must not look like one that did.** This
   is the single most dangerous failure available to this module. An agent that
   believes it told someone, when the webhook 404'd or the password was stale,
   will go on to act as though a human were informed. So every send returns a
   `Delivery` carrying what actually happened — the HTTP status, the SMTP
   exception — and `send` never reports success it did not observe. There is no
   silent fallback and no "best-effort" flag.

2. **An unconfigured channel is an error, not a no-op.** `send` with nothing
   configured raises. The tempting alternative — log a line and return — makes
   the agent's belief in a human being notified depend on a config file it
   cannot see the contents of.

3. **The credentials are never read back out.** They arrive from the config,
   go into a header or an SMTP login, and appear in no string this module
   returns. A `Delivery` reports the channel name and the status; a failure at
   login says "authentication failed", not the password that failed.

Two kinds of channel, because the environments differ: a webhook (a URL that
takes a JSON POST — a chat bridge, an automation runner, anything that speaks
HTTP) and email (SMTP, because that is what works when there is no public
endpoint to post to).

Transport is stdlib on purpose. This is the code that has to run on whatever
machine the agent was left on, and a notifications path that needs `requests`
installed is one more way for the agent to end up unable to speak.
"""
from __future__ import annotations

import json
import smtplib
import ssl
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from email.message import EmailMessage
from typing import Any

DEFAULT_TIMEOUT = 20.0

WEBHOOK = "webhook"
EMAIL = "email"
KINDS = (WEBHOOK, EMAIL)


class NotifyError(RuntimeError):
    """Nothing was sent: the request itself was not answerable."""


@dataclass
class Delivery:
    """What actually happened on one channel. Never optimistic."""

    channel: str
    kind: str
    ok: bool
    detail: str = ""
    status: int | None = None
    duration_ms: float = 0.0

    def line(self) -> str:
        where = f"http {self.status}" if self.status is not None else "sent"
        if self.ok:
            return f"{self.channel} ({self.kind}): {where}"
        return f"{self.channel} ({self.kind}): FAILED — {self.detail or where}"


@dataclass
class Channel:
    """One configured way to reach a human."""

    name: str
    kind: str
    url: str = ""
    method: str = "POST"
    headers: dict[str, str] = field(default_factory=dict)
    host: str = ""
    port: int = 0
    user: str = ""
    password: str = ""
    sender: str = ""
    to: list[str] = field(default_factory=list)
    tls: str = "starttls"      # ssl | starttls | none
    subject: str = "autoforge"
    timeout: float = DEFAULT_TIMEOUT

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Channel":
        if not isinstance(d, dict):
            raise NotifyError(f"channel must be an object, got {type(d).__name__}")
        kind = str(d.get("kind") or "").strip().lower()
        if kind not in KINDS:
            raise NotifyError(
                f"channel kind must be one of {KINDS}, got {kind or '(missing)'!r}")
        name = str(d.get("name") or kind)
        if kind == WEBHOOK:
            url = str(d.get("url") or "").strip()
            if not url:
                raise NotifyError(f"webhook channel {name!r} has no url")
            if not url.startswith(("http://", "https://")):
                raise NotifyError(
                    f"webhook channel {name!r}: url must start with http:// or https://")
            headers = d.get("headers") or {}
            if not isinstance(headers, dict):
                raise NotifyError(f"webhook channel {name!r}: headers must be an object")
            return cls(name=name, kind=WEBHOOK, url=url,
                       method=str(d.get("method") or "POST").upper(),
                       headers={str(k): str(v) for k, v in headers.items()},
                       timeout=float(d.get("timeout", DEFAULT_TIMEOUT)))
        # email
        host = str(d.get("host") or "").strip()
        if not host:
            raise NotifyError(f"email channel {name!r} has no host")
        to = d.get("to") or []
        if isinstance(to, str):
            to = [to]
        if not isinstance(to, list) or not to:
            raise NotifyError(f"email channel {name!r} has no 'to' address")
        port = int(d.get("port") or (465 if str(d.get("tls", "starttls")) == "ssl" else 587))
        return cls(
            name=name, kind=EMAIL, host=host, port=port,
            user=str(d.get("user") or ""), password=str(d.get("password") or ""),
            sender=str(d.get("sender") or d.get("user") or ""),
            to=[str(x) for x in to], tls=str(d.get("tls") or "starttls").lower(),
            subject=str(d.get("subject") or "autoforge"),
            timeout=float(d.get("timeout", DEFAULT_TIMEOUT)),
        )

    def to_dict(self) -> dict[str, Any]:
        """Safe to print or log: the password is replaced, not included.

        A config dump that carries the SMTP password ends up in a transcript, a
        bug report, or a screenshot. The field is present so a reader can see
        that one is set, and says only that.
        """
        base: dict[str, Any] = {"name": self.name, "kind": self.kind, "timeout": self.timeout}
        if self.kind == WEBHOOK:
            base.update({"url": self.url, "method": self.method,
                         "headers": sorted(self.headers)})
        else:
            base.update({"host": self.host, "port": self.port, "user": self.user,
                         "sender": self.sender, "to": list(self.to), "tls": self.tls,
                         "subject": self.subject,
                         "password": "(set)" if self.password else "(unset)"})
        return base

    def describe(self) -> str:
        if self.kind == WEBHOOK:
            return f"{self.name} [webhook] {self.method} {self.url}"
        return (f"{self.name} [email] {self.host}:{self.port} ({self.tls}) "
                f"{self.sender} -> {', '.join(self.to)}")


def channels_from_config(config: dict[str, Any] | None) -> tuple[list[Channel], list[str]]:
    """Read `notify.channels`. Same contract as the MCP reader: report, skip, carry on."""
    block = (config or {}).get("notify")
    if not isinstance(block, dict):
        return [], []
    raw = block.get("channels")
    if raw is None:
        return [], []
    if not isinstance(raw, list):
        return [], ["notify.channels is not a list; no channels were read"]

    channels: list[Channel] = []
    problems: list[str] = []
    for i, entry in enumerate(raw):
        try:
            channel = Channel.from_dict(entry)
        except NotifyError as exc:
            problems.append(f"notify channel #{i + 1}: {exc}")
            continue
        if isinstance(entry, dict) and entry.get("enabled") is False:
            continue
        channels.append(channel)
    return channels, problems


class Notifier:
    """Every configured channel, and an honest account of each send."""

    def __init__(self, channels: list[Channel] | None = None) -> None:
        self.channels = list(channels or [])

    def __bool__(self) -> bool:
        return bool(self.channels)

    @property
    def names(self) -> list[str]:
        return [c.name for c in self.channels]

    def describe(self) -> str:
        if not self.channels:
            return ("No notification channels are configured. Add them under "
                    "`notify.channels` in the config file.")
        lines = [f"{len(self.channels)} channel(s) configured:"]
        lines += [f"  {c.describe()}" for c in self.channels]
        return "\n".join(lines)

    def send(self, text: str, subject: str = "", only: list[str] | None = None,
             ) -> list[Delivery]:
        """Deliver to every selected channel. Raises only if none could be tried.

        A channel that fails does not stop the others: the point of two channels
        is that one may be unreachable. The returned list is the record, and the
        caller is expected to read it rather than assume.
        """
        if not self.channels:
            raise NotifyError(
                "no notification channels are configured, so nothing was sent. "
                "Add them under `notify.channels` in the config file.")
        chosen = [c for c in self.channels if only is None or c.name in only]
        if not chosen:
            raise NotifyError(
                f"no channel matches {only!r}; configured: {', '.join(self.names)}")
        if not text.strip():
            raise NotifyError("refusing to send an empty notification")

        out: list[Delivery] = []
        for channel in chosen:
            started = time.perf_counter()
            try:
                if channel.kind == WEBHOOK:
                    delivery = self._send_webhook(channel, text, subject)
                else:
                    delivery = self._send_email(channel, text, subject)
            except Exception as exc:  # noqa: BLE001 - a send must never raise upward
                delivery = Delivery(channel.name, channel.kind, False,
                                    f"{type(exc).__name__}: {exc}")
            delivery.duration_ms = (time.perf_counter() - started) * 1000
            out.append(delivery)
        return out

    # -- channels ------------------------------------------------------
    def _send_webhook(self, ch: Channel, text: str, subject: str) -> Delivery:
        payload = {"text": text, "subject": subject or ch.subject,
                   "source": "autoforge"}
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json; charset=utf-8",
                   "User-Agent": "autoforge-notify/1"}
        headers.update(ch.headers)
        request = urllib.request.Request(ch.url, data=body, headers=headers,
                                         method=ch.method)
        try:
            with urllib.request.urlopen(request, timeout=ch.timeout) as response:
                status = getattr(response, "status", None) or response.getcode()
                # Read a little so the connection closes cleanly; the body is
                # not interpreted -- webhooks answer with HTML, JSON, or nothing.
                try:
                    response.read(2048)
                except OSError:
                    pass
        except urllib.error.HTTPError as exc:
            return Delivery(
                ch.name, ch.kind, False,
                f"the endpoint answered {exc.code}"
                + (f" ({exc.reason})" if getattr(exc, "reason", None) else "")
                + " — the message was NOT delivered",
                status=exc.code,
            )
        except urllib.error.URLError as exc:
            return Delivery(ch.name, ch.kind, False,
                            f"could not reach the endpoint: {exc.reason}")
        except (TimeoutError, OSError) as exc:
            return Delivery(ch.name, ch.kind, False,
                            f"could not reach the endpoint: {type(exc).__name__}: {exc}")

        if not (200 <= int(status) < 300):
            return Delivery(ch.name, ch.kind, False,
                            f"the endpoint answered {status}, which is not a success "
                            f"— the message was NOT delivered", status=int(status))
        return Delivery(ch.name, ch.kind, True, status=int(status))

    def _send_email(self, ch: Channel, text: str, subject: str) -> Delivery:
        message = EmailMessage()
        message["From"] = ch.sender or ch.user
        message["To"] = ", ".join(ch.to)
        message["Subject"] = subject or ch.subject
        message.set_content(text)

        try:
            if ch.tls == "ssl":
                server: smtplib.SMTP = smtplib.SMTP_SSL(
                    ch.host, ch.port, timeout=ch.timeout,
                    context=ssl.create_default_context())
            else:
                server = smtplib.SMTP(ch.host, ch.port, timeout=ch.timeout)
            try:
                server.ehlo()
                if ch.tls == "starttls":
                    server.starttls(context=ssl.create_default_context())
                    server.ehlo()
                if ch.user:
                    server.login(ch.user, ch.password)
                server.send_message(message)
            finally:
                try:
                    server.quit()
                except (smtplib.SMTPException, OSError):
                    server.close()
        except smtplib.SMTPAuthenticationError as exc:
            # Names the failure, never the credential.
            return Delivery(ch.name, ch.kind, False,
                            f"the server rejected the login (code {exc.smtp_code}); "
                            f"the message was NOT delivered")
        except smtplib.SMTPRecipientsRefused as exc:
            return Delivery(ch.name, ch.kind, False,
                            f"every recipient was refused: "
                            f"{', '.join(str(k) for k in exc.recipients)}")
        except (smtplib.SMTPException, ssl.SSLError, OSError, TimeoutError) as exc:
            return Delivery(ch.name, ch.kind, False,
                            f"{type(exc).__name__}: {exc} — the message was NOT delivered")
        return Delivery(ch.name, ch.kind, True)

    def summary(self, deliveries: list[Delivery]) -> str:
        """One string the agent can act on: did anybody actually get told?"""
        if not deliveries:
            return "Nothing was sent."
        ok = [d for d in deliveries if d.ok]
        bad = [d for d in deliveries if not d.ok]
        lines = [d.line() for d in deliveries]
        if bad and not ok:
            lines.append("NOBODY WAS NOTIFIED — every channel failed.")
        elif bad:
            lines.append(f"{len(ok)} of {len(deliveries)} channels delivered.")
        return "\n".join(lines)


__all__ = [
    "Channel", "Delivery", "Notifier", "NotifyError", "WEBHOOK", "EMAIL",
    "KINDS", "channels_from_config",
]
