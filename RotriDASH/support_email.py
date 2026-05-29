import os
import smtplib
import ssl
from email.message import EmailMessage


def send_support_email(subject: str, body: str, reply_to: str | None = None) -> tuple[bool, str]:
    """Send a support email via SMTP using environment configuration.

    Expected environment variables:
      - SUPPORT_SMTP_HOST (required)
      - SUPPORT_SMTP_PORT (default: 587)
      - SUPPORT_SMTP_USERNAME (optional; defaults to FROM address)
      - SUPPORT_SMTP_PASSWORD (required for authenticated SMTP)
      - SUPPORT_SMTP_USE_TLS (default: true)
      - SUPPORT_EMAIL_TO (default: support.rotrix@reude.tech)
      - SUPPORT_EMAIL_FROM (default: SUPPORT_EMAIL_TO)
    """
    host = os.getenv("SUPPORT_SMTP_HOST")
    if not host:
        return False, "Support email is not configured (SUPPORT_SMTP_HOST is missing)."

    to_addr = os.getenv("SUPPORT_EMAIL_TO", "kandanvadivel222@gmail.com")
    from_addr = os.getenv("SUPPORT_EMAIL_FROM", to_addr)
    port_str = os.getenv("SUPPORT_SMTP_PORT", "587")
    try:
        port = int(port_str)
    except ValueError:
        port = 587

    username = os.getenv("SUPPORT_SMTP_USERNAME", from_addr)
    password = os.getenv("SUPPORT_SMTP_PASSWORD", "")
    if not password:
        return False, "Support email password is not configured (SUPPORT_SMTP_PASSWORD)."

    use_tls = os.getenv("SUPPORT_SMTP_USE_TLS", "true").lower() in ("1", "true", "yes")

    msg = EmailMessage()
    msg["Subject"] = subject or "Rotrix support request"
    msg["From"] = from_addr
    msg["To"] = to_addr
    if reply_to:
        msg["Reply-To"] = reply_to
    msg.set_content(body or "")

    try:
        if port == 465 and use_tls:
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL(host, port, context=context, timeout=15) as server:
                server.login(username, password)
                server.send_message(msg)
        else:
            with smtplib.SMTP(host, port, timeout=15) as server:
                if use_tls:
                    context = ssl.create_default_context()
                    server.starttls(context=context)
                server.login(username, password)
                server.send_message(msg)
        return True, "Support request sent successfully."
    except Exception as exc:
        return False, f"Failed to send support email: {exc}"

