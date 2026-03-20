"""
Email sender for delivering analysis reports and visualizations.
Attaches: strategy_report.md + all PNG charts from output/ and reports/figures/
"""
import os
import ssl
import logging
import argparse
import smtplib
from pathlib import Path
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def build_email_body(report_path: Path) -> str:
    if report_path.exists():
        content = report_path.read_text(encoding="utf-8")
        preview = content[:800].replace("#", "").strip()
        return f"""Xin chào,

Dưới đây là báo cáo phân tích Facebook tự động cho Di tích Nhà tù Hỏa Lò.
Báo cáo được tạo lúc: {datetime.now().strftime('%d/%m/%Y %H:%M')}

--- TÓM TẮT ---
{preview}
...

---
Chi tiết đầy đủ trong file đính kèm.
Charts trực quan cũng được đính kèm để dễ theo dõi.

Đây là email tự động từ pipeline Hỏa Lò Facebook Analysis.
"""
    return "Báo cáo phân tích đính kèm. Vui lòng kiểm tra file attachment."


def attach_file(msg: MIMEMultipart, filepath: Path):
    if not filepath.exists():
        logger.warning(f"Attachment not found: {filepath}")
        return

    with open(filepath, "rb") as f:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(f.read())
    encoders.encode_base64(part)
    part.add_header("Content-Disposition", f"attachment; filename={filepath.name}")
    msg.attach(part)
    logger.info(f"Attached: {filepath.name}")


def collect_attachments() -> list[Path]:
    """Collect all files to attach: report + charts."""
    attachments = []

    report = Path("reports/strategy_report.md")
    if report.exists():
        attachments.append(report)

    for chart_dir in [Path("output"), Path("reports/figures")]:
        if chart_dir.exists():
            for png in sorted(chart_dir.glob("*.png")):
                attachments.append(png)

    for json_file in [Path("output/EDA_REPORT.json"), Path("output/nlp_analysis_report.json")]:
        if json_file.exists():
            attachments.append(json_file)

    return attachments


def send_report():
    sender = os.getenv("EMAIL_SENDER")
    password = os.getenv("EMAIL_PASSWORD")
    recipients_str = os.getenv("EMAIL_RECIPIENTS", "")

    if not sender or not password or not recipients_str:
        logger.error("Email credentials not set. Set EMAIL_SENDER, EMAIL_PASSWORD, EMAIL_RECIPIENTS in .env")
        return False

    recipients = [r.strip() for r in recipients_str.split(",") if r.strip()]

    msg = MIMEMultipart()
    msg["Subject"] = f"[Hoa Lo Analysis] Báo cáo phân tích {datetime.now().strftime('%d/%m/%Y')}"
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)

    report_path = Path("reports/strategy_report.md")
    body = build_email_body(report_path)
    msg.attach(MIMEText(body, "plain", "utf-8"))

    attachments = collect_attachments()
    logger.info(f"Attaching {len(attachments)} file(s)...")
    for filepath in attachments:
        attach_file(msg, filepath)

    smtp_host = "smtp.gmail.com"
    smtp_port = 587

    try:
        context = ssl.create_default_context()
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls(context=context)
            server.login(sender, password)
            server.sendmail(sender, recipients, msg.as_string())
        logger.info(f"✅ Report sent to {', '.join(recipients)}")
        return True
    except Exception as e:
        logger.error(f"Failed to send email: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Send analysis report via email")
    parser.add_argument("--dry-run", action="store_true", help="Print email content without sending")
    args = parser.parse_args()

    if args.dry_run:
        logger.info("=== DRY RUN ===")
        report_path = Path("reports/strategy_report.md")
        print(build_email_body(report_path))
        attachments = collect_attachments()
        logger.info(f"Would attach {len(attachments)} file(s):")
        for a in attachments:
            logger.info(f"  - {a} ({a.stat().st_size / 1024:.0f} KB)")
        return

    send_report()


if __name__ == "__main__":
    main()
