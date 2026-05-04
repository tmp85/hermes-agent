"""Sophie Meadow Telegram attachment/link capture.

Profile-scoped helper for Sophie's passive Telegram group. Captures only when
@Sophie is explicitly invoked and writes only under the Meadow sources folder.
"""

from __future__ import annotations

import asyncio
import html
import json
import logging
import mimetypes
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

MEADOW_GROUP_ID = os.getenv("SOPHIE_MEADOW_TELEGRAM_CHAT_ID", "-5227578901")
SOURCES_ROOT = Path(
    os.getenv(
        "SOPHIE_MEADOW_SOURCES_DIR",
        "/Users/tycho/tycho-workspace/agents/sophie/projects/meadow/sources",
    )
)
MEADOW_REPO = SOURCES_ROOT.parent  # the meadow project root, also the git checkout
MAX_SLUG_WORDS = 6
MAX_INLINE_DOC_BYTES = 5 * 1024 * 1024
MAX_LINK_PREVIEW_WORDS = 200

_URL_RE = re.compile(r"https?://[^\s<>()\[\]{}\"']+", re.I)
_MENTION_RE = re.compile(r"(?i)(?:^|\W)@sophie(?:bot)?\b")


def is_sophie_profile() -> bool:
    home = os.getenv("HERMES_HOME", "")
    return Path(home).name == "sophie" or home.endswith("/profiles/sophie")


def in_meadow_group(message: Any) -> bool:
    chat = getattr(message, "chat", None)
    if not chat:
        return False
    chat_id = str(getattr(chat, "id", ""))
    if MEADOW_GROUP_ID and chat_id == str(MEADOW_GROUP_ID):
        return True
    title = str(getattr(chat, "title", "") or "")
    return "meadow" in title.lower()


def message_text(message: Any) -> str:
    return (getattr(message, "text", None) or getattr(message, "caption", None) or "").strip()


def mentions_sophie(text: str, bot_username: Optional[str] = None) -> bool:
    if not text:
        return False
    if _MENTION_RE.search(text):
        return True
    if bot_username:
        return re.search(rf"(?i)(?:^|\W)@{re.escape(bot_username)}\b", text) is not None
    return False


def has_media(message: Any) -> bool:
    return bool(
        getattr(message, "photo", None)
        or getattr(message, "document", None)
        or getattr(message, "voice", None)
        or getattr(message, "video", None)
        or getattr(message, "audio", None)
    )


def sender_name(message: Any) -> str:
    user = getattr(message, "from_user", None)
    if not user:
        return "unknown"
    parts = [getattr(user, "first_name", None), getattr(user, "last_name", None)]
    name = " ".join(p for p in parts if p).strip()
    return name or getattr(user, "username", None) or "unknown"


def strip_mentions(text: str, bot_username: Optional[str] = None) -> str:
    text = re.sub(r"(?i)@sophie(?:bot)?\b[,:\-]*\s*", "", text or "")
    if bot_username:
        text = re.sub(rf"(?i)@{re.escape(bot_username)}\b[,:\-]*\s*", "", text)
    return text.strip()


def slugify(seed: str) -> str:
    words = re.findall(r"[A-Za-z0-9]+", seed.lower())[:MAX_SLUG_WORDS]
    return "-".join(words) if words else "untitled"


def unique_capture_dir(slug: str, captured_at: datetime) -> Path:
    base_slug = f"{captured_at.date().isoformat()}-{slug or 'untitled'}"
    candidate = SOURCES_ROOT / base_slug
    n = 2
    while candidate.exists():
        candidate = SOURCES_ROOT / f"{base_slug}-{n}"
        n += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def yaml_escape(value: Any) -> str:
    if value is None or value == "":
        return ""
    s = str(value).replace("\\", "\\\\").replace('"', '\\"')
    if any(ch in s for ch in [":", "#", "\n", "'", '"']) or s.strip() != s:
        return f'"{s}"'
    return s


def frontmatter(*, captured_at: datetime, sender: str, message_id: Any, original_filename: str = "", content_type: str = "", trigger: str) -> str:
    fields = {
        "captured": captured_at.isoformat(),
        "sender": sender,
        "channel": "telegram-meadow-group",
        "telegram_message_id": message_id,
        "original_filename": original_filename or "",
        "content_type": content_type or "",
        "trigger": trigger,
    }
    lines = ["---"]
    for k, v in fields.items():
        lines.append(f"{k}: {yaml_escape(v)}")
    lines.append("---")
    return "\n".join(lines) + "\n\n"


def first_plain_words(text: str, limit: int) -> str:
    clean = re.sub(r"```.*?```", " ", text or "", flags=re.S)
    clean = re.sub(r"[#>*_`\[\]()]", " ", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    words = clean.split()
    return " ".join(words[:limit]) + ("…" if len(words) > limit else "")


def html_title_and_text(body: str) -> tuple[str, str]:
    title_match = re.search(r"<title[^>]*>(.*?)</title>", body, re.I | re.S)
    title = html.unescape(re.sub(r"\s+", " ", title_match.group(1)).strip()) if title_match else ""
    body_no_script = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", body, flags=re.I | re.S)
    text = re.sub(r"<[^>]+>", " ", body_no_script)
    text = html.unescape(re.sub(r"\s+", " ", text).strip())
    return title, text


def fetch_link(url: str) -> tuple[str, str, str]:
    req = Request(url, headers={"User-Agent": "Sophie Meadow capture/1.0"})
    with urlopen(req, timeout=20) as resp:
        content_type = resp.headers.get("content-type", "")
        raw = resp.read(2 * 1024 * 1024)
    text = raw.decode("utf-8", errors="replace")
    title, body_text = html_title_and_text(text)
    if not title:
        title = urlparse(url).netloc or url
    return title, body_text, content_type


async def local_short_summary(kind: str, text: str, fallback: str) -> str:
    # Local-only: do not call frontier providers. Use Ollama's OpenAI-compatible
    # endpoint when available; otherwise return a deterministic extract.
    base_url = os.getenv("SOPHIE_LOCAL_LLM_BASE_URL", "http://100.69.3.110:11434/v1").rstrip("/")
    model = os.getenv("SOPHIE_LOCAL_LLM_MODEL", "gemma4:e4b-mlx-bf16")
    prompt = (
        f"You are Sophie. In one short paragraph, explain why this {kind} matters "
        "for Project Meadow. Do not invent facts.\n\n"
        f"Content:\n{text[:5000]}"
    )

    def call() -> str:
        import urllib.request
        payload = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 180,
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{base_url}/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json", "Authorization": "Bearer ollama"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=45) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
        return (data.get("choices") or [{}])[0].get("message", {}).get("content", "").strip()

    try:
        result = await asyncio.to_thread(call)
        return result or fallback
    except Exception as exc:
        logger.warning("[SophieCapture] Local summary failed: %s", exc)
        return fallback


async def analyze_image_local(path: Path, caption: str = "") -> str:
    try:
        from tools.vision_tools import vision_analyze_tool
        prompt = (
            "Describe this image for Project Meadow source capture. Focus on observable facts, "
            "any text visible in the image, and why it may be relevant. Keep it concise."
        )
        if caption:
            prompt += f"\nCaption/context: {caption}"
        raw = await vision_analyze_tool(str(path), user_prompt=prompt)
        data = json.loads(raw)
        return data.get("analysis") or "Image captured. Local vision returned no description."
    except Exception as exc:
        logger.warning("[SophieCapture] Image analysis failed: %s", exc, exc_info=True)
        return f"Image captured. Local vision analysis failed: {exc}"


def safe_filename(name: str, fallback: str) -> str:
    name = Path(name or fallback).name
    cleaned = re.sub(r"[^A-Za-z0-9._ -]+", "_", name).strip(" .")
    return cleaned or fallback


async def download_file_to_path(fileish: Any, dest: Path) -> int:
    file_obj = await fileish.get_file()
    data = await file_obj.download_as_bytearray()
    raw = bytes(data)
    dest.write_bytes(raw)
    return len(raw)


def extract_document_preview(path: Path, mime_type: str, size: int) -> tuple[Optional[int], str, bool]:
    if size > MAX_INLINE_DOC_BYTES:
        return None, "deferred — too large for inline read", True
    suffix = path.suffix.lower()
    try:
        if suffix in {".txt", ".md", ".csv", ".json", ".html", ".htm"}:
            text = path.read_text(errors="replace")
            return None, first_plain_words(text, 220), False
        if suffix == ".pdf":
            try:
                import fitz  # PyMuPDF
                doc = fitz.open(str(path))
                page_count = doc.page_count
                text = "\n".join(page.get_text("text") for page in doc[: min(page_count, 5)])
                doc.close()
                return page_count, first_plain_words(text, 220), False
            except Exception:
                return None, "deferred — PDF text extraction unavailable", True
    except Exception as exc:
        logger.warning("[SophieCapture] Document preview failed: %s", exc)
    return None, "deferred — too large for inline read" if size > MAX_INLINE_DOC_BYTES else "deferred — unsupported inline read type", True


async def transcribe_media(path: Path) -> str:
    def call() -> str:
        from tools.transcription_tools import transcribe_audio
        result = transcribe_audio(str(path))
        if isinstance(result, dict):
            return result.get("transcript") or result.get("text") or str(result)
        return str(result)
    try:
        return await asyncio.to_thread(call)
    except Exception as exc:
        logger.warning("[SophieCapture] Transcription failed: %s", exc, exc_info=True)
        return f"[transcription failed: {exc}]"


def content_kind_from_message(message: Any) -> str:
    if getattr(message, "photo", None):
        return "image"
    if getattr(message, "voice", None):
        return "voice"
    if getattr(message, "video", None):
        return "video"
    if getattr(message, "audio", None):
        return "voice"
    if getattr(message, "document", None):
        return "document"
    return "unknown"


async def capture_image(message: Any, trigger_message: Any, trigger: str, bot_username: Optional[str]) -> Path:
    caption = strip_mentions(message_text(trigger_message) or message_text(message), bot_username)
    captured_at = datetime.now().astimezone()
    slug = slugify(caption or "image")
    out_dir = unique_capture_dir(slug, captured_at)
    photo = message.photo[-1]
    file_obj = await photo.get_file()
    ext = ".jpg"
    file_path = getattr(file_obj, "file_path", "") or ""
    for candidate in (".png", ".webp", ".gif", ".jpeg", ".jpg"):
        if file_path.lower().endswith(candidate):
            ext = candidate
            break
    image_path = out_dir / f"image{ext}"
    data = bytes(await file_obj.download_as_bytearray())
    image_path.write_bytes(data)
    mime = mimetypes.guess_type(str(image_path))[0] or f"image/{ext.lstrip('.')}"
    description = await analyze_image_local(image_path, caption)
    body = frontmatter(captured_at=captured_at, sender=sender_name(trigger_message), message_id=getattr(trigger_message, "message_id", ""), content_type=mime, trigger=trigger)
    body += description.strip() + "\n"
    if caption:
        body += f"\n## Caption\n\n{caption}\n"
    (out_dir / "index.md").write_text(body, encoding="utf-8")
    return out_dir


async def capture_document(message: Any, trigger_message: Any, trigger: str, bot_username: Optional[str]) -> Path:
    doc = message.document
    captured_at = datetime.now().astimezone()
    original = safe_filename(getattr(doc, "file_name", "") or "document", "document")
    slug = slugify(original)
    out_dir = unique_capture_dir(slug, captured_at)
    dest = out_dir / original
    size = await download_file_to_path(doc, dest)
    mime = getattr(doc, "mime_type", None) or mimetypes.guess_type(str(dest))[0] or "application/octet-stream"
    page_count, preview, deferred = extract_document_preview(dest, mime, size)
    summary = preview if deferred else await local_short_summary("document", preview, preview)
    body = frontmatter(captured_at=captured_at, sender=sender_name(trigger_message), message_id=getattr(trigger_message, "message_id", ""), original_filename=original, content_type=mime, trigger=trigger)
    body += "## File metadata\n\n"
    body += f"- filename: {original}\n- size: {size} bytes\n- type: {mime}\n"
    if page_count is not None:
        body += f"- page count: {page_count}\n"
    body += f"\n## Summary\n\n{summary}\n"
    caption = strip_mentions(message_text(trigger_message) or message_text(message), bot_username)
    if caption:
        body += f"\n## Caption\n\n{caption}\n"
    (out_dir / "index.md").write_text(body, encoding="utf-8")
    return out_dir


async def capture_voice_video(message: Any, trigger_message: Any, trigger: str, bot_username: Optional[str]) -> Path:
    captured_at = datetime.now().astimezone()
    is_video = bool(getattr(message, "video", None))
    media = message.video if is_video else (message.voice or message.audio)
    content_type = "video" if is_video else "voice"
    caption = strip_mentions(message_text(trigger_message) or message_text(message), bot_username)
    slug = slugify(caption or content_type)
    out_dir = unique_capture_dir(slug, captured_at)
    ext = ".mp4" if is_video else ".ogg"
    if getattr(media, "mime_type", None):
        ext = mimetypes.guess_extension(media.mime_type) or ext
    dest = out_dir / f"media{ext}"
    await download_file_to_path(media, dest)
    transcript = await transcribe_media(dest)
    body = frontmatter(captured_at=captured_at, sender=sender_name(trigger_message), message_id=getattr(trigger_message, "message_id", ""), content_type=content_type, trigger=trigger)
    body += transcript.strip() + "\n"
    if caption:
        body += f"\n## Caption\n\n{caption}\n"
    (out_dir / "transcript.md").write_text(body, encoding="utf-8")
    return out_dir


async def capture_link(trigger_message: Any, trigger: str, bot_username: Optional[str]) -> Path:
    text = message_text(trigger_message)
    match = _URL_RE.search(text)
    if not match:
        raise ValueError("No URL found in capture message")
    url = match.group(0).rstrip(".,);]")
    captured_at = datetime.now().astimezone()
    title, page_text, content_type = await asyncio.to_thread(fetch_link, url)
    slug = slugify(title or urlparse(url).netloc or "link")
    out_dir = unique_capture_dir(slug, captured_at)
    preview = first_plain_words(page_text, MAX_LINK_PREVIEW_WORDS)
    why = await local_short_summary("link", f"Title: {title}\nURL: {url}\nPreview: {preview}", "Captured for Meadow reference; review the linked material for relevance to current planning.")
    body = frontmatter(captured_at=captured_at, sender=sender_name(trigger_message), message_id=getattr(trigger_message, "message_id", ""), content_type="link", trigger=trigger)
    body += f"URL: {url}\n\n"
    body += f"Fetched title: {title}\n\n"
    body += f"## Preview\n\n{preview}\n\n"
    body += f"## Why this matters in Meadow\n\n{why}\n"
    (out_dir / "index.md").write_text(body, encoding="utf-8")
    return out_dir


async def capture_forward(trigger_message: Any, trigger: str, bot_username: Optional[str]) -> Path:
    captured_at = datetime.now().astimezone()
    text = strip_mentions(message_text(trigger_message), bot_username)
    slug = slugify(text or "forward")
    out_dir = unique_capture_dir(slug, captured_at)
    body = frontmatter(captured_at=captured_at, sender=sender_name(trigger_message), message_id=getattr(trigger_message, "message_id", ""), content_type="forward", trigger=trigger)
    body += text.strip() + "\n"
    (out_dir / "forward.md").write_text(body, encoding="utf-8")
    return out_dir


def _commit_and_push_sync(out_dir: Path, kind: str) -> None:
    """Stage, commit, and push the new capture from inside MEADOW_REPO.

    Failures are logged but never raised — the pull-rebase + janitor crons
    (and #2's retry loop in server.js) handle eventual convergence.
    """
    commit_env = {
        **os.environ,
        # Identify the gateway distinctly so commits are easy to filter.
    }
    git_id = [
        "-c", "user.email=sophie-meadow@tycho.local",
        "-c", "user.name=Sophie Meadow gateway",
    ]
    repo = ["-C", str(MEADOW_REPO)]
    try:
        subprocess.run(
            ["git", *repo, "add", "-A"],
            check=True, capture_output=True, timeout=20, env=commit_env,
        )
        status = subprocess.run(
            ["git", *repo, "status", "--porcelain"],
            check=True, capture_output=True, text=True, timeout=10, env=commit_env,
        )
        if not status.stdout.strip():
            logger.info("[SophieCapture] nothing to commit after capture %s", out_dir.name)
            return
        message = f"capture {kind}: {out_dir.name}"
        subprocess.run(
            ["git", *repo, *git_id, "commit", "-q", "-m", message],
            check=True, capture_output=True, timeout=20, env=commit_env,
        )
        try:
            subprocess.run(
                ["git", *repo, "push", "--quiet"],
                check=True, capture_output=True, timeout=30, env=commit_env,
            )
            logger.info("[SophieCapture] committed + pushed: %s", message)
        except subprocess.CalledProcessError as push_exc:
            stderr = (push_exc.stderr or b"").decode("utf-8", errors="replace").strip()
            logger.warning(
                "[SophieCapture] commit ok, push failed for %s: %s (cron will sync)",
                out_dir.name, stderr,
            )
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or b"").decode("utf-8", errors="replace").strip()
        logger.warning(
            "[SophieCapture] git op failed for %s: cmd=%r stderr=%s",
            out_dir.name, exc.cmd, stderr,
        )
    except Exception as exc:
        logger.warning(
            "[SophieCapture] commit/push failed for %s: %s", out_dir.name, exc, exc_info=True,
        )


async def commit_and_push(out_dir: Path, kind: str) -> None:
    await asyncio.to_thread(_commit_and_push_sync, out_dir, kind)


async def acknowledge(adapter: Any, message: Any, text: str) -> None:
    bot = getattr(adapter, "_bot", None)
    if not bot:
        return
    thread_id = getattr(message, "message_thread_id", None)
    kwargs = {
        "chat_id": int(getattr(message.chat, "id")),
        "text": text,
        "reply_to_message_id": int(getattr(message, "message_id")),
    }
    if thread_id is not None:
        kwargs["message_thread_id"] = int(thread_id)
    await bot.send_message(**kwargs)


def relative_capture_path(path: Path) -> str:
    try:
        return str(path.relative_to(SOURCES_ROOT.parents[2])) + "/"
    except Exception:
        return f"projects/meadow/sources/{path.name}/"


async def maybe_capture(adapter: Any, message: Any) -> bool:
    if not is_sophie_profile() or not in_meadow_group(message):
        return False
    bot_username = getattr(getattr(adapter, "_bot", None), "username", None)
    text = message_text(message)
    if not mentions_sophie(text, bot_username):
        return False

    target = message
    trigger = "explicit-capture"
    if has_media(message):
        trigger = "mention-with-attachment"
    elif getattr(message, "reply_to_message", None) is not None and has_media(message.reply_to_message):
        target = message.reply_to_message
        trigger = "reply-with-mention"
    elif _URL_RE.search(text):
        trigger = "explicit-capture"
    elif getattr(message, "forward_origin", None) or getattr(message, "forward_from", None) or getattr(message, "forward_sender_name", None):
        trigger = "explicit-capture"
    else:
        return False

    try:
        kind = content_kind_from_message(target)
        if kind == "image":
            out_dir = await capture_image(target, message, trigger, bot_username)
            capture_kind = "image"
        elif kind == "document":
            out_dir = await capture_document(target, message, trigger, bot_username)
            capture_kind = "document"
        elif kind in {"voice", "video"}:
            out_dir = await capture_voice_video(target, message, trigger, bot_username)
            capture_kind = kind
        elif _URL_RE.search(text):
            out_dir = await capture_link(message, trigger, bot_username)
            capture_kind = "link"
        else:
            out_dir = await capture_forward(message, trigger, bot_username)
            capture_kind = "forward"
        await commit_and_push(out_dir, capture_kind)
        await acknowledge(adapter, message, f"Captured → {relative_capture_path(out_dir)}")
        logger.info("[SophieCapture] Captured Telegram source to %s", out_dir)
        return True
    except Exception as exc:
        logger.warning("[SophieCapture] Capture failed: %s", exc, exc_info=True)
        try:
            await acknowledge(adapter, message, f"Capture failed → {exc}")
        finally:
            return True
