import os
import io
import base64
import logging
import requests
from typing import List, Dict, Optional
from PIL import Image
from llm import llm_config

logger = logging.getLogger(__name__)

VISION_MODEL_NAME = os.getenv("VISION_MODEL_NAME", "/app/models/Qwen3-VL-8B-Instruct")
VISION_API_BASE = os.getenv("VISION_API_BASE")

IMAGE_MIME_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/gif", "image/webp"}
PDF_MIME_TYPES = {"application/pdf"}

# Qwen-VL converts pixels -> visual tokens (~28x28 px per token after patch+merge).
# Capping the longest side keeps token usage bounded regardless of original file size.
MAX_IMAGE_DIMENSION = 1280
JPEG_QUALITY = 85


def _resize_image_if_needed(base64_data: str, mime_type: str, file_name: str) -> tuple:
    """
    Downscale image so its longest side <= MAX_IMAGE_DIMENSION px.
    Controls Qwen-VL visual-token count. Falls back to original data on
    any failure — never blocks the flow.
    """
    try:
        base64_data = base64_data.replace("\n", "").replace("\r", "").strip()
        raw_bytes = base64.b64decode(base64_data)
        img = Image.open(io.BytesIO(raw_bytes))

        width, height = img.size
        longest_side = max(width, height)

        if longest_side <= MAX_IMAGE_DIMENSION:
            return base64_data, mime_type

        scale = MAX_IMAGE_DIMENSION / longest_side
        new_size = (int(width * scale), int(height * scale))

        if img.mode != "RGB":
            img = img.convert("RGB")

        resized = img.resize(new_size, Image.LANCZOS)

        buffer = io.BytesIO()
        resized.save(buffer, format="JPEG", quality=JPEG_QUALITY)
        new_bytes = buffer.getvalue()

        logger.info(
            f"Resized image | file={file_name} "
            f"original={width}x{height} -> {new_size[0]}x{new_size[1]} "
            f"original_bytes={len(raw_bytes)} new_bytes={len(new_bytes)}"
        )

        return base64.b64encode(new_bytes).decode("utf-8"), "image/jpeg"

    except Exception as e:
        logger.error(f"Image resize failed | file={file_name} err={e} — using original")
        return base64_data, mime_type


def _describe_image(base64_data: str, mime_type: str, file_name: str) -> str:
    base64_data, mime_type = _resize_image_if_needed(base64_data, mime_type, file_name)

    data_uri = f"data:{mime_type};base64,{base64_data}"

    payload = {
        "model": VISION_MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Describe what is in this image. If it contains text "
                            "(screenshot, error message, document, ID card), transcribe "
                            "the visible text exactly. Be concise. No speculation."
                        )
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": data_uri}
                    }
                ]
            }
        ],
        "temperature": 0.0,
        "max_tokens": 800
    }

    headers = {
        "Authorization": f"Bearer {llm_config.token}",
        "Content-Type": "application/json"
    }

    base_url = VISION_API_BASE or llm_config.url

    try:
        response = requests.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=60,
            verify='./IDFCBANKCA.pem'
        )
        response.raise_for_status()
        data = response.json()
        description = data["choices"][0]["message"]["content"]
        logger.info(f"Image described | file={file_name}")
        return description.strip()
    except Exception as e:
        logger.error(f"Image description failed | file={file_name} err={e}")
        return "IMAGE UNREADABLE"


def _extract_pdf(base64_data: str, file_name: str) -> str:
    try:
        from pypdf import PdfReader

        pdf_bytes = base64.b64decode(base64_data)
        reader = PdfReader(io.BytesIO(pdf_bytes))

        text_parts = []
        for i, page in enumerate(reader.pages):
            page_text = (page.extract_text() or "").strip()
            if page_text:
                text_parts.append(f"[Page {i + 1}]\n{page_text}")

        if text_parts:
            combined = "\n\n".join(text_parts)
            if len(combined) > 5000:
                combined = combined[:5000] + "\n...(truncated)"
            logger.info(f"PDF extracted | file={file_name}")
            return combined
        return "PDF appears to be scanned; no extractable text."
    except Exception as e:
        logger.error(f"PDF extraction failed | file={file_name} err={e}")
        return "PDF UNREADABLE"


def _process_one(file: Dict) -> Optional[str]:
    file_name = file.get("fileName", "unknown")
    file_type = (file.get("fileType") or "").lower()
    encoding = (file.get("contentEncoding") or "").lower()
    content = file.get("fileContent")

    if not content or encoding != "base64":
        return None

    if file_type in IMAGE_MIME_TYPES:
        desc = _describe_image(content, file_type, file_name)
        return f"[Attached image: {file_name}]\n{desc}"

    if file_type in PDF_MIME_TYPES:
        text = _extract_pdf(content, file_name)
        return f"[Attached PDF: {file_name}]\n{text}"

    logger.warning(f"Skipping unsupported file type | file={file_name} type={file_type}")
    return None


def process_attachments(files: Optional[List[Dict]]) -> str:
    """
    Sync file processor. Returns "" if no files or any failure.
    Never raises — incident creation must succeed regardless.
    """
    if not files:
        return ""

    try:
        descriptions = []
        for f in files:
            result = _process_one(f)
            if result:
                descriptions.append(result)
        if not descriptions:
            return ""
        return "\n\n".join(descriptions)
    except Exception as e:
        logger.error(f"process_attachments failed: {e}")
        return ""
