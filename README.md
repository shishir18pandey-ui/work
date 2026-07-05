import os
import base64
import logging
import requests
from typing import List, Dict, Optional
from utils.llm import llm_config

logger = logging.getLogger(__name__)

VISION_MODEL_NAME = os.getenv("VISION_MODEL_NAME", "/app/models/Qwen3-VL-8B-Instruct")
VISION_API_BASE = os.getenv("VISION_API_BASE")

IMAGE_MIME_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/gif", "image/webp"}
PDF_MIME_TYPES = {"application/pdf"}


def _describe_image(base64_data: str, mime_type: str, file_name: str) -> str:
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
                    {"type": "image_url", "image_url": {"url": data_uri}}
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
        import io
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
        else:
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
    Never raises — incident flow must continue regardless.
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
        return "\n\n--- ATTACHED FILES ---\n" + "\n\n".join(descriptions)
    except Exception as e:
        logger.error(f"process_attachments failed: {e}")
        return ""
















        def payload_to_incident_description(payload):
    from utils.file_processor import process_attachments   # ← NEW LINE

    tracer = get_tracer(__name__)
    with tracer.start_as_current_span("payload_to_incident_description") as span:
        span.set_attribute("short_description_length", len(payload.get('shortDescription','')))
        span.set_attribute("description_length", len(payload.get('description','')))
        span.set_attribute("individualUCIC", payload.get('individualUCIC','i'))
        
        short_description = payload.get('shortDescription','')
        description = payload.get('description','')
        individualUCIC = payload.get('individualUCIC','i')
        result = f"Short Description: {short_description}\nDescription: {description}"

        file_text = process_attachments(payload.get('files') or [])   # ← NEW LINE
        if file_text:                                                  # ← NEW LINE
            result = result + "\n\n" + file_text                       # ← NEW LINE

        print(f"Generated incident description for UCIC {individualUCIC}")
        return result, individualUCIC





        openai:
  vl_base_url: "https://qwen3-vl-8b.iservebetter.idfcfirstbank.com/v1"
  vl_model_name: "/app/models/Qwen3-VL-8B-Instruct"




  - name: VISION_API_BASE
  value: "{{ .Values.openai.vl_base_url }}"
- name: VISION_MODEL_NAME
  value: "{{ .Values.openai.vl_model_name }}"
