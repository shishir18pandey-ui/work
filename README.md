file processor.py 
import os
import base64
import logging
import requests
from typing import List, Dict, Optional
from llm import llm_config

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
                    {"type": "image_url", "image_url": data_uri}
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




if not incident:
            # NEW INCIDENT
            file_description = ""
            if incident_data.files:
                file_dicts = [f.model_dump() for f in incident_data.files]
                file_description = process_attachments(file_dicts)
                logger.info(f"  Files processed | incident={incident_id} count={len(file_dicts)} has_description={bool(file_description)}")

            incident_record = incident_data.model_dump()
            incident_record.pop("files", None)   # ← strip heavy base64 before Kafka/DB
            incident_record["file_description"] = file_description
            incident_record["created_at"] = datetime.now().isoformat()
            incident_record["status"] = "created"
            incident_record["interaction_counter"] = 0
            incident_record["headers"] = get_header_details()

            try:
                await upsert_incident_payload_async(incident_id, json.dumps(incident_record))
                logger.info(f"  DB saved | incident={incident_id}")
            except Exception as e:
                logger.error(f"✗ DB upsert FAILED | incident={incident_id} error={e}")
                raise HTTPException(status_code=500, detail="Incident server error")

            publish_to_kafka(incident_id, "new_incident", incident_record)
            logger.info(f"✓ New incident done | incident={incident_id}")
            return ServiceNowIncidentResponse(code="202", details="Accepted")



else:
            # EXISTING INCIDENT
            additional_comments = incident_data.additionalComments
            payload = incident['payload']

            if additional_comments:
                current_status = get_incident_status(incident_id)
                logger.info(f"  Existing incident | incident={incident_id} status={current_status}")
                span.set_attribute("current_status", current_status or "unknown")

                if current_status == 'on_hold':
                    payload["additionalComments"] = additional_comments

                    if incident_data.files:
                        file_dicts = [f.model_dump() for f in incident_data.files]
                        new_file_description = process_attachments(file_dicts)
                        if new_file_description:
                            payload["file_description"] = new_file_description
                            logger.info(f"  Follow-up files processed | incident={incident_id}")

                    set_incident_status(incident_id, 'in_progress')
                    publish_to_kafka(incident_id, "additional_comments", payload)
                    logger.info(f"✓ Additional comments published | incident={incident_id}")

                elif current_status == 'in_progress':
                    logger.info(f"  Ignored | incident={incident_id} reason=already_in_progress")

                elif current_status in ['resolved', 'rejected']:
                    logger.info(f"  Ignored | incident={incident_id} reason=final_status={current_status}")

                else:
                    logger.info(f"  Ignored | incident={incident_id} reason=unknown_status={current_status}")

            else:
                handle_no_comments(incident_id)

            return ServiceNowIncidentResponse(code="202", details="Accepted")






def payload_to_incident_description(payload):
    tracer = get_tracer(__name__)
    with tracer.start_as_current_span("payload_to_incident_description") as span:
        span.set_attribute("short_description_length", len(payload.get('shortDescription','')))
        span.set_attribute("description_length", len(payload.get('description','')))
        span.set_attribute("individualUCIC", payload.get('individualUCIC','i'))

        short_description = payload.get('shortDescription','')
        description = payload.get('description','')
        individualUCIC = payload.get('individualUCIC','i')
        result = f"Short Description: {short_description}\nDescription: {description}"

        file_description = payload.get('file_description', '')
        if file_description:
            result = result + "\n\n--- ATTACHED FILES ---\n" + file_description

        print(f"Generated incident description for UCIC {individualUCIC}")
        return result, individualUCIC









        fixes 








            








        
