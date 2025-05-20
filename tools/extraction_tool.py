import torch
import json
from typing import List, Dict, Union
from transformers import AutoProcessor, AutoModelForVision2Seq
from qwen_vl_utils import process_vision_info
from PIL import Image

# Load model and processor once
device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForVision2Seq.from_pretrained(
    "qwen_model", 
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto",
    trust_remote_code=True
)

processor = AutoProcessor.from_pretrained("qwen_model", trust_remote_code=True)

# Optional field aliases to guide Qwen
FIELD_ALIASES = {
    "mc": "MC (Medical Certificate)",
    "mc_days": "MC days covered",
    "mc_serial_number": "MC serial number",
    "log_number": "LOG (Letter of Guarantee) number",
    "diagnosis_code": "Diagnosis code (e.g. ICD-10)",
    "admission_hospital": "Admission hospital name",
    "admission_date": "Admission date",
    "claim_type": "Type of claim (e.g. outpatient, inpatient)",
    "company_name": "Patient's employer",
}

def _safe_parse_json(output_text: str) -> Union[Dict, str]:
    try:
        return json.loads(output_text.strip())
    except Exception:
        return {
            "error": "parse_failed",
            "raw": output_text.strip()
        }

def extract_fields_with_qwen(
    image_path: str,
    fields: List[str],
    debug: bool = False
) -> Union[Dict, str]:
    """
    Extract structured fields from a document image using Qwen-VL.

    Args:
        image_path (str): Path to the image file.
        fields (List[str]): Fields to extract based on doc type.
        debug (bool): Whether to print prompt and output.

    Returns:
        dict: Extracted data or error object.
    """
    # Build readable field labels for prompt
    pretty_fields = [f"- {FIELD_ALIASES.get(f, f)}" for f in fields]

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert insurance claim processor specializing in reading scanned claim forms "
                "and extracting structured data for digital processing. You understand insurance-specific "
                "terminology such as 'ward class', 'MC serial number', 'letter of guarantee', 'consultation', "
                "'claim type', and 'ineligible amount'."
            )
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": (
                    "Extract the following fields from this insurance claim document image:\n"
                    + "\n".join(pretty_fields) +
                    "\n\nRespond with only a valid JSON object.\n"
                    "Do not include any explanation, markdown, or comments. Use snake_case keys exactly as listed above.\n"
                    "Leave missing fields as empty strings (\"\") or null values.\n"
                    "Output example:\n"
                    "{\n  \"patient_name\": \"...\",\n  \"visit_date\": \"...\",\n  ...\n}"
                )}
            ]
        }
    ]

    # Build prompt
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Preprocess
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[prompt_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(device)

    # Generate
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=512)

    # Trim prompt from output
    trimmed_ids = [
        out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        trimmed_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    if debug:
        print("🔍 Prompt:\n", prompt_text)
        print("📤 Raw Output:\n", output_text)

    return _safe_parse_json(output_text)
