from transformers import AutoProcessor, AutoModelForVision2Seq
from qwen_vl_utils import process_vision_info
from PIL import Image
import torch
import yaml

device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForVision2Seq.from_pretrained(
    "qwen_model",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto",
    trust_remote_code=True
)

processor = AutoProcessor.from_pretrained("qwen_model", trust_remote_code=True)

def classify_image_with_qwen(image_path: str, categories: list[str]) -> str:
    prompt_text = (
        "Classify the following document page into one of the following categories:\n"
        + "\n".join(f"- {cat}" for cat in categories) +
        "\n\nRespond with only the category label. Do not include explanations, formatting, or any other output."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt_text}
            ]
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=512)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    return output_text.strip()


def load_categories(config_path="configs/config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)["categories"]
