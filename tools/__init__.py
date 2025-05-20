def classify_image_with_qwen(image, categories):
    prompt = f"""
You are a document classification agent. 
Given this page, classify it into one of the following categories:
{', '.join(categories)}.

Respond with one category name only.
"""
    # REPLACE THIS with real Qwen-VL call
    simulated_response = "claim_form"  # for testing
    return simulated_response.strip().lower()
