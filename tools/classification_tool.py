from crewai.tools import BaseTool
from typing import Optional
from utils.qwen_doc_classifier import classify_image_with_qwen, load_categories

class DocumentClassificationTool(BaseTool):
    name: str = "Document Classifier"
    description: str = (
        "Classifies document images into categories (MC, invoice, etc.) "
        "using QWEN model. Returns the classification result."
    )

    def _run(self, image_path: str) -> str:
        """Main execution method for the tool"""
        try:
            categories = load_categories()
            result = classify_image_with_qwen(image_path, categories)
            return f"Classification result: {result}"
        except Exception as e:
            return f"Classification failed: {str(e)}"