from crew.document_crew_runner import run_classification_crew
from dotenv import load_dotenv
load_dotenv() 

import os
os.getenv("OPENAI_API_KEY")
print("Loaded OPENAI_API_KEY:")

if __name__ == "__main__":
    # Example PDF path
    pdf_path = "your path"

    results = run_classification_crew(pdf_path)

    print("\n🧾 Classification Results:")
    for page, category in results.items():
        print(f"{page}: {category}")
