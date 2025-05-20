from crew.document_crew_runner import run_classification_crew

if __name__ == "__main__":
    # Example PDF path
    pdf_path = r"C:\Users\lee_jayyang\PythonProjects\agentic_ai\attachments\Group Outpatient Medical Claim Form (1 May 2025) - Yenny.pdf"

    results = run_classification_crew(pdf_path)

    print("\n🧾 Classification Results:")
    for page, category in results.items():
        print(f"{page}: {category}")
