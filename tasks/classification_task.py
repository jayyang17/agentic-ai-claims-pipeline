from crewai import Task
from agents.classification_agent import create_classifier_agent  # import the factory

def create_classification_task(image_path: str, categories: list[str]) -> Task:
    agent = create_classifier_agent()   # always build a fresh agent

    prompt = (
        "Classify the document image into one of the following categories:\n"
        + "\n".join(f"- {cat}" for cat in categories)
        + "\n\nOnly return the category label."
    )

    return Task(
        description=prompt,
        agent=agent,
        expected_output="A single document category label.",
        input={"image_path": image_path}  # correct field for passing args
    )
