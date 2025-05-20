from crewai import Agent
from tools.classification_tool import DocumentClassificationTool

def create_classifier_agent():
    classify_tool = DocumentClassificationTool()

    return Agent(
        role="Document Specialist",
        goal="Accurately classify the uploaded document image into one of the known insurance document types.",
        backstory="You are an expert in insurance form taxonomy.",
        tools=[classify_tool],
        verbose=True
    )
