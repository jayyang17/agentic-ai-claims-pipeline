import yaml
from utils.utils import convert_pdf_to_images
from crewai import Crew
from tasks.classification_task import create_classification_task

def load_categories(config_path="configs/config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)["categories"]

def run_classification_crew(pdf_path: str, config_path="configs/config.yaml"):
    categories = load_categories(config_path)
    image_paths = convert_pdf_to_images(pdf_path)
    results = {}

    for i, img_path in enumerate(image_paths):
        print(f"\n📄 Page {i+1}")
        task = create_classification_task(img_path, categories)
        crew = Crew(agents=[task.agent], tasks=[task], verbose=True)
        result = crew.kickoff()
        results[f"page_{i+1}"] = result.strip()

    return results
