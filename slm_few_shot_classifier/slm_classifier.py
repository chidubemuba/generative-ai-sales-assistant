import torch
from transformers import pipeline

# Check if CUDA is available
device = 0 if torch.cuda.is_available() else -1


# Initialize the zero-shot classification pipeline
classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli",
                      device = device)

def classify_text(text, categories):
    """
    Classify the given text into one of the specified categories using a pre-trained classifier.

    Args:
        text (str): The text to be classified.
        categories (list of str): A list of categories to classify the text into.

    Returns:
        tuple: A tuple containing:
            - str: The label of the highest-scoring category.
            - float: The score associated with the highest-scoring category.
    """
    result = classifier(text, categories)
    return result['labels'][0], result['scores'][0]

# Define the topic categories
topic_categories = ["pricing", "competition", "technical_issues", "security_compliance", "other"]

# Define the guideline checklist
guideline_checklist = ["build_rapport","active listening", "objection_handling", 
                       "identify_decision_makers", "call_wrapup"]

# Function to classify text
def classify_text(text, categories):
    result = classifier(text, categories)
    return result['labels'][0], result['scores'][0]

