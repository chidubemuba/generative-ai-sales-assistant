
import ollama
from openai import OpenAI 
import json
import re

def init_client(ollama_server,api_key = "ollama"):
    client = OpenAI(
        base_url=ollama_server,
        api_key=api_key,  
    )
    return client


def build_prompt(thread, company ="Google"):

  system = f"""
  You are a zero-shot multi-class classification AI. Your role is to classify a string from a sales conversation 
  between a seller and a prospective customer {company}. You should classify the string into one of the categories below by assigning 
  softmax probabilities to each label.

  1. Rapport building: establishing a connection and trust with the prospect
  2. Needs assessment: asking open-ended questions to understand the prospect's needs, challenges, and goals
  3. Handling objections: customizing conversation to focus on solutions relevant to the buyer's specific situation
  4. Decision Makers: strategy and efforts made to ensure the decision-maker is involved in the conversation
  5. Wrapping up a call: Clarity and commitment in setting next steps or follow-up actions


#   Your response MUST be a valid JSON object with your predictions. If it is not a JSON object you will FAIL THIS TASK. 
  """

  user = f"""email thread:{thread}\n\n\n
  INSTRUCTIONS: Your response MUST be a valid JSON object with the 10 k:v pairs below, along with your sigmoid predictions for each. Use this format:
  {{"label":"Rapport_building", "predicted_confidence":<float value representing a softmax prediction between 0-1>,
    "label":"Needs_assesment", "predicted_confidence":<float value representing a softmax prediction between 0-1>,
    "label":"Handling_objections", "predicted_confidence":<float value representing a softmax prediction between 0-1>,
    "label":"Decision_makers", "predicted_confidence":<float value representing a softmax prediction between 0-1>,
    "label":"Wrapping_up_a_call", "predicted_confidence":<float value representing a softmax prediction between 0-1>,
    "label":"Other", "predicted_confidence":<float value representing a softmax prediction between 0-1>,
    etc.}}
  If you do not return a JSON object you will FAIL THIS TASK. 
  If the predicted confidences do not sum to 1, you will FAIL THIS TASK!"""

  return system,user

def run_llm_call(client, model, email, stream = False, company = "Google"):
    # build prompts
    system, user = build_prompt(email,company)

    # Create a chat completion
    response = client.chat.completions.create(
        model=model,
        temperature = 0,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        stream=stream
    )
    
    # Extract usage stats
    usage_stats = dict(dict(response)["usage"])

    # Convert the response to a dictionary that can be serialized
    response_dict = {
        "id": response.id,
        "created": response.created,
        "model": response.model,
        "object": response.object,
        "system_fingerprint": response.system_fingerprint,
        "completion_tokens": usage_stats['completion_tokens'],
        "prompt_tokens": usage_stats['prompt_tokens'],
        "total_tokens": usage_stats['total_tokens'],
        "choices": [{"content": choice.message.content} for choice in response.choices]
    }

    return response_dict


def extract_json(text):
    try:
        # Regex pattern to match JSON content
        pattern = re.compile(r'\{.*\}', re.DOTALL)
        match = pattern.search(text)
        if match:
            json_str = match.group()
            
            # Fix common JSON format issues
            json_str = json_str.replace('\n', '').replace(',}', '}')
            json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas before closing braces

            # Manually parse JSON to handle repeated keys
            entries = re.findall(r'\"label\":\s*\"(.*?)\",\s*\"predicted_confidence\":\s*(\d*\.?\d+)', json_str)
            standardized_json = [{"label": label, "predicted_confidence": float(conf)} for label, conf in entries]
            
            return standardized_json
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return None
