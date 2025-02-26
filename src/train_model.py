from openai import OpenAI
import openai
import time
import os
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
import json
import random
from src.etl.loading import supabase_loading

from IPython.display import display

openai.api_key = 'sk-vGXLTIxgYyXy5ouqNHkvzkZVJ6MrVlS0PHC1nnWzjaT3BlbkFJ6jIOA923yyIEXBsQHmehrSxtHzw5dZhWcW3zii79IA'
MODEL = "gpt-4o-2024-08-06"

client = OpenAI()

def get_prompt_content():

    system_content = {"role": "system", "content": """I am a chatbot to analyze images coming from the IFBB Bodybuilding competitions"""}

    user_content_prompt = """Please analyze this image and return a structured JSON with the following key-value pairs where all the lists have the same length so that I can create a pandas DataFrame from it. If any information (like a competitor's country or total points) is missing, use None for unknown places to maintain the same list length across all keys: place: A flat list of the places each competitor earned (use None for missing data). judging: A flat list of the judging score each competitor earned (use None for missing data). finals: A flat list of the finals score each competitor earned (use None for missing data).  total: A flat list of the total points of each competitor (use None for missing data). competitors_name: A flat list of competitors' names. country: A flat list of the competitors' respective countries (use None for missing data). competition_type: A flat list of the type of competition (e.g., "Men's Classic Physique - Open"). Make sure the length of this list is the same as the other lists in the response dictionary """

    user_content = { "role": "user", "content": user_content_prompt}

    supabase_parent_path = 'https://qesnrciwmhxfhdaojwwo.supabase.co/storage/v1/object/public/resultImages/2024'

    return(system_content, user_content_prompt,user_content, supabase_parent_path)

def prompt_gpt(model_id: str, supabase_img_url: str):
        
        _, user_content_prompt, _, _ = get_prompt_content()

        response = client.chat.completions.create(
            model = model_id, 
        response_format={"type": "json_object"},
        messages=[
        {
        "role": "user",

        "content": [
            {"type": "text", "text": user_content_prompt},
            {
            "type": "image_url",
            "image_url": {
                "url": f"{supabase_img_url}",
                },
                },
            ],
            }
            ],
            )
        obtained_dict = json.loads(response.choices[0].message.content)
        
        return(obtained_dict)


def test_gpt_model(model_id:str, test_path: str, results_path = None):

    metrics_aggregated = pd.DataFrame()

    _, _,_, supabase_parent_path  = get_prompt_content()

    def normalize_for_metrics(expected: dict, obtained: dict):
        expected_df = pd.DataFrame(expected)
        max_length = expected_df.shape[0]
        expected_df.loc[:,'finals'] = expected_df['finals'].fillna('0')
        
        # Keys that you want to ensure exist
        required_keys = [
            "competition", "location", "date", "competitors_name", "country",
            "judging", "finals", "total", "place", "competition_type"
        ]
        
        # Ensure all required keys exist in the obtained dictionary, initializing as empty lists if missing
        normalized_data = {}
        
        for key in required_keys:
            # Get the current list from obtained, defaulting to an empty list if the key is missing
            current_list = obtained.get(key, [])

            # Ensure the list has a length of max_length by either trimming or padding with empty strings
            if len(current_list) > max_length:
                normalized_data[key] = current_list[:max_length]  # Trim to max_length
            else:
                normalized_data[key] = current_list + [None] * (max_length - len(current_list))
        
        # Create DataFrame from the normalized data
        obtained_df = pd.DataFrame(normalized_data)
        expected_df = expected_df.fillna("MISSING")
        obtained_df.loc[:,'finals'] = obtained_df['finals'].fillna('0')
        obtained_df = obtained_df.fillna("MISSING").astype(str)
        
        return expected_df, obtained_df
    
    def compare_results(obtained_dict: pd.DataFrame, img_url: str):

        expected_json_file_path = f'./data/full_data/test/json/{img_url}.json'

        if os.path.exists(expected_json_file_path):
            with open(expected_json_file_path, "r") as file:
                json_file = json.load(file)
        else:
            return(pd.DataFrame())

        metrics = {}

        expected_df, obtained_df = normalize_for_metrics(expected = json_file, obtained= obtained_dict)

        expected_df_copy = expected_df.copy()
        obtained_df_copy = obtained_df.copy()

        common_columns = set(expected_df_copy.columns) & set(obtained_df.columns)

        for column in common_columns:

            expected_values = expected_df_copy[column]
            obtained_values = obtained_df_copy[column]

            precision = precision_score(expected_values, obtained_values, average='micro', zero_division=0)

            metrics[column] = {'Precision': precision}

            metrics_list = [
                {
                    'tag': metric,
                    'value': value, 
                    'img_url': img_url,
                    'column_name': column,
                    'nr_competitors': expected_df.shape[0]
                }
                for column, metric_values in metrics.items()
                for metric, value in metric_values.items()
            ]

        return(pd.DataFrame(metrics_list))
    
    images_to_test = os.listdir(test_path)
    
    for img in images_to_test:
        supabase_img_url = f'{supabase_parent_path}/{img}'
        if os.path.exists(f'./data/test_results/{results_path}{img}.json'):
            print("File already exists. Fetching from test_results folder")
            with open(f'./data/test_results/{results_path}{img}.json', "r") as file:
                obtained_dict = json.load(file)
        else:
            try:
                print("Prompting ChatGPT")
                obtained_dict = prompt_gpt(model_id = model_id, supabase_img_url = supabase_img_url)
            except Exception as e:
                print(f"Error {e}")
                continue
        metrics = compare_results(obtained_dict = obtained_dict, img_url = img)

        metrics_aggregated = pd.concat([metrics_aggregated, metrics])
    
    return(metrics_aggregated)
            

        



    
    




