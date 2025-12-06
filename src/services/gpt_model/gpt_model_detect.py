import os
import json

import src.services.gpt_model.setup_gpt as setup_gpt
from src.etl.loading import supabase_loading 

client = setup_gpt.retrieve_client()

def prompt_gpt(model_id: str, supabase_img_url: str):
        
        _, user_content_prompt, _, _ = setup_gpt.get_prompt_content()

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

def detect_with_gpt_model(model_id: str, detect_path: str):

    _, _, _, supabase_parent_path = setup_gpt.get_prompt_content()

    bucket_url = 'pipeline'
    destination_path = 'jsons-fromGPT/2024'

    for img_url in os.listdir(detect_path):
        if img_url.split('.')[-1] not in ['jpg', 'png']:
            continue
        else:
            supabase_img_url = f'{supabase_parent_path}/{img_url}'
        
        if os.path.exists(f'{detect_path}/json/{img_url}.json'):
            pass
        else:
            try:
                print(f"Prompting ChatGPT {supabase_img_url}")
                obtained_dict = prompt_gpt(model_id = model_id, supabase_img_url = supabase_img_url)
            except Exception as e:
                print(f"Error {e}")
                continue

            with open(f'{detect_path}/json/{img_url}.json', 'w') as f:
                json.dump(obtained_dict, f)

        # TODO: I should check if the file already exists in the supabase directory
        # Load to supabase database
        print("Loading json file to supabase")
        supabase_loading.load_files_to_supabase_bucket(file_path=f'{detect_path}/json/{img_url}.json',
                                                        bucket_url= bucket_url,
                                                        destination_path= f'{destination_path}/{img_url}.json')