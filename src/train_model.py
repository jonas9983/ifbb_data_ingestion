from openai import OpenAI
import openai
import time
import os
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import json
import random

from IPython.display import display

openai.api_key = 'sk-vGXLTIxgYyXy5ouqNHkvzkZVJ6MrVlS0PHC1nnWzjaT3BlbkFJ6jIOA923yyIEXBsQHmehrSxtHzw5dZhWcW3zii79IA'
MODEL = "gpt-4o-2024-08-06"

client = OpenAI()

def get_prompt_content():

    system_content = {"role": "system", "content": """I am a chatbot to analyze images coming from the IFBB Bodybuilding competitions"""}

    user_content_prompt = """Please analyze this image and return a structured JSON with the following key-value pairs where all the lists have the same length so that I can create a pandas DataFrame from it. If any information (like a competitor's country or total points) is missing, use None for unknown places to maintain the same list length across all keys: place: A flat list of the places each competitor earned (use None for missing data). judging: A flat list of the judging score each competitor earned (use None for missing data). finals: A flat list of the finals score each competitor earned (use None for missing data).  total: A flat list of the total points of each competitor (use None for missing data). competitors_name: A flat list of competitors' names. country: A flat list of the competitors' respective countries (use None for missing data). competition_type: A flat list of the type of competition (e.g., "Men's Classic Physique - Open"). Make sure the length of this list is the same as the other lists in the response dictionary """

    user_content = { "role": "user", "content": user_content_prompt}

    supabase_parent_path = 'https://qesnrciwmhxfhdaojwwo.supabase.co/storage/v1/object/public/finetuning/2024'

    return(system_content, user_content_prompt,user_content, supabase_parent_path)

def configure_messages_to_fine_tune_model(num_images: None):
    
    def create_train_test_folders():
        """ Each time this function is called the data is
        """

        train_test_parent_path = './data/train_test_split'

        train_folder_imgs = f'{train_test_parent_path}/train/imgs'
        validation_folder_imgs = f'{train_test_parent_path}/validation/imgs'

        train_folder_json = f'{train_test_parent_path}/train/json'
        validation_folder_json = f'{train_test_parent_path}/validation/json'

        if os.path.exists(train_test_parent_path):
            shutil.rmtree(train_test_parent_path)

        os.makedirs(train_folder_imgs, exist_ok=True)
        os.makedirs(validation_folder_imgs, exist_ok=True)
        os.makedirs(train_folder_json, exist_ok=True)
        os.makedirs(validation_folder_json, exist_ok=True)

        return(train_folder_imgs, validation_folder_imgs, train_folder_json, validation_folder_json)
    
    def split_data_train_test(test_size = 0.2, random_state = 42):
        """Randomly split the data into training and validation test sets 
        """
        img_folder = './data/full_data/imgs/'
        json_folder = './data/full_data/json/'

        train_folder_imgs, validation_folder_imgs, train_folder_json, validation_folder_json = create_train_test_folders()

        all_files_imgs = os.listdir(img_folder)

        if num_images is not None:
            random.seed(random_state)
            all_files_imgs = random.sample(all_files_imgs, num_images)

        train_files_imgs, test_files_imgs = train_test_split(
                all_files_imgs, test_size=test_size, random_state=random_state
            )

        # Move files to respective folders
        for file in train_files_imgs:
            shutil.copy(os.path.join(img_folder, file), os.path.join(train_folder_imgs, file))
            shutil.copy(os.path.join(json_folder, f'{file}.json'), os.path.join(train_folder_json, f'{file}.json'))

        for file in test_files_imgs:
            shutil.copy(os.path.join(img_folder, file), os.path.join(validation_folder_imgs, file))
            shutil.copy(os.path.join(json_folder, f'{file}.json'), os.path.join(validation_folder_json, f'{file}.json'))
        
        print(f"Training files: {len(train_files_imgs)}, Testing files: {len(test_files_imgs)}")
    
    def create_messages_json(path: str):
        """Create json file with messages to train chatGPT

        Args:
            path (str): Parent path for the images and json values. Should either be 'train' or 'validation'
        """
        system_content, _, user_content, supabase_parent_path = get_prompt_content()
        parent_path = './data/train_test_split'
        all_images = os.listdir(f'{parent_path}/{path}/imgs')
        for img_filename in all_images:
            supabase_path = f'{supabase_parent_path}/{img_filename}'
            temp_dict = {"messages": []}
            user_content_imgs = {"role": "user", "content": [
                {"type": "image_url",
                "image_url": {
                    "url": f"{supabase_path}"
                }}
            ]}
            with open(f'{parent_path}/{path}/json/{img_filename}.json', "r") as file:
                json_file = json.load(file)

            json_stringify = json.dumps(json_file)
            
            assintant_content = { "role": "assistant", "content": json_stringify}

            temp_dict["messages"].append(system_content)
            temp_dict["messages"].append(user_content)
            temp_dict['messages'].append(user_content_imgs)
            temp_dict['messages'].append(assintant_content)

            with open(f'{parent_path}/{path}_fine_tuning.json', 'a') as file:
                file.write(json.dumps(temp_dict) + '\n') # use `json.loads` to do the reverse
    
    split_data_train_test()
    create_messages_json(path = 'train')
    create_messages_json(path = 'validation')

def train_custom_model(training_file_name: str, validation_file_name: str, model: str):
    def upload_file(file_name: str, purpose: str) -> str:
        with open(file_name, "rb") as file_fd:
            response = client.files.create(file=file_fd, purpose=purpose)
        return response.id

    def configure_file_ids(training_file_id: str, validation_file_id):
        response = client.fine_tuning.jobs.create(
            training_file=training_file_id,
            validation_file=validation_file_id,
            model=model,
            suffix="recipe-ner",
        )
        
        job_id = response.id

        return(job_id)

    def retrieve_fine_tuned_model(job_id: str):

        response = client.fine_tuning.jobs.list_events(job_id)
        events = response.data
        events.reverse()
        while True:
            # Retrieve the fine-tuning job details
            response = client.fine_tuning.jobs.retrieve(job_id)
            
            # Check the status of the job
            status = response.get('status', '').lower()
            
            if status == 'succeeded':
                # If the job succeeded, retrieve the fine-tuned model ID
                fine_tuned_model_id = response.get('fine_tuned_model')
                print("Fine-tuning completed. Model ID:", fine_tuned_model_id)
                return fine_tuned_model_id
            elif status == 'failed':
                # Handle the case where fine-tuning failed
                raise Exception("Fine-tuning failed. Check logs for details.")
            
            # Wait for some time before polling again
            print("Fine-tuning in progress... Current status:", status)
            time.sleep(10)  # Poll every 10 seconds
    

    training_file_id = upload_file(training_file_name, "fine-tune")
    validation_file_id = upload_file(validation_file_name, "fine-tune")

    job_id = configure_file_ids(training_file_id= training_file_id, validation_file_id= validation_file_id)

    fine_tuned_model_id = retrieve_fine_tuned_model(job_id)

    return(fine_tuned_model_id)

def test_gpt_model(model_id:str):

    metrics_aggregated = pd.DataFrame()

    _, user_content_prompt,_, supabase_parent_path  = get_prompt_content()

    def prompt_gpt(supabase_img_url: str):

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
        img_url_temp = supabase_img_url.split('/')[-1]
        with open(f"./data/test_results/{img_url_temp}.json", 'w') as f:
            json.dump(obtained_dict, f)
        return(pd.DataFrame(obtained_dict))
    
    def compare_results(obtained_df: pd.DataFrame, img_url: str):

        with open(f'./data/full_data/test/json/{img_url}.json', "r") as file:
            json_file = json.load(file)
        expected_df = pd.DataFrame(json_file)

        metrics = {}

        expected_df_copy = expected_df.copy()
        obtained_df_copy = obtained_df.copy()

        common_columns = set(expected_df_copy.columns) & set(obtained_df_copy.columns)
        expected_df_copy = expected_df_copy[list(common_columns)].sort_index(axis=1).copy()
        obtained_df_copy = obtained_df_copy[list(common_columns)].sort_index(axis=1).copy()


        for column in common_columns:
            if column in ['finals', 'judging', 'place', 'total']:
                expected_df_copy[column] = expected_df_copy[column].fillna(0).astype(str)
                obtained_df_copy[column] = obtained_df_copy[column].fillna(0).astype(str)

            if expected_df_copy[column].dtype == 'object':  # Categorical columns
                expected_values = expected_df_copy[column]
                obtained_values = obtained_df_copy[column]

                precision = precision_score(expected_values, obtained_values, average='micro', zero_division=0)
                recall = recall_score(expected_values, obtained_values, average='micro', zero_division=0)
                f1 = f1_score(expected_values, obtained_values, average='micro', zero_division=0)

                metrics[column] = {'Precision': precision, 'Recall': recall, 'F1-score': f1}

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
    
    images_to_test = os.listdir('./data/full_data/test/imgs/')
    
    for img in images_to_test:
        supabase_img_url = f'{supabase_parent_path}/{img}'
        if os.path.exists(f'./data/test_results/{img}.json'):
            print("File already exists. Fetching from test_results folder")
            with open(f'./data/test_results/{img}.json', "r") as file:
                json_file = json.load(file)
            obtained_df = pd.DataFrame(json_file)
        else:
            try:
                print("Prompting ChatGPT")
                obtained_df = prompt_gpt(supabase_img_url = supabase_img_url)
            except Exception as e:
                print(f"Error {e}")
                continue
        metrics = compare_results(obtained_df = obtained_df, img_url = img)

        metrics_aggregated = pd.concat([metrics_aggregated, metrics])
    
    return(metrics_aggregated)


    
    




