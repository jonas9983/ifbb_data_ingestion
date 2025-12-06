import src.services.gpt_model.setup_gpt as setup_gpt

import time
import json
import shutil
import os
from sklearn.model_selection import train_test_split
import sys


client = setup_gpt.retrieve_client()

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
        system_content, _, user_content, supabase_parent_path = setup_gpt.get_prompt_content()
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

def train_custom_model(training_file_name: str, validation_file_name: str, model_id: str):
    def upload_file(file_name: str, purpose: str) -> str:
        with open(file_name, "rb") as file_fd:
            response = client.files.create(file=file_fd, purpose=purpose)
        return response.id

    def configure_file_ids(training_file_id: str, validation_file_id):
        response = client.fine_tuning.jobs.create(
            training_file=training_file_id,
            validation_file=validation_file_id,
            model=model_id,
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
            status = response.status.lower()
            
            if status == 'succeeded':
                # If the job succeeded, retrieve the fine-tuned model ID
                response = client.fine_tuning.jobs.retrieve(job_id)
                fine_tuned_model_id = response.fine_tuned_model
                print("Fine-tuning completed. Model ID:", fine_tuned_model_id)
                return fine_tuned_model_id
            elif status == 'failed':
                # Handle the case where fine-tuning failed
                raise Exception("Fine-tuning failed. Check logs for details.")
            
            # Wait for some time before polling again
            print("Fine-tuning in progress... Current status:", status)
            time.sleep(10)  # Poll every 10 seconds
    
    print("Uploading training and validation files ")
    training_file_id = upload_file(training_file_name, "fine-tune")
    validation_file_id = upload_file(validation_file_name, "fine-tune")
    print("Files uploaded")

    print(training_file_id)
    print(validation_file_id)

    print("Configuring file ids")
    job_id = configure_file_ids(training_file_id= training_file_id, validation_file_id= validation_file_id)

    print(job_id)
    
    print("Fine tuning model")
    fine_tuned_model_id = retrieve_fine_tuned_model(job_id)

    return(fine_tuned_model_id)

if __name__ == '__main__':
    model_id = sys.argv[1]
    num_images = sys.argv[2]
    training_file_name = './data/train_test_split/train_fine_tuning.json'
    validation_file_name = './data/train_test_split/validation_fine_tuning.json'


    configure_messages_to_fine_tune_model(num_images= num_images)

    train_custom_model(training_file_name= training_file_name,
                       validation_file_name = validation_file_name,
                       model_id = model_id)

