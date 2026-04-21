import ifbb_data_ingestion.etl.supabase_config as supabase_config

supabase = supabase_config.retrieve_supabase_client()

def load_files_to_supabase_bucket(file_path: str, bucket_url: str, destination_path: str):
    
    with open(file_path, 'rb') as f:
        response = supabase.storage.from_(bucket_url).upload( 
            path = destination_path,
            file = f,
            file_options={"cache-control": "3600", "upsert": "false"}
        )
    
    return(response)

def delete_files_from_supabase_bucket(file_path: str, bucket_url: str):
    response = supabase.storage.from_(bucket_url).remove([file_path])
    if len(response) > 0:
        print(f"File deleted successfully")
        return(response)
    else:
        print("No file to delete")