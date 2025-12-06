import src.etl.supabase_config as supabase_config

supabase = supabase_config.retrieve_supabase_client()

def read_files_from_supabase_bucket(bucket_url: str, folder: str = None):
    response = supabase.storage.from_(bucket_url).list(
        folder,
        {"limit": 100, "offset": 0, "sortBy": {"column": "name", "order": "desc"}}
        )
    
    return(response)

def get_all_buckets():
    response = supabase.storage.list_buckets()
    return(response)
