import supabase
from supabase import create_client, Client
import os
from os.path import join, dirname
from dotenv import load_dotenv

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

SUPABASE_URL: str = os.environ.get("SUPABASE_URL")
SUPABASE_KEY: str = os.environ.get("SUPABASE_KEY")
SUPABASE_USERNAME = os.environ.get("SUPABASE_USERNAME")
SUPABASE_PASSWORD = os.environ.get("SUPABASE_PASSWORD")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Sign-in function
def sign_in():
    response = supabase.auth.sign_in_with_password({
        "email": SUPABASE_USERNAME,
        "password": SUPABASE_PASSWORD
    })

    if response.user.aud == 'authenticated':
        print(f"Signed in as: {response.user.email}")
    else:
        print("Error signing in to supabase")
    return response

# Sign-out function
def sign_out():
    supabase.auth.sign_out()
    print("Signed out successfully.")

def retrieve_supabase_client() -> Client:
    return(supabase)