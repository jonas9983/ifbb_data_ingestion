from openai import OpenAI
import openai
import os
from os.path import join, dirname
from dotenv import load_dotenv
import json

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

openai.api_key = os.environ.get("OPENAI_KEY")

client = OpenAI()

def get_prompt_content():

    system_content = {"role": "system", "content": """I am a chatbot to analyze images coming from the IFBB Bodybuilding competitions"""}

    user_content_prompt = """Please analyze this image and return a structured JSON with the following key-value pairs where all the lists have the same length so that I can create a pandas DataFrame from it. If any information (like a competitor's country or total points) is missing, use None for unknown places to maintain the same list length across all keys: place: A flat list of the places each competitor earned (use None for missing data). judging: A flat list of the judging score each competitor earned (use None for missing data). finals: A flat list of the finals score each competitor earned (use None for missing data).  total: A flat list of the total points of each competitor (use None for missing data). competitors_name: A flat list of competitors' names. country: A flat list of the competitors' respective countries (use None for missing data). competition_type: A flat list of the type of competition (e.g., "Men's Classic Physique - Open"). Make sure the length of this list is the same as the other lists in the response dictionary """

    user_content = { "role": "user", "content": user_content_prompt}

    supabase_parent_path = 'https://qesnrciwmhxfhdaojwwo.supabase.co/storage/v1/object/public/resultImages/2024'

    return(system_content, user_content_prompt,user_content, supabase_parent_path)

def retrieve_client():
      return(client)