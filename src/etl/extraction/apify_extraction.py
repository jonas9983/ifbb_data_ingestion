import os
from typing import Dict, List, Set, Tuple, Optional
from dotenv import load_dotenv
from dataclasses import dataclass, asdict
# from apify_client import ApifyClient

@dataclass
class ApifyInput:
    """ Return the input for the ApifyClient based on a list of athletes.
    This list can be obtained from Supabase for example. 
    The expected input is a list containing usernames and the output is something as follows:

    run_input = {
        "username" : [
            "natgeo",
            "https://www.instagram.com/natgeo/",
            "https://www.instagram.com/<POST_LINK>" 
        ],
        "resultsLimit": 30,
        "onlyPostsNewerThan": None,
        "skipPinnedPosts": None,
        etc,
    }
     
    There are several arguments that could be read from yaml file stored in the ./configs directory """

    username: List[str] # Can be Instagram ID (which is unique), post or full instagram profile link

class ActorDefinition:
    """Define an Actor from Apify and use the API to get the posts from the list defined from ApifyInput"""

    load_dotenv()
    APIFY_API_TOKEN = os.getenv("APIFY_API_TOKEN")

    def __init__(self, username: list, actorID: str):
        self.actorID = actorID # Unique ID from Apify website
        # self.client = ApifyClient(APIFY_API_TOKEN)

        self.apify_input = asdict(ApifyInput(username = username))

    def run_actor(self):
        result = self.client.actor(self.actorID).call(run_input = self.apify_input)

        return(result)