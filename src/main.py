import requests
from IPython.display import display
from bs4 import BeautifulSoup
import os
import openai
from openai import OpenAI
import pandas as pd
import json
import logging

# Set up logging configuration
logging.basicConfig(filename='./logging.log', 
                    level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def prompt_gpt(img_url: str, input_gpt: list):
    logging.info(f"Prompting GPT for competition {img_url}")
    prompt = prompt = f"""
    Please analyze this image and return a structured JSON with the following key-value pairs where all the lists have the same length so that I can create a pandas DataFrame from it. If any information (like a competitor's country or total points) is missing, use a placeholder such as an empty string "", 0 for numbers, or - for unknown places to maintain the same list length across all keys:

    place: A flat list of the places each competitor earned (use "-" for missing data).
    judging: A flat list of the judging score each competitor earned (use 0 for missing data).
    finals: A flat list of the finals score each competitor earned (use 0 for missing data). 
    total: A flat list of the total points of each competitor (use 0 for missing data).

    competitors_name: A flat list of competitors' names.
    country: A flat list of the competitors' respective countries (use "" for missing data).
    competition_type: A flat list of the type of competition (e.g., "Men's Classic Physique - Open"). Make sure the length of this list is the same as the other lists in the response dictionary
    
    Important Notes:

    Please ignore the IFBB Professional League name that usually appears 
    Please ensure that the values found on the place list are all different from each other. The competitor's name and the place are the most import columns
    Ensure that all lists (competition_type, competitors_name, country, place) are simple, flat lists that directly match their respective values without embedding them inside a dictionary or nested structure. Guarantee that the length of each list is consistent by filling in any missing values as described above.
    """

    openai.api_key = 'sk-vGXLTIxgYyXy5ouqNHkvzkZVJ6MrVlS0PHC1nnWzjaT3BlbkFJ6jIOA923yyIEXBsQHmehrSxtHzw5dZhWcW3zii79IA'

    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-4o", 
        response_format={"type": "json_object"},
        messages=[
            {
            "role": "user",
            
            "content": [
                {"type": "text", "text": prompt},
                {
                "type": "image_url",
                "image_url": {
                    "url": f"{img_url}",
                },
                },
                
            ],
            }
        ],
    )

    # display(json.loads(response.choices[0].message.content))

    response_df = pd.DataFrame(json.loads(response.choices[0].message.content))
    response_df.loc[:, 'competition'] = input_gpt[2].split('/')[-2]
    response_df.loc[:, 'location'] = input_gpt[1]
    response_df.loc[:, 'date'] = input_gpt[0]
    response_df.loc[:, 'url'] = input_gpt[2]
    response_df.loc[:, 'img_url'] = img_url # This column will be removed later, it's just an identifier


    return(response_df)

def save_results_from_url(urls: list):
    """_summary_

    Args:
        urls (list): _description_
    """
    results_path = 'results_df'

    if os.path.exists(f'{results_path}.csv'):
        results_df = pd.read_csv(f'{results_path}.csv')
    else:
        results_df = pd.DataFrame()

    for url in urls:
        idx = 0
        try:
            html = requests.get(url)
            full_html = html.text
            soup = BeautifulSoup(full_html, 'html.parser')
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching URL {url}: {e}")
            continue

        # Get date, location from the html side bar
        try:
            date_html= soup.find_all('abbr', class_ = 'tribe-events-abbr')[0]
            date = date_html['title'] if date_html else 'Unknown'
            location_html = soup.find_all('span', class_ = 'tribe-address')[0]
            location = location_html.get_text(separator = '').strip() if location_html else 'Unknown'
        except (IndexError, AttributeError) as e:
            logging.error(f"Error parsing date or location for {url}: {e}")
            continue

        try:
            result_div = soup.find_all('div', id = 'Results')[0]
            imgs_in_div= result_div.find_all('img') if result_div else []
        except AttributeError as e:
            logging.error(f"No results found for {url}: {e}")
            continue

        while idx < len(imgs_in_div): # Since img_file may have more than one image assigned
            img_file = imgs_in_div[idx].attrs.get('data-src', '')
            if 'Contest-Photos' not in img_file and 'IFBB_Pro_League_Logo' not in img_file:
                if not results_df.empty and (results_df['url'] == img_file).any():
                    idx +=1
                    continue
                try:
                    results_df = pd.concat([prompt_gpt(img_url=img_file, input_gpt=[date, location,url]), results_df])
                    results_df.to_csv(f'{results_path}.csv', index = False)
                    logging.info(f"{img_file} saved.")
                except Exception as e:
                    logging.error(f"Error processing {img_file}: {e}")

            idx += 1

    return(results_df)





def get_comps_page(main_url: str):
    """_summary_

    Args:
        url: _description_
    """
    i = 1
    competition_links = []
    while True:
        if i == 1:
            main_url_extra = main_url
        else:
            main_url_extra = f'{main_url}/page/{i}/'

        url = requests.get(main_url_extra)
        full_html = url.text
        soup = BeautifulSoup(full_html, 'html.parser')
        links =[a['href'] for a in soup.find_all('a', class_ = 'fusion-column-anchor', href = True)]
        competition_links.extend(links)
        
        if len(links) == 0:
            return(competition_links)
        i = i + 1

def main():
    comp_urls = get_comps_page(main_url = "https://www.ifbbpro.com/results")
    save_results_from_url(urls = comp_urls)


if __name__ == '__main__':
    main()
