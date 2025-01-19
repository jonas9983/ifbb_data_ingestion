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

def prompt_gpt(img_url: str, input_gpt: list, model_id: str):
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
    Please ensure that the values found on the place list are all different from each other. The competitor's name and the place are the most important columns
    Ensure that all lists (competition_type, competitors_name, country, place) are simple, flat lists that directly match their respective values without embedding them inside a dictionary or nested structure. Guarantee that the length of each list is consistent by filling in any missing values as described above.
    """

    openai.api_key = 'sk-vGXLTIxgYyXy5ouqNHkvzkZVJ6MrVlS0PHC1nnWzjaT3BlbkFJ6jIOA923yyIEXBsQHmehrSxtHzw5dZhWcW3zii79IA'

    client = OpenAI()

    response = client.chat.completions.create(
        model = model_id, 
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

def get_sidebar_info(urls:list):
    sidebar_info = pd.DataFrame()
    for url in urls:
        display(url)
        sidebar_info_temp = pd.DataFrame()
        try:
            html = requests.get(url)
            full_html = html.text
            soup = BeautifulSoup(full_html, 'html.parser')
        except requests.exceptions.RequestException as e:
            # logging.error(f"Error fetching URL {url}: {e}")
            print(f"Error fetching URL {url}: {e}")
            continue

        try:
            # Find start date
            start_date_html = soup.find_all('abbr', class_ = 'tribe-events-start-date')
            start_date = start_date_html[0].get_text().strip() if start_date_html else None
        except (IndexError, AttributeError) as e:
            print(f"Error parsing start date for {url}: {e}")
            # logging.error(f"Error parsing date or location for {url}: {e}")
            continue

        try:
            # Find end date
            end_date_html = soup.find_all('abbr', class_ = 'tribe-events-end-date')
            end_date = end_date_html[0].get_text().strip() if end_date_html else None
        except (IndexError, AttributeError) as e:
            print(f"Error parsing end date for {url}: {e}")
            # logging.error(f"Error parsing date or location for {url}: {e}")
            continue

        try:
            location_html = soup.find_all('span', class_ = 'tribe-address')
            location = location_html[0].get_text(separator = '').strip() if location_html else None
        except (IndexError, AttributeError) as e:
            print(f"Error parsing location for {url}: {e}")
            # logging.error(f"Error parsing date or location for {url}: {e}")
            continue
            
            
        try:
            comp_type_html = soup.find_all('dd', class_ = 'tribe-events-event-categories')
            comp_type = comp_type_html[0].get_text() if comp_type_html else None
        except (IndexError, AttributeError) as e:
            print(f"Error parsing competition type for {url}: {e}")
            # logging.error(f"Error parsing date or location for {url}: {e}")
            continue

        try:
            divisions_html = soup.find_all('dd', class_ = 'tribe-events-event-division')
            divisions = divisions_html[0].text.strip().split('\n') if divisions_html else None
        except (IndexError, AttributeError) as e:
            print(f"Error parsing divisions for {url}: {e}")
            # logging.error(f"Error parsing date or location for {url}: {e}")
            continue

        try:
            division_type_html = soup.find_all('dd', class_ = 'tribe-events-event-division-type')
            division_type = division_type_html[0].text.strip().split('\n')[0] if division_type_html else None
        except (IndexError, AttributeError) as e:
            print(f"Error parsing division type for {url}: {e}")
            # logging.error(f"Error parsing date or location for {url}: {e}")
            continue

        try:
            promoter_html = soup.find_all('dd', class_ = 'tribe-organizer')
            promoter = promoter_html[0].text.strip().split('\n')[0] if promoter_html else None
        except (IndexError, AttributeError) as e:
            print(f"Error parsing promoter for {url}: {e}")
            # logging.error(f"Error parsing date or location for {url}: {e}")
            continue

        try:
            promoter_website_html = soup.find_all('dd', class_ = 'tribe-organizer-url')
            promoter_website = promoter_website_html[0].find_all('a', href = True)[0]['href'] if promoter_website_html else None
        except (IndexError, AttributeError) as e:
            print(f"Error parsing promoter_website for {url}: {e}")
            # logging.error(f"Error parsing date or location for {url}: {e}")
            continue

        sidebar_info_temp = pd.DataFrame({'start_date': start_date,
                                        'end_date': end_date,
                                        'url': url, 
                                            'location': location,
                                            'comp_type': comp_type,
                                            'divisions': divisions,
                                            'division_type': division_type,
                                            'promoter': promoter,
                                            'promoter_website': promoter_website})
        
        sidebar_info = pd.concat([sidebar_info, sidebar_info_temp])

    sidebar_df = sidebar_info.explode('divisions').reset_index(drop = True)
    
    return(sidebar_df)

def get_results_from_tables(urls: list):
    results_df = pd.DataFrame()
    for url in urls:
        print(url)
        idx = 0
        try:
            html = requests.get(url)
            full_html = html.text
            soup = BeautifulSoup(full_html, 'html.parser')
        except requests.exceptions.RequestException as e:
            print(f"Error fetching URL {url}: {e}")
            continue

        # Get date, location from the html side bar
        try:
            date_html= soup.find_all('abbr', class_ = 'tribe-events-abbr')[0]
            date = date_html['title'] if date_html else 'Unknown'
            location_html = soup.find_all('span', class_ = 'tribe-address')[0]
            location = location_html.get_text(separator = '').strip() if location_html else 'Unknown'
        except (IndexError, AttributeError) as e:
            print(f"Error parsing date or location for {url}: {e}")
            continue

        try:
            result_div = soup.find_all('div', id = 'Results')[0]
            tables_in_div= result_div.find_all('table') if result_div else []
        except AttributeError as e:
            print(f"No results found for {url}: {e}")
            continue
        while idx < len(tables_in_div):
            table = tables_in_div[idx]
            if 'id' in table.attrs:            
                if table['id'] in 'mwResultListTable_Desktop':
                    class_name = table.find_all('td', class_ = 'className')
                    class_name = [name.string.lower() for name in class_name]

                    comp_details = table.find_all('td', class_ = 'compDetails')
                    comp_details = [detail.string for detail in comp_details]
                    comp_details_grouped = [comp_details[i:i + 2] for i in range(0, len(comp_details), 2)]

                    score_values = table.find_all('td', class_ = 'scoreValue')
                    score_values = [values.string for values in score_values]
                    score_values_grouped = [score_values[i:i + 3] for i in range(0, len(score_values), 3)]

                    placing_values = table.find_all('td', class_ = 'placingValue')
                    placing_values = [values.string for values in placing_values]

                    df_placing = pd.DataFrame(placing_values, columns = ['place'])
                    df_scores = pd.DataFrame(score_values_grouped, columns = ['judging', 'finals', 'total'])
                    df_details = pd.DataFrame(comp_details_grouped, columns = ['competitors_name', 'country'])

                    results_df_temp = pd.concat([df_details, df_scores, df_placing], axis = 1)

                    results_df_temp['competition_type'] = class_name[0]
                    results_df_temp['date'] = date
                    results_df_temp['location'] = location
                    results_df_temp['url'] = url
                    results_df_temp['img_url'] = None
                    results_df_temp['competition'] = url.split('/')[-2]

                    results_df = pd.concat([results_df, results_df_temp])
            idx += 1
            
    return(results_df)
    

def main():
    comp_urls = get_comps_page(main_url = "https://www.ifbbpro.com/results")
    save_results_from_url(urls = comp_urls)


if __name__ == '__main__':
    main()
