from bs4 import BeautifulSoup
import pandas as pd
import requests

def get_full_schedule_df(soup):

    table_comp_names = ['men-open-bodybuilding','men-212-bodybuilding', 'men-classic-physique', 
                    'men-physique', 'men-wheelchair', 'women-bodybuilding', 'women-fitness', 
                    'women-figure', 'women-bikini', 'women-physique', 'women-wellness']
    
    comp_types = ["Men\'s Bodybuilding", "Men\'s 212 Bodybuilding", "Men\'s Classic Physique", 
                "Men\'s Physique", "Men\'s Wheelchair", "Women\'s Bodybuilding", 
                "Women\'s Fitness", "Women\'s Figure", "Women\'s Bikini", "Women\'s Physique",
                "Women\'s Wellness"]
    
    full_schedule_df = pd.DataFrame()
    comp_data_list = []

    for i in range(len(comp_types)):
        tab_content_specific_table = soup.find_all('table', id = f'tablepress-2024-{table_comp_names[i]}')[0]

        table_body = tab_content_specific_table.find_all('tbody')[0]

        for row in table_body.find_all('tr'):
            columns = row.find_all("td")
                
            
            row_data = [col.get_text(strip=True) for col in columns]
            try:
                if columns[0].find('a') is not None:
                    row_data.append(columns[0].find('a')['href'])
                else:
                    row_data.append(None)
            except Exception as e:
                print(f"Error {e}")
                print(row_data)
                break

            try:
                if columns[0].find('div') is not None:
                    row_data.append(columns[0].find('div').get_text(strip = True))
                else:
                    row_data.append(None)
            except Exception as e:
                print(f"Error {e}")
                print(row_data)
                break

            comp_data_list.append(row_data)
        comp_types_df = pd.DataFrame(comp_data_list)
        comp_types_df.columns = ['competition', 'date', 'location', 'promoter', 'comp_url', 'competition_subtype']
        comp_types_df['competition_type'] = comp_types[i]

        full_schedule_df = pd.concat([full_schedule_df,comp_types_df])

    return(full_schedule_df.reset_index(drop = True))


def parse_html(url: str):
    """Parse html from url to access its elements
    Args:
        url (str): URL for the PRO schedule website
    """
    try:
        html = requests.get(url)
        full_html = html.text
        soup = BeautifulSoup(full_html, 'html.parser')
    except Exception as e:
        print(f"Error {e}")

    return(soup)

def main(url: str):
    soup = parse_html(url = url)

    full_schedule_df = get_full_schedule_df(soup= soup)

    # This data should eventually be compared with what is already on supabase.

    return(full_schedule_df)


if __name__ == '__main__':
    url = 'https://www.ifbbpro.com/schedule/'
    main(url = url)