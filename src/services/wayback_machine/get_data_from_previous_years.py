import src.etl.extraction.wayback_machine_extraction as wayback_machine_extraction

import os
from urllib.request import urlretrieve

from IPython.display import display

def download_image(path: str, url_list: list):

    if not os.path.exists(path):
        os.makedirs(path)
        imgs_not_in_folder = url_list
    else:
        imgs_in_folder = os.listdir(path)
        imgs_not_in_folder = [url for url in url_list if f"{url.split('/')[-2]}_{url.split('/')[-1]}" not in imgs_in_folder]

    while len(imgs_not_in_folder) != 0:
        for url in imgs_not_in_folder:
            print(url)
            filename = f"{url.split('/')[-2]}_{url.split('/')[-1]}"
            outpath = os.path.join(path, filename)
            new_url = f"{url.split('/http')[0]}if_/http{url.split('/http')[1]}"
            try:
                urlretrieve(new_url, outpath)
            except Exception as e:
                print(f'Error {e}')
                continue
        
        imgs_in_folder = os.listdir(path)
        imgs_not_in_folder = [url for url in url_list if f"{url.split('/')[-2]}_{url.split('/')[-1]}" not in imgs_in_folder]


def main(years_list: list, path: str):
    for year in years_list:
        print(year)
        all_urls = wayback_machine_extraction.call_api_wayback_machine(year = year)

        if all_urls is not None:
            download_image(path = path, url_list= all_urls)


if __name__ == '__main__':
    years_list = [2022]
    main(years_list= years_list)
