import requests

def call_api_wayback_machine(year: int, call_url: str = None):
    # Define the base API URL
    api_url = "http://web.archive.org/cdx/search/cdx"

    if call_url is None:
        if year <= 2015:
            call_url = f'http://www.ifbbpro.com/wp-content/uploads/image/{year}/results/*'
        else:
            call_url = f'http://www.ifbbpro.com/wp-content/uploads/{year}/*'
    else:
        call_url = f'{call_url}/{year}*'
    
    print(call_url)

    # Parameters for the API request
    params = {
        'url': call_url,  # Target URL
        'output': 'json',  # Request JSON output
        'limit': 2000,      # Limit the number of results
        'filter': 'statuscode:200'  # Filter only valid HTTP 200 responses
    }

    response = requests.get(api_url, params=params)
    all_urls = []

    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()
        
        # The first entry is a header; skip it
        snapshots = data[1:]
        
        # Extract and print snapshot details
        for snapshot in snapshots:
            timestamp = snapshot[1]  # Timestamp of the snapshot
            archive_url = f"http://web.archive.org/web/{timestamp}/{snapshot[2]}"  # Archived URL
            all_urls.append(archive_url)
        print("Retrieved urls from Wayback Machine API")
        
        return(all_urls)
    else:
        print(f"Failed to retrieve data. Status code: {response.status_code}")
        return(None)
