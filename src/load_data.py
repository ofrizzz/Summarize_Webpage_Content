import json
import pandas as pd

def load_summaries_df():
    with open('data/summaries_1k.json', 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    data = json_data['data']
    url_count = json_data['url_count']

    webpages_content = pd.json_normalize(data)

    return webpages_content
