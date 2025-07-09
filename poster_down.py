import os
import requests

def download_posters(indexes, df, api_key):
    os.makedirs("posters", exist_ok=True)
    posters = []
    for i in indexes:
        title = df.loc[i]['name']
        imdb_id = df.loc[i]['movie_id']

        api_url = api_url = f"http://www.omdbapi.com/?t={title}&apikey={api_key}"
        response = requests.get(api_url)
        data = response.json()

        if data.get("Response") == "True" and "Poster" in data:
            poster_url = data["Poster"]
            img_data = requests.get(poster_url).content
            file_path = f"posters/{title}.jpg"
            with open(file_path, "wb") as handler:
                handler.write(img_data)
            posters.append((title, file_path))
        else:
            posters.append((title, None))
    return posters
