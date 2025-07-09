import os
import requests

def download_posters(indexes, df, api_key):
    os.makedirs("posters", exist_ok=True)
    posters = []
    for i in indexes:
        title = df.loc[i]['name']
        imdb_id = df.loc[i]['movie_id']

        api_url = f"http://www.omdbapi.com/?t={title}&apikey={api_key}"
        response = requests.get(api_url)
        data = response.json()

        if data.get("Response") == "True" and "Poster" in data:
            poster_url = data["Poster"]
            rating = data.get("imdbRating", "N/A")
            genre = data.get("Genre", "N/A")
            year = data.get("Year", "N/A")

            info = {
                "imdbRating": rating,
                "Genre": genre,
                "Year": year
            }

            img_data = requests.get(poster_url).content
            safe_title = title.replace("/", "-").replace(" ", "_")
            file_path = f"posters/{safe_title}.jpg"

            with open(file_path, "wb") as handler:
                handler.write(img_data)

            posters.append((title, file_path, info))
        else:
            posters.append((title, None, {
                "imdbRating": "N/A",
                "Genre": "N/A",
                "Year": "N/A"
            }))
    return posters
