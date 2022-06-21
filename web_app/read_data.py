import pandas as pd
import pickle as pic
from get_TMDB_info import TMDBInfo
import config as config
from tmdbv3api import TMDb, Movie

DF_RATINGS = pd.read_csv('web_app/data/ratings.csv')
DF_MOVIES = pd.read_csv('web_app/data/df_movies_f.csv', index_col="movieId")
DF_LINKS = pd.read_csv('web_app/data/df_links_f.csv',index_col="movieId")

# pivot rating table
DF_RATINGS_PIVOT = DF_RATINGS.pivot(index="userId", columns="movieId", values="rating")
def filter_ratings(df_ratings_pivot):
    # Delete columns containing either 90% or more than 90% NaN Values
    perc = 90.0
    min_count =  int(((100-perc)/100)*df_ratings_pivot.shape[0] + 1)
    df_ratings_pivot = df_ratings_pivot.dropna( axis=1, 
                thresh=min_count)
    return df_ratings_pivot

DF_RATINGS_PIVOT=filter_ratings(DF_RATINGS_PIVOT)

R=DF_RATINGS_PIVOT.fillna(value=0)
# Imputer 
IMPUTER=pic.load(open("web_app/model/imputer.pk", "rb"))
# Model
MODEL=pic.load(open("web_app/model/nmf_model.pk", "rb"))
# Q Matrix
DF_Q=pd.read_csv('web_app/data/Q_df.csv', index_col=0)
Q = DF_Q.to_numpy()

# tmdb instance to connect TMDb
tmdb = TMDb()
tmdb.api_key = config.API_KEY


def get_movie_infos(rec):
    # get information from TMDB
    rec_link = rec.join(DF_LINKS)
    rec_link["tmdbId"] = rec_link["tmdbId"].astype(int)
    movie_info = pd.DataFrame(columns=["title", "overview", "image_url", "popularity",
                                       "release_date", "video_url"])
    for i in rec_link["tmdbId"]:
        t = TMDBInfo(movieId=i, api_key=tmdb.api_key, tmdb=TMDb())
        overview, image_url, title, popularity, release_date = t.get_details()
        t.get_movie_trailer()
        video_url = t.get_video_url()

        args = {"title": title, "overview": overview, "image_url": image_url, "popularity": popularity,
                "release_date": release_date, "video_url": video_url}
        movie_info = movie_info.append(args, ignore_index=True)
        
    return movie_info