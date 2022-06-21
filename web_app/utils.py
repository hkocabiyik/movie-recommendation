
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import random
import pickle as pic
from sklearn.metrics.pairwise import cosine_similarity

def create_ramdom_user(num_ratings, df_ratings_pivot):
    #
    random_dic={}
    for i in range(num_ratings):
        random_r=random.randint(1,5)
        movie_index=random.randint(0,len(df_ratings_pivot.columns)-1)
        random_dic[df_ratings_pivot.columns[movie_index]] =random_r
    return random_dic

def get_user_arr(user, df_ratings_pivot, for_what):
    #
    arr = np.empty((1,len(df_ratings_pivot.columns)))
    if(for_what=="NMF"):
        arr[:] = np.NaN    
    else:
        arr[:] = 0    
    for key, value in user.items():
        # find the index no
        index_no = df_ratings_pivot.columns.get_loc(key)
        arr[0][index_no]=value
    return arr   

def get_prediction_nmf(Q,user,imputer,model,df_ratings_pivot):
    #
    arr=get_user_arr(user, df_ratings_pivot, "NMF")
    user_clean = imputer.transform(arr)
    user_P = model.transform(user_clean) # how strongly our user likes the n "genres"
    #new user R - reconstruct R but for this new user only
    user_R = np.dot(user_P,Q)
    return user_R[0]

def nmf_recoms(pred_array,user, df_ratings_pivot,df_movies, best, worst):
    #
    recom_df = pd.DataFrame({'predicted_ratings':pred_array}, index = df_ratings_pivot.columns)
    recom_df=recom_df.drop(index=user.keys())
    # Default left join
    recom_final = recom_df.join(df_movies)
    recom_final.sort_values(by = 'predicted_ratings', ascending= False, inplace=True)
    recom_best=recom_final.head(best)
    recom_worst=recom_final.tail(worst)
    return recom_best, recom_worst  

def get_user_id_ratings(user, df_movies):
    #
    user_n={}
    for key, value in user.items():
        user_n[np.asscalar(df_movies[df_movies["title"]==key].index.values)]=value
    return user_n 


def cal_cosine_simularity(R):
    # Returns numpy array:
    cosine_similarity(R)
    # We can turn this into a dataframe:
    cos_sim_table = pd.DataFrame(cosine_similarity(R), index= R.index, columns=R.index)
    return cos_sim_table

def add_user_to_R(new_user, R):
    arr=get_user_arr(new_user, R, "similarity")
    user_id=R.index.max()+1
    # new user dataframe
    df_new_user=pd.DataFrame(arr,index=[user_id], columns = R.columns)
    R=R.append(df_new_user)
    return R, user_id

def get_transpose(R):
    return R.T

def similarity_recoms(active_user,cos_sim_table, R_t,df_movies, best, nn=3):
    # create a list of unseen movies for this user
    unseen_movies = list(R_t.index[R_t[active_user]==0])
    cos_sim_table[active_user].sort_values(ascending=False)
    # Create a list of top nn similar user (nearest neighbours)
    neighbours= list(cos_sim_table[active_user].sort_values(ascending=False).index[1:(nn+1)])
    # create the recommendation (predicted/rated movie)
    predicted_ratings_movies = []
    
    for movie in unseen_movies:
        # we check the users who watched the movie
        people_who_have_seen_the_movie = list(R_t.columns[R_t.loc[movie] > 0])
    
        num = 0
        den = 0
        for user in neighbours:
        # if this person has seen the movie
            if user in people_who_have_seen_the_movie:
            #  we want extract the ratings and similarities
                rating = R_t.loc[movie,user]
                similarity = cos_sim_table.loc[active_user,user]
                num = num + rating*similarity
                den = den + similarity
        if(den!=0):
            predicted_ratings = num/den
            predicted_ratings_movies.append([predicted_ratings,movie])
        
    # create df pred
    df_pred = pd.DataFrame(predicted_ratings_movies, columns=['rating', 'movieId'])
    df_pred.set_index("movieId", inplace=True)
    df_pred = df_pred.join(df_movies)
    df_pred=df_pred.sort_values(by=["rating"],ascending=False)
    recom_best=df_pred.head(best)
    return recom_best
