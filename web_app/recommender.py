"""
Contains various recommondation implementations
all algorithms return a list of movieids
"""

import pandas as pd
import numpy as np
from utils import create_ramdom_user, get_prediction_nmf, nmf_recoms, get_user_id_ratings
from utils import add_user_to_R,cal_cosine_simularity, similarity_recoms, get_transpose
from read_data import DF_RATINGS, DF_MOVIES, DF_LINKS, DF_RATINGS_PIVOT, IMPUTER, MODEL, Q
from read_data import R 


def recommend_with_NMF(user, best):

    global DF_MOVIES
    user_n=get_user_id_ratings(user, DF_MOVIES)
    #get_prediction_array(Q,user,imputer,model,df_ratings_pivot):
    pred_array= get_prediction_nmf(Q,user_n,IMPUTER,MODEL,DF_RATINGS_PIVOT)
    df_best_list,df_worst_list=nmf_recoms(pred_array,user_n,DF_RATINGS_PIVOT,DF_MOVIES, best)
    return df_best_list

def recommend_with_user_similarity(user, best):
    
    global DF_MOVIES
    user_n=get_user_id_ratings(user, DF_MOVIES)
    R_updated, user_id=add_user_to_R(user_n, R)
    cos_sim_table=cal_cosine_simularity(R_updated)
    R_updated_t=get_transpose(R_updated)
    df_best_list=similarity_recoms(user_id,cos_sim_table, R_updated_t,DF_MOVIES, best)
    return df_best_list
    
