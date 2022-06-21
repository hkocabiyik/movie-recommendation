from recommender import recommend_random
import pandas as pd
from utils import create_ramdom_user, get_prediction_array, get_list_of_recommendation,get_user_id_ratings
from utils import df_ratings, df_movies, df_links, df_ratings_pivot, l_imputer, l_model, l_Q
from recommender import recommend_with_NMF
# example input of web application
#r_user=create_ramdom_user(3, df_ratings_pivot)

print('\n\n')
print('Random User with Movie IDs')
#print(r_user)

### Terminal recommender:
print('>>>> Here are some movie recommendations for you<<<<')
print('')
print('Recommended  movies for a random user')
dic={"Shawshank Redemption, The (1994)":4, "Godfather, The (1972)":5, "Fight Club (1999)":5,"Usual Suspects, The (1995)": 3}
user_n=get_user_id_ratings(dic, df_movies)
print(user_n)
pred_array= get_prediction_array(l_Q,user_n,l_imputer,l_model,df_ratings_pivot)
best_list,worst_list=get_list_of_recommendation(pred_array,user_n,df_ratings_pivot,df_movies, best=5)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(best_list)
    
user_r=recommend_with_NMF(dic,5)
print(user_r)



