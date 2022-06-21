import re
from recommender import recommend_with_NMF,recommend_with_user_similarity
from flask import Flask,render_template,request
from read_data import DF_MOVIES, get_movie_infos

import pandas as pd

# construct our flask instance, pass name of module
app = Flask(__name__)

# route decorator for mapping urls to functions
@app.route('/')
def welcome():
    # renders the html page as the output of this function
    return render_template('index.html',name="Stationary Srirachas 🌶", movies=DF_MOVIES['title'].tolist())
    # 'movies' variable is passed from python file to the html file for accessing it inside the html file

@app.route('/recommend')
def recommend():
    #read user input from url/webpage
    print(request.args)
    titles = request.args.getlist('title') # taking lists of titles only from user input
    ratings = request.args.getlist('ratings') # taking lists of ratings only from user input
    
    print(titles,ratings)
    # converting lists of titles and ratings into dict to pass to our recommender model
    user = dict(zip(titles,ratings)) 
    hs = open("web_app/data/sample.txt","a")
    hs.write('\n')
    hs.write('The user\n')
    hs.write(str(user))
    hs.write('\n')
    hs.write('\n')
    hs.close() 
    #rec = recommend_with_NMF(user, best=5)
    rec = recommend_with_user_similarity(user, best=7)
    
    rec_titles=rec["title"].to_list()
    hs = open("web_app/data/sample.txt","a")
    hs.write('\n')
    hs.write('The Pred\n')
    print(' '.join(rec_titles))
    hs.write(' '.join(rec_titles))
    hs.close() 
    
    # display only the titles
    recs = rec["title"]
    movie_info=get_movie_infos(rec)
    return render_template('recommender.html', movies=recs, movie_info=movie_info)
    
    #renders the html page as the output of this function
    #return  render_template('recommender.html',movie_ids=movie_ids) 
    # 'movie_ids' variable is passed from python file to the html file for accessing it inside the html file

# Runs the app (main module)
if __name__=='__main__':
    app.run(debug=True,port=5000)




