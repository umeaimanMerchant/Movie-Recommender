
# import library
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import numpy.ma as ma
from numpy import genfromtxt
from collections import defaultdict
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import tabulate
from recsysNN_utils import *
from tensorflow.keras.models import load_model
pd.set_option("display.precision", 1)
import pickle5 as pickle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# load the model
model = load_model("movie_recommender")

# load scaler
scalerItem = pickle.load(open('scalers_files/scalerItem.pickle', 'rb'))
scalerUser = pickle.load(open('scalers_files/scalerUser.pickle', 'rb'))
scaler = pickle.load(open('scalers_files/scaler.pickle', 'rb'))

# load data
item_train, user_train, y_train, item_features, user_features, item_vecs, movie_dict, user_to_genre = load_data()

# config
u_s = 3  # start of columns to use in training, user
i_s = 1  # start of columns to use in training, items
scaledata =True

# formating
def gen_user_vecs(user_vec, num_items):
    """ given a user vector return:
        user predict maxtrix to match the size of item_vecs """
    user_vecs = np.tile(user_vec, (num_items, 1))
    return(user_vecs)

# predict on  everything, filter on print/use
def predict_uservec(user_vecs, item_vecs, model, u_s, i_s, scaler, ScalerUser, ScalerItem, scaledata=True):
    """ given a user vector, does the prediction on all movies in item_vecs returns
        an array predictions sorted by predicted rating,
        arrays of user and item, sorted by predicted rating sorting index
    """
    
    # if data is scaled, scale new user before prediction
    if scaledata:
        scaled_user_vecs = ScalerUser.transform(user_vecs)
        scaled_item_vecs = ScalerItem.transform(item_vecs)
        y_p = model.predict([scaled_user_vecs[:, u_s:], scaled_item_vecs[:, i_s:]])
    else:
        y_p = model.predict([user_vecs[:, u_s:], item_vecs[:, i_s:]])
        
    y_pu = scaler.inverse_transform(y_p)
    
    # check if prediction is positive
    if np.any(y_pu < 0) : 
        print("Error, expected all positive predictions")
        
    # get largest rating first
    sorted_index = np.argsort(-y_pu,axis=0).reshape(-1).tolist()  
    sorted_ypu   = y_pu[sorted_index]
    sorted_items = item_vecs[sorted_index]
    sorted_user  = user_vecs[sorted_index]
    return(sorted_index, sorted_ypu, sorted_items, sorted_user)


def send_pred(y_p, user, item, movie_dict, maxcount=10):
    """ print results of prediction of a new user. inputs are expected to be in
        sorted order, unscaled. """
    count = 0
    # note all movie id in dict
    movies_listed = defaultdict(int)
    # disp = [["y_p", "movie id", "rating ave", "title", "genres"]]
    movies = {}
    for i in range(0, y_p.shape[0]):
        if count == maxcount:
            break
        count += 1
        
        # get movie id from item_train or item
        movie_id = item[i, 0].astype(int)
        
        if movie_id in movies_listed:
            continue
        movies_listed[movie_id] = 1
        movies[movie_dict[movie_id]['title']] = movie_dict[movie_id]['genres']
        # disp.append([y_p[i, 0], item[i, 0].astype(int), item[i, 2].astype(float),
        #             movie_dict[movie_id]['title'], movie_dict[movie_id]['genres']])

    # table = tabulate.tabulate(disp, tablefmt='html',headers="firstrow")
    # return(movies)
    return jsonify(movies)


# create app
app = Flask(__name__)
CORS(app) 


@app.route('/model', methods=['POST'])
def recommender():
    request_data = request.get_json(force=True)
    
    # Extract values for each genre
    new_action = request_data.get('action')
    new_adventure = request_data.get('adventure')
    new_animation = request_data.get('animation')
    new_childrens = request_data.get('childrens')
    new_comedy = request_data.get('comedy')
    new_crime = request_data.get('crime')
    new_documentary = request_data.get('documentary')
    new_drama = request_data.get('drama')
    new_fantasy = request_data.get('fantasy')
    new_horror = request_data.get('horror')
    new_mystery = request_data.get('mystery')
    new_romance = request_data.get('romance')
    new_scifi = request_data.get('scifi')
    new_thriller = request_data.get('thriller')

    # Print values (optional)
    print("Action:", new_action)
    print("Adventure:", new_adventure)
    print("Animation:", new_animation)
    print("Children's:", new_childrens)
    print("Comedy:", new_comedy)
    print("Crime:", new_crime)
    print("Documentary:", new_documentary)
    print("Drama:", new_drama)
    print("Fantasy:", new_fantasy)
    print("Horror:", new_horror)
    print("Mystery:", new_mystery)
    print("Romance:", new_romance)
    print("Sci-Fi:", new_scifi)
    print("Thriller:", new_thriller)

    # we don't consider it for prediction
    new_user_id = 5000  
    new_rating_ave = 1.0
    new_rating_count = 3

    user_vec = np.array([[new_user_id, new_rating_count, new_rating_ave,
                        new_action, new_adventure, new_animation, new_childrens,
                        new_comedy, new_crime, new_documentary,
                        new_drama, new_fantasy, new_horror, new_mystery,
                        new_romance, new_scifi, new_thriller]])


    # generate and replicate the user vector to match the number movies in the data set.
    user_vecs = gen_user_vecs(user_vec,len(item_vecs))

    # scale the vectors and make predictions for all movies. Return results sorted by rating.
    sorted_index, sorted_ypu, sorted_items, sorted_user = predict_uservec(user_vecs,  item_vecs, model, u_s, i_s, 
                                                                        scaler, scalerUser, scalerItem, scaledata=scaledata)

    dict_movie = send_pred(sorted_ypu, sorted_user, sorted_items, movie_dict, maxcount = 10)
    print(dict_movie)

    return dict_movie


if __name__=='__main__':
    app.run(port=8000, debug=True)
