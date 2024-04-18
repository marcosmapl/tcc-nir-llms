import numpy as np
import json

PROMPT_TEMPLATE1 = """You are a movie expert provide the answer for the question based on the given context. If you don't know the answer to a question, please don't share false information.

### MY WATCHED MOVIES LIST: {}.

### QUESTION: Based on my watched movies list. Tell me what features are most important to me when selecting movies (Summarize my preferences briefly)?

### ANSWER:"""

PROMPT_TEMPLATE2 = """You are a movie expert provide the answer for the question based on the given context. If you don't know the answer to a question, please don't share false information.

### MY WATCHED MOVIES LIST: {}.

### MY MOVIE PREFERENCES: {}.

### QUESTION: Create an enumerated list selecting the five most featured movies from the watched movies according to my movie preferences.

### ANSWER:"""

PROMPT_TEMPLATE3 = """You are a movie expert provide the answer for the question based on the given context. If you don't know the answer to a question, please don't share false information.

### CANDIDATE MOVIE SET: {}.

### MY WATCHED MOVIES LIST: {}.

### MY MOVIE PREFERENCES: {}.

### MY FIVE MOST FEATURED MOVIES: {}.

### QUESTION: Can you recommend 10 movies from the "Candidate movie set" similar to the "Five most featured movies" I've watched?. Use format "Recommended movie" # "Similar movie".

### ANSWER:"""

# Default encoding
DEFAULT_ENCODING = 'utf-8'

def read_json(path: str):
    with open(path) as f:
        return json.load(f)

def build_index_dict(data):
    """
    Builds a dictionary mapping movie names to their indices.

    Args:
    - data (list): List of tuples where each tuple contains user data.

    Returns:
    - dict: Dictionary mapping movie names to their indices.
    """
    movie_names = set()
    for elem in data:
        seq_list = elem[0].split(' | ')
        movie_names.update(seq_list)
    return {movie: idx for idx, movie in enumerate(movie_names)}

def build_user_similarity_matrix(data, movie_idx):
    """
    Builds a user similarity matrix based on the watched movies data.

    Args:
    - data (list): List of tuples where each tuple contains user data.
    - movie_idx (dict): Dictionary mapping movie names to their indices.

    Returns:
    - numpy.ndarray: User similarity matrix.
    """
    user_matrix = [] # user matrix
    for elem in data:    # iterate over user watched movies
        item_hot_list = np.zeros(len(movie_idx))  # create one hot user-movie vector
        for movie_name in elem[0].split(' | '):  # iterate over each movie and update one hot vector
            item_pos = movie_idx[movie_name]
            item_hot_list[item_pos] = 1
        user_matrix.append(item_hot_list)   # add user vector to user matrix
    user_matrix = np.array(user_matrix)
    return np.dot(user_matrix, user_matrix.transpose()) # compute similarity (dot product)

def build_movie_popularity_dict(data):
    """
    Builds a dictionary mapping movie names to their popularity count.

    Args:
    - data (list): List of tuples where each tuple contains user data.

    Returns:
    - dict: Dictionary mapping movie names to their popularity count.
    """
    pop_dict = {}
    for elem in data:   # iterate over dataset
        seq_list = elem[0].split(' | ')
        for movie in seq_list:  # iterate over each movie
            if movie not in pop_dict:
                pop_dict[movie] = 0
            pop_dict[movie] += 1 # increment movie popularity
    return pop_dict

def build_item_similarity_matrix(data):
    """
    Builds an item similarity matrix based on the watched movies data.

    Args:
    - data (list): List of tuples where each tuple contains user data.

    Returns:
    - numpy.ndarray: Item similarity matrix.
    """
    i_item_dict = {}
    i_item_user_dict = {}
    i_item_p = 0

    for i, elem in enumerate(data):
        seq_list = elem[0].split(' | ') # user watched movie list
        for movie in seq_list:
            if movie not in i_item_user_dict:
                item_hot_list = np.zeros(len(data))
                i_item_user_dict[movie] = item_hot_list
                i_item_dict[movie] = i_item_p
                i_item_p += 1
            i_item_user_dict[movie][i] += 1

    item_matrix = np.array([x for x in i_item_user_dict.values()])
    return np.dot(item_matrix, item_matrix.transpose())

def sort_user_filtering_items(data, watched_movies, user_similarity_array, num_u, num_i):
    """
    Sorts and filters items based on user similarity.

    Args:
    - data (list): List of tuples where each tuple contains user data.
    - watched_movies (set): Set of watched movie names.
    - user_similarity_matrix (numpy.ndarray): User similarity matrix.
    - num_u (int): Number of users to consider.
    - num_i (int): Number of items to recommend.

    Returns:
    - list: List of recommended movie names.
    """
    candidate_movies_dict = {}
    sorted_us = sorted(list(enumerate(user_similarity_array)), key=lambda x: x[-1], reverse=True)[:num_u]
    dvd = sum([e[-1] for e in sorted_us])
    for us_i, us_v in sorted_us:
        us_w = us_v * 1.0 / dvd
        us_elem = data[us_i]
        us_seq_list = us_elem[0].split(' | ')
        for us_m in us_seq_list:
            if us_m not in watched_movies:
                if us_m not in candidate_movies_dict:
                    candidate_movies_dict[us_m] = 0.
                candidate_movies_dict[us_m] += us_w
    candidate_pairs = list(sorted(candidate_movies_dict.items(), key=lambda x:x[-1], reverse=True))
    candidate_items = [e[0] for e in candidate_pairs][:num_i]
    return candidate_items

def get_candidate_ids_list(data, id_list, user_matrix_sim, num_u, num_i):
    cand_ids = []
    for i in id_list:
        watched_movies = data[i][0].split(' | ')
        candidate_items = sort_user_filtering_items(data, watched_movies, user_matrix_sim[i], num_u, num_i)
        if data[i][-1] in candidate_items:
            cand_ids.append(i)
    return cand_ids
