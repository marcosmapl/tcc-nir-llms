import json
import numpy as np
import os
import pandas as pd
import pickle
import time
import torch

from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.model_selection import StratifiedKFold
from transformers import BitsAndBytesConfig, TrainingArguments, AutoModelForCausalLM, AutoTokenizer, Trainer
from trl import SFTTrainer

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

def create_bnb_config(args):
    return BitsAndBytesConfig(
        load_in_4bit=args.use_4bit,
        bnb_4bit_quant_type=args.b4b_qtype,
        bnb_4bit_compute_dtype=getattr(torch, args.b4b_cdtype),
        bnb_4bit_use_double_quant=args.nested_quantization,
    )

def create_peft_config(args):
    return LoraConfig(
        lora_alpha=args.lora_a,
        lora_dropout=args.lora_d,
        r=args.lora_r,
        bias='none',
        task_type='CAUSAL_LM',
    )

def create_training_arguments(model_name: str, args):
    os.makedirs(f'../train', exist_ok=True)
    return TrainingArguments(
        output_dir=f'../train/{model_name}',
        num_train_epochs=args.nte,
        per_device_train_batch_size=args.pdtbs,
        gradient_accumulation_steps=args.ga_steps,
        optim=args.optm,
        save_steps=args.sv_steps,
        save_total_limit=args.sv_ttl,
        logging_steps=args.lg_steps,
        learning_rate=args.lr,
        weight_decay=args.wd,
        fp16=args.fp16,
        bf16=args.bf16,
        max_grad_norm=args.max_gnorm,
        max_steps=-1,
        warmup_ratio=args.wm_ratio,
        group_by_length=args.group_batches,
        lr_scheduler_type=args.lr_scheduler,
        # report_to="wandb" # comment this out if you are not using wandb
    )

def create_dataset(model_name: str, args, cand_ids):
    os.makedirs(f'../results', exist_ok=True)
    with open(f'../results/ml100k-zs-nir-su{args.nsu}-ci{args.nci}-{model_name}.pkl', 'rb') as fp:
        results = pickle.load(fp)

    data = []
    for i in cand_ids:
        input_text = results[i]['input_3'].split('### QUESTION:')[0].strip()
        input_text = input_text + '\n\n### QUESTION: Can you recommend a movie from the "Candidate movie set" similar to the "Five most featured movie"?'
        input_text = input_text + '\n\n###ANSWER: ' + results[i]['ground_truth'] + ' [end-gen]'
        data.append((i, results[i]['input_3'], results[i]['ground_truth'], input_text))

    return Dataset.from_pandas(pd.DataFrame(data, columns=['id', 'prompt', 'ground_truth', 'input_text']))

def train_model_cv(model_name, ds, args):
    # create fold object
    skf = StratifiedKFold(n_splits=args.kfold, random_state=args.seed, shuffle=True)

    results = dict()
    results['namespace'] = args.model
    results['model_name'] = model_name
    results['args'] = args
    results['results'] = {}
    for k, (train_idx, test_idx) in enumerate(skf.split(ds['input_text'], ds['ground_truth'])):
        print(f'Fold {k}, Tran size {len(train_idx)}, Test size {len(test_idx)}')
        # load base model on GPU
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=create_bnb_config(args),
            device_map=args.device
        )
        # reduce memory usage during fine-tuning
        model.gradient_checkpointing_enable()
        # re-enable for inference to speed up predictions for similar inputs
        model.config.use_cache = False
        # model.config.pretraining_tp = 1

        # load model tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.add_special_tokens({"additional_special_tokens" : ['[end-gen]']})

        # resize model tokens size to match with tokenizer
        model.resize_token_embeddings(len(tokenizer))

        # prepare model using peft function
        model = prepare_model_for_kbit_training(model)

        # create trainer object
        trainer = SFTTrainer(
            model=model,
            train_dataset=ds.select(train_idx),
            peft_config=create_peft_config(args),
            dataset_text_field='input_text',
            max_seq_length=args.max_seq_len,
            tokenizer=tokenizer,
            args=create_training_arguments(model_name, args),
            packing=args.packing,
        )

        # MODEL FINE TUNING
        if args.train_model:
            train_result = trainer.train()            
            # save fine tuned model to disk
            if args.save_tuned:
                os.makedirs(f"../models", exist_ok=True)
                trainer.model.save_pretrained(f"../models/ml100k-su{args.nsu}-ci{args.nci}-{model_name}")
                # trainer.log_metrics('train', train_result.metrics)
                trainer.save_metrics('train', train_result.metrics)
                trainer.save_state()
                print(f'TRAINING METRICS:\n\t{train_result.metrics}')

        lora_config = LoraConfig.from_pretrained(f"../models/ml100k-su{args.nsu}-ci{args.nci}-{model_name}")

        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=create_bnb_config(args),
            device_map=args.device
        )

        tuned_model = get_peft_model(model, lora_config)
        tuned_model.print_trainable_parameters()
        tuned_model.eval()

        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.add_special_tokens({"additional_special_tokens" : ['[end-gen]']})

        # resize model embedding
        tuned_model.resize_token_embeddings(len(tokenizer))

        fold_key = f'{k}_fold'
        fold_results = dict()
        fold_results['start_time'] = time.time()
        fold_results['results'] = []
        print(f"START TIME: {fold_results['start_time']}")

        for idx in test_idx.tolist():
            sample_result = dict()
            # results_data[f'{k}_fold'][idx]['id'] = ds[idx]['id']
            sample_result['input'] = ds[idx]['prompt']
            sample_result['ground_truth'] = ds[idx]['ground_truth']
            sample_result['id'] = ds[idx]['id']
            
            encoding = tokenizer(ds[idx]['prompt'], return_tensors="pt").to("cuda:0")
            with torch.inference_mode():
                outputs = tuned_model.generate(
                    input_ids=encoding.input_ids, 
                    attention_mask=encoding.attention_mask,
                    max_new_tokens=args.mtk, 
                    eos_token_id=[tokenizer.get_vocab()["[end-gen]"]]
                )
                # outputs = tuned_model.generate(**encoding, max_new_tokens=args.mtk, eos_token_id=[tokenizer.get_vocab()["[end-gen]"]])
            generated = tokenizer.decode(outputs[0], skip_special_tokens=False)
            #   print(generated)
            sample_result['output'] = generated.split('[end-gen]')[0]
            # Check if the ground truth movie is in the final predictions.
            sample_result['hit'] = sample_result['ground_truth'].lower() in sample_result['output'].lower()
            print(f'{fold_key}, sample nº {idx}')
            print(f"\tOUTPUT: {sample_result['output'] }\n")
            print(f"\tGROUND TRUTH: {sample_result['ground_truth']}\n\n")
            fold_results['results'].append(sample_result)

        fold_results['end_time'] = time.time()
        fold_results['total_time'] = fold_results['end_time'] - fold_results['start_time']
        print(f"END TIME: {fold_results['total_time']}")
        print(f"Total execution time: {(fold_results['total_time'] / 60):.2f} min")

        results['results'][fold_key] = fold_results 
    return results
