import argparse
import time
import json
import random
import transformers
import torch
import pickle
import util

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from huggingface_hub import login

# Setup the argument parser with descriptions for each option.
parser = argparse.ArgumentParser(description="Script for generating recommendations using a HuggingFace model.")
parser.add_argument('-mdl', '--model', help="HuggingFace model name", type=str, default=None)
parser.add_argument('-tk' '--token', help="HuggingFace authentication token", type=str, default=None)
parser.add_argument("-ll", '-lenlimit', help="Length limit", type=int, default=8)
parser.add_argument('-ncd', '--fitems', help="Number of similar candidate items for recommendation", type=int, default=19)
parser.add_argument('-nuf', '--fusers', help="Number of users to consider for filtering recommendations", type=int, default=12)
parser.add_argument('-seed', help="Random seed for reproducibility", type=int, default=0)
parser.add_argument('-qtp', '--qtype', help="Quantization type for BitsAndBytes configuration", type=str, default='nf4')
parser.add_argument('--load-4bit', dest='load_4bit', action='store_true', help="Enable loading models in 4-bit quantization")
parser.add_argument('--no-load-4bit', dest='load_4bit', action='store_false', help="Disable loading models in 4-bit quantization")
parser.set_defaults(load_4bit=True)
parser.add_argument('--doubleq', dest='doubleq', action='store_true', help="Enable double quantization")
parser.add_argument('--no-doubleq', dest='doubleq', action='store_false', help="Disable double quantization")
parser.set_defaults(doubleq=True)
parser.add_argument('-dmp', '--devmap', help="Device map policy for PyTorch (e.g., 'auto')", type=str, default='auto')
parser.add_argument('--return-full-text', dest='rft', action='store_true', help="Enable returning the full text from the pipeline")
parser.add_argument('--no-return-full-text', dest='rft', action='store_false', help="Disable returning the full text from the pipeline")
parser.set_defaults(rft=True)
parser.add_argument('-task', help="Task type for the pipeline (e.g., 'text-generation')", type=str, default='text-generation')
parser.add_argument('--do-sample', dest='do_sample', action='store_true', help="Enable sampling in generation")
parser.add_argument('--no-do-sample', dest='do_sample', action='store_false', help="Disable sampling in generation")
parser.set_defaults(do_sample=True)
parser.add_argument('-tmp', '--temperature', help="Temperature for generation", type=float, default=0.1)
parser.add_argument('-rpt', '--penality', help="Penalty for repetition in generation", type=float, default=1.15)
parser.add_argument('-mnt', '--tokens', help="Maximum new tokens to generate", type=int, default=1024)
parser.add_argument('-tpp', '--topp', help="Top-p sampling probability", type=float, default=1.0)
parser.add_argument('-tpk', '--topk', help="Top-k sampling limit", type=int, default=50)
args = parser.parse_args()

# Set the random seed for reproducibility.
random.seed(args.seed)

# Load the MovieLens 100k data.
data_ml_100k = None
with open("../data/ml_100k.json", 'r', encoding=util.DEFAULT_ENCONDING) as file:
    data_ml_100k = json.load(file)

if not data_ml_100k:
    raise Exception('Unable to find MovieLens 100k data json file.')

# Building indexes and similarity matrices for users and movies.
movie_indexes = util.build_index_dict(data_ml_100k)
assert(len(movie_indexes) == 1493)

user_sim_matrix = util.build_user_similarity_matrix(data_ml_100k, movie_indexes)
assert(user_sim_matrix.shape[0] == 943)
assert(user_sim_matrix.shape[-1] == 943)

movie_popularity_dict = util.build_movie_popularity_dict(data_ml_100k)
assert(len(movie_popularity_dict) == 1493)

item_sim_matrix = util.build_item_similarity_matrix(data_ml_100k)
assert(item_sim_matrix.shape == (1493, 1493))

id_list = list(range(0, len(data_ml_100k)))
assert(len(id_list) == 943)

# Authenticate with Hugging Face using the provided token.
login(token=args.token)

# Load the model tokenizer.
tokenizer = AutoTokenizer.from_pretrained(args.model)

# Setup BitsAndBytes configuration for model quantization.
bnb_config = BitsAndBytesConfig(
    load_in_4bit=args.load_4bit,
    bnb_4bit_quant_type=args.qtype,
    bnb_4bit_use_double_quant=args.doubleq
)

# Load the Hugging Face model with the specified configuration.
model = AutoModelForCausalLM.from_pretrained(
    args.model,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    device_map=args.devmap,
    trust_remote_code=True
)

# Create a pipeline for text generation with the loaded model and tokenizer.
pipe = transformers.pipeline(
    model=model,
    tokenizer=tokenizer,
    return_full_text=args.rft,
    task=args.task,
    do_sample=args.do_sample,
    temperature=args.temperature,
    repetition_penalty=args.penality,
    max_new_tokens=args.tokens,
    top_p=args.topp,
    top_k=args.topk
)

# Process each user in the dataset and generate recommendations.
results_data = {'args': vars(args), 'start_time': time.time()}
print(f"START TIME: {results_data['start_time']}")

for i in id_list:
    print(f'SAMPLE: {i} of {len(id_list)} ...')
    results_data[i] = {}

    watched_mv = data_ml_100k[i][0].split(' | ')[::-1]
    results_data[i]['ground_truth'] = data_ml_100k[i][-1]

    # Generate candidate items based on user filtering.
    candidate_items = util.sort_user_filtering_items(data_ml_100k, watched_mv, user_sim_matrix[i], args.fusers, args.fitems)
    random.shuffle(candidate_items)
    results_data[i]['candidate_set'] = candidate_items
    print('\tCANDIDATE MOVIE SET: ', candidate_items)

    # Process inputs through the pipeline in steps and record predictions.
    # STEP 01
    input_1 = util.PROMPT_TEMPLATE1.format(', '.join(watched_mv[-args.lenlimit:]))
    results_data[i]['input_1'] = input_1
    response = pipe(input_1)[0]['generated_text'].strip()
    predictions_1 = response[len(input_1):].strip()
    results_data[i]['predictions_1'] = predictions_1
    print('Output 1\n\t', predictions_1)

    # STEP 02
    input_2 = util.PROMPT_TEMPLATE2.format(', '.join(watched_mv[-args.lenlimit:]), predictions_1)
    results_data[i]['input_2'] = input_2
    response = pipe(input_2)[0]['generated_text'].strip()
    predictions_2 = response[len(input_2):].strip()
    results_data[i]['predictions_2'] = predictions_2
    print('\nOutput 2\n\t', predictions_2)

    # STEP 03
    input_3 = util.PROMPT_TEMPLATE3.format(', '.join(candidate_items), ', '.join(watched_mv[-args.lenlimit:]), predictions_1, predictions_2)
    results_data[i]['input_3'] = input_3
    response = pipe(input_3)[0]['generated_text'].strip()
    predictions_3 = response[len(input_3):].strip()
    results_data[i]['predictions_3'] = predictions_3
    print('\nOutput 3\n\t', predictions_3, '\n\n')
    
     # Check if the ground truth movie is in the final predictions.
    results_data[i]['hit'] = data_ml_100k[i][-1].lower() in predictions_3.lower()

results_data['end_time'] = time.time()
results_data['total_time'] = results_data['end_time'] - results_data['start_time']
print(f"END TIME: {results_data['end_time']}")
print(f"Total execution time: {results_data['total_time'] / 60:.2f} min")

# save dictionary to pickle file
filename = args.model.split('/')[-1].lower()
with open(f'../results/{filename}.pkl', 'wb') as file:
    pickle.dump(results_data, file)
