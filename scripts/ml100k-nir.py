import argparse
import time
import json
import random
import transformers
import torch
import pickle
import util

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import login

# Setup the argument parser with descriptions for each option.
parser = argparse.ArgumentParser(description="Script for generating recommendations using a HuggingFace model.")
parser.add_argument('-model', help="HuggingFace model name", type=str, default=None)
parser.add_argument('-hf', help="HuggingFace authentication token", type=str, default=None)
parser.add_argument("-nsu", help="Number of similar users for filtering", type=int, default=12)
parser.add_argument("-nci", help="Number of candidate items for recommendation", type=int, default=19)
parser.add_argument("-lenlimit", help="Length limit", type=int, default=8)
parser.add_argument('-seed', help="Random seed for reproducibility", type=int, default=0)
parser.add_argument('-qtp', '--qtype', help="Quantization type for BitsAndBytes configuration", type=str, default='nf4')
parser.add_argument('--use-4bit', dest='use_4bit', action='store_true', help="Enable loading models in 4-bit quantization")
parser.add_argument('--no-use-4bit', dest='use_4bit', action='store_false', help="Disable loading models in 4-bit quantization")
parser.set_defaults(use_4bit=True)
parser.add_argument("-b4b-cdtype", help="This parameter allows you to change the compute dtype of the quantized model such as torch.bfloat16.", type=str, default="float16")
parser.add_argument("-b4b-qtype", help='This parameter dictates the quantization data type employed in the "bnb.nn.Linear4Bit" layers.', type=str, default="nf4")
parser.add_argument('--nested-quantization', dest='nested_quantization', action='store_true')
parser.add_argument('--no-nested-quantization', dest='nested_quantization', action='store_false')
parser.set_defaults(nested_quantization=True)
parser.add_argument('-devmap', help="Device map policy for PyTorch (e.g., 'auto')", type=str, default='auto')
parser.add_argument('--return-full-text', dest='rft', action='store_true', help="Enable returning the full text from the pipeline")
parser.add_argument('--no-return-full-text', dest='rft', action='store_false', help="Disable returning the full text from the pipeline")
parser.set_defaults(rft=True)
parser.add_argument('-task', help="Task type for the pipeline (e.g., 'text-generation')", type=str, default='text-generation')
parser.add_argument('--do-sample', dest='do_sample', action='store_true', help="Enable sampling in generation")
parser.add_argument('--no-do-sample', dest='do_sample', action='store_false', help="Disable sampling in generation")
parser.set_defaults(do_sample=True)
parser.add_argument('-temperature', help="Temperature for generation", type=float, default=0.1)
parser.add_argument('-penality', help="Penalty for repetition in generation", type=float, default=1.15)
parser.add_argument('-maxtokens', help="Maximum new tokens to generate", type=int, default=1024)
parser.add_argument('-topp', help="Top-p sampling probability", type=float, default=1.0)
parser.add_argument('-topk', help="Top-k sampling limit", type=int, default=50)
args = parser.parse_args()

# Set the random seed for reproducibility.
random.seed(args.seed)

# load movie lens 100k dataset
data_ml_100k = util.read_json("../data/ml_100k.json")

id_list = list(range(0, len(data_ml_100k)))
assert(len(id_list) == 943)

# Building indexes and similarity matrices for users and movies.
movie_idx = util.build_index_dict(data_ml_100k)
user_sim_matrix = util.build_user_similarity_matrix(data_ml_100k, movie_idx)
pop_dict = util.build_movie_popularity_dict(data_ml_100k)
item_sim_matrix = util.build_item_similarity_matrix(data_ml_100k)

# Authenticate with Hugging Face using the provided token.
login(token=args.hf)

# gets model simple name from huggingface namespace
model_name = args.model.split('/')[-1].lower().strip()

# Setup BitsAndBytes configuration for model quantization.
bnb_config = util.create_bnb_config(args)

# Load the Hugging Face model with the specified configuration.
model = AutoModelForCausalLM.from_pretrained(
    args.model,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    device_map=args.devmap,
    trust_remote_code=True
)
# Load the model tokenizer.
tokenizer = AutoTokenizer.from_pretrained(args.model)

# Create a pipeline for text generation with the loaded model and tokenizer.
pipe = transformers.pipeline(
    model=model,
    tokenizer=tokenizer,
    return_full_text=args.rft,
    task=args.task,
    do_sample=args.do_sample,
    temperature=args.temperature,
    repetition_penalty=args.penality,
    max_new_tokens=args.maxtokens,
    top_p=args.topp,
    top_k=args.topk
)

# Process each user in the dataset and generate recommendations.
results = {'args': vars(args), 'start_time': time.time()}
results['namespace'] = args.model
results['model_name'] = model_name
results 
print(f"START TIME: {results['start_time']}")

for i in id_list:
    print(f'SAMPLE: {i} of {len(id_list)} ...')
    results[i] = {}

    watched_mv = data_ml_100k[i][0].split(' | ')[::-1]
    results[i]['ground_truth'] = data_ml_100k[i][-1]

    # Generate candidate items based on user filtering.
    candidate_items = util.sort_user_filtering_items(data_ml_100k, watched_mv, user_sim_matrix[i], args.nsu, args.nci)
    random.shuffle(candidate_items)
    results[i]['candidate_set'] = candidate_items
    print('\tCANDIDATE MOVIE SET: ', candidate_items)

    # Process inputs through the pipeline in steps and record predictions.
    # STEP 01
    input_1 = util.PROMPT_TEMPLATE1.format(', '.join(watched_mv[-args.lenlimit:]))
    results[i]['input_1'] = input_1
    response = pipe(input_1)[0]['generated_text'].strip()
    predictions_1 = response[len(input_1):].strip()
    results[i]['predictions_1'] = predictions_1
    print('Output 1\n\t', predictions_1)

    # STEP 02
    input_2 = util.PROMPT_TEMPLATE2.format(', '.join(watched_mv[-args.lenlimit:]), predictions_1)
    results[i]['input_2'] = input_2
    response = pipe(input_2)[0]['generated_text'].strip()
    predictions_2 = response[len(input_2):].strip()
    results[i]['predictions_2'] = predictions_2
    print('\nOutput 2\n\t', predictions_2)

    # STEP 03
    input_3 = util.PROMPT_TEMPLATE3.format(', '.join(candidate_items), ', '.join(watched_mv[-args.lenlimit:]), predictions_1, predictions_2)
    results[i]['input_3'] = input_3
    response = pipe(input_3)[0]['generated_text'].strip()
    predictions_3 = response[len(input_3):].strip()
    results[i]['predictions_3'] = predictions_3
    print('\nOutput 3\n\t', predictions_3, '\n\n')
    
     # Check if the ground truth movie is in the final predictions.
    results[i]['hit'] = data_ml_100k[i][-1].lower() in predictions_3.lower()

results['end_time'] = time.time()
results['total_time'] = results['end_time'] - results['start_time']
print(f"END TIME: {results['end_time']}")
print(f"Total execution time: {results['total_time'] / 60:.2f} min")

# save dictionary to pickle file
with open(f'../results/ml100k-zs-nir-su{args.nsu}-ci{args.nci}-{model_name}.pkl', 'wb') as file:
    pickle.dump(results, file)
