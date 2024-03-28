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

parser = argparse.ArgumentParser()
parser.add_argument("-mdl", help="HuggingFace model name", type=str, default=None)
parser.add_argument("-hft", help="HuggingFace auth token", type=str, default=None)
parser.add_argument("-ll", help="Length limit", type=int, default=8)
parser.add_argument("-ncd", help="Number of similar candidates", type=int, default=19)
parser.add_argument("-nuf", help="Number of similar users for filtering", type=int, default=12)
parser.add_argument("-seed", help="Random seed", type=int, default=0)
parser.add_argument('--load-4bit', dest='load_4bit', action='store_true')
parser.add_argument('--no-load-4bit', dest='load_4bit', action='store_false')
parser.set_defaults(load_4bit=True)
parser.add_argument('--load-8bit', dest='load_8bit', action='store_true')
parser.add_argument('--no-load-8bit', dest='load_8bit', action='store_false')
parser.set_defaults(load_8bit=True)
parser.add_argument("-dmp", help="GPU device map policy", type=str, default='auto')
parser.add_argument("-rft", help="Return full text", type=bool, default=True)
parser.add_argument('--return-full-text', dest='rft', action='store_true')
parser.add_argument('--no-return-full-text', dest='rft', action='store_false')
parser.set_defaults(rft=True)
parser.add_argument("-qcfg", help="Quantize config", type=str, default=None)
parser.add_argument("-task", help="Model task", type=str, default='text-generation')
parser.add_argument('--do-sample', dest='do_sample', action='store_true')
parser.add_argument('--no-do-sample', dest='do_sample', action='store_false')
parser.set_defaults(do_sample=True)
parser.add_argument("-temperature", help="Model temperature", type=float, default=0.1)
parser.add_argument("-rep_penality", help="Model repetition penality", type=float, default=1.15)
parser.add_argument("-max_new_tokens", help="Model max new tokens", type=int, default=1024)
parser.add_argument("-top_p", help="Model top p", type=float, default=1.0)
parser.add_argument("-top_k", help="Model top k", type=int, default=50)
args = parser.parse_args()

random.seed(args.seed)

# load movie lens 100k data from json file
data_ml_100k = None
with open("ml_100k.json") as f:
    data_ml_100k = json.load(f)

if not data_ml_100k:
    raise Exception('Unable to find MovieLens 100k data json file.')

# movie index dict
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

# hugging face authentication    
login(token=args.hft)

models_list = [
    'meta-llama/Llama-2-13b-chat-hf'
]

for base_model in models_list:
    args.mdl = base_model
    # load model tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=args.load_4bit,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
    )

    # load hugging face model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map=args.dmp,
        trust_remote_code=True
    )

    # create pipeline
    pipe = transformers.pipeline(
        model=model,
        tokenizer=tokenizer,
        return_full_text=args.rft,
        task=args.task,
        do_sample=args.do_sample,
        temperature=args.temperature,
        repetition_penalty=args.rep_penality,
        max_new_tokens=args.max_new_tokens,
        top_p=args.top_p,
        top_k=args.top_k
    )

    results_data = dict()
    results_data['args'] = args
    results_data['start_time'] = time.time()
    print(f"START TIME: {results_data['start_time']}")

    for i in id_list:
        print(f'SAMPLE: {i} of {len(id_list)} ...')
        results_data[i] = dict()

        watched_mv = data_ml_100k[i][0].split(' | ')[::-1]
        results_data[i]['gt'] = data_ml_100k[i][-1]

        candidate_items = util.sort_user_filtering_items(data_ml_100k, watched_mv, user_sim_matrix[i], args.nuf, args.ncd)
        random.shuffle(candidate_items)
        results_data[i]['candidate_set'] = candidate_items
        print('\n CANDIDATE MOVIE SET: ', candidate_items)

        # STEP 01
        input_1 = util.PROMPT_TEMPLATE1.format(', '.join(watched_mv[-args.ll:]))
        results_data[i]['input_1'] = input_1
        response = pipe(input_1)
        response = response[0]['generated_text'].strip()
        predictions_1 = response[len(input_1):].strip()
        results_data[i]['predictions_1'] = predictions_1
        print('predictions_1\n', predictions_1)

        # STEP 02
        input_2 = util.PROMPT_TEMPLATE2.format(', '.join(watched_mv[-args.ll:]), predictions_1)
        results_data[i]['input_2'] = input_2
        response = pipe(input_2)
        response = response[0]['generated_text'].strip()
        predictions_2 = response[len(input_2):].strip()
        results_data[i]['predictions_2'] = predictions_2
        print('\npredictions_2\n', predictions_2)

        # STEP 03
        input_3 = util.PROMPT_TEMPLATE3.format(', '.join(candidate_items), ', '.join(watched_mv[-args.ll:]), predictions_1, predictions_2)
        results_data[i]['input_3'] = input_3
        response = pipe(input_3)
        response = response[0]['generated_text'].strip()
        predictions_3 = response[len(input_3):].strip()
        results_data[i]['predictions_3'] = predictions_3
        print('\npredictions_3\n', predictions_3, '\n\n')
        
        hit = False
        if data_ml_100k[i][-1].lower() in predictions_3.lower():
            hit = True

        results_data[i]['hit'] = hit

    results_data['end_time'] = time.time()
    print(f"END TIME: {results_data['end_time']}")
    total_time = results_data['end_time'] - results_data['start_time']
    results_data['total_time'] = total_time
    print(f"Total execution time: {(total_time / 60):.2f} min")

    # save dictionary to pickle file
    filename = base_model.split('/')[-1].lower()
    with open(f'{filename}.pkl', 'wb') as fp:
        pickle.dump(results_data, fp)

# save dictionary to json file
# with open(args.json, 'w') as fp:
#     json.dump(results_data, fp)