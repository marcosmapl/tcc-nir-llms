import argparse
import random
import util
import os
import pickle

from huggingface_hub import login
# from transformers.generation.utils import top_k_top_p_filtering
# from trl import SFTTrainer
# from peft import get_peft_model, prepare_model_for_kbit_training
# from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("-model", help="HuggingFace model namespace", type=str, default=None)
parser.add_argument("-hf", help="HuggingFace auth token", type=str, default=None)
parser.add_argument("-nsu", help="Number of similar users for filtering", type=int, default=12)
parser.add_argument("-nci", help="Number of candidate items for recommendation", type=int, default=19)
parser.add_argument("-kfold", help="Cross validation K-Fold splits", type=int, default=4)
parser.add_argument("-seed", help="random seed", type=int, default=0)
parser.add_argument("-lora-r", help="the LoRA attention dimension, specifies the rank of the update matrices.", type=int, default=64)
parser.add_argument("-lora-a", help="regulates the LoRA scaling factor. Adjusting this ratio will help you to balance between over-fitting and under-fitting.", type=int, default=16)
parser.add_argument("-lora-d", help="Dropout probability for LoRA layers.", type=float, default=0.1)
parser.add_argument('--use-4bit', dest='use_4bit', action='store_true')
parser.add_argument('--no-use-4bit', dest='use_4bit', action='store_false')
parser.set_defaults(use_4bit=True)
parser.add_argument("-b4b-cdtype", help="This parameter allows you to change the compute dtype of the quantized model such as torch.bfloat16.", type=str, default="float16")
parser.add_argument("-b4b-qtype", help='This parameter dictates the quantization data type employed in the "bnb.nn.Linear4Bit" layers.', type=str, default="nf4")
parser.add_argument('--nested-quantization', dest='nested_quantization', action='store_true')
parser.add_argument('--no-nested-quantization', dest='nested_quantization', action='store_false')
parser.set_defaults(nested_quantization=True)
parser.add_argument("-lg-steps", help="Determines the frequency of logging, i.e., log every N steps.", type=int, default=10)
parser.add_argument("-sv-steps", help="Determines the frequency of saving the model, i.e., save model checkpoint every N steps.", type=int, default=20)
parser.add_argument("-sv_ttl", help="Limits the total amount of checkpoints and deletes older checkpoints.", type=int, default=2)
parser.add_argument("-nte", help="Specifies the number of training epochs.", type=int, default=1)
parser.add_argument('--fp16', dest='fp16', action='store_true')
parser.add_argument('--no-fp16', dest='fp16', action='store_false')
parser.set_defaults(fp16=True)
parser.add_argument('--bf16', dest='bf16', action='store_true')
parser.add_argument('--no-bf16', dest='bf16', action='store_false')
parser.set_defaults(bf16=False)
parser.add_argument("-pdtbs", help="Defines the batch size per GPU for training.", type=int, default=1)
parser.add_argument("-ga-steps", help="Specifies the number of steps to accumulate the gradients.", type=int, default=1)
parser.add_argument('--gc', dest='gc', action='store_true')
parser.add_argument('--no-gc', dest='gc', action='store_false')
parser.set_defaults(gc=True) # gradient checkpoint
parser.add_argument("-max-gnorm", help="Maximum gradient normal (gradient clipping).", type=float, default=0.3)
parser.add_argument("-lr", help="Initial learning rate (AdamW optimizer).", type=float, default=2e-4)
parser.add_argument("-lr-scheduler", help="Specifies learning rate scheduler, e.g., constant, linear, cosine, and polynomial decay.", type=str, default="constant")
parser.add_argument("-wd", help="Specifies weight decay to the parameters.", type=float, default=0.001)
parser.add_argument("-optm", help="Optimizer to use.", type=str, default="paged_adamw_32bit")
parser.add_argument("-wm-ratio", help="Ratio steps for a linear warm up (from 0 to learning rate).", type=float, default=0.03)
parser.add_argument('--group-batches', dest='group_batches', action='store_true')
parser.add_argument('--no-group-batches', dest='group_batches', action='store_false')
parser.set_defaults(group_batches=True)
parser.add_argument("-max-seq-len", help="Maximum sequence length to use.", type=int, default=None)
parser.add_argument('--packing', dest='packing', action='store_true')
parser.add_argument('--no-packing', dest='packing', action='store_false')
parser.set_defaults(packing=True)
parser.add_argument("-device", help="GPU number.", type=int, default=0)
parser.add_argument('-temperature', help="Temperature for generation", type=float, default=0.1)
parser.add_argument('-penality', help="Penalty for repetition in generation", type=float, default=1.15)
parser.add_argument('-maxtokens', help="Maximum new tokens to generate", type=int, default=1024)
parser.add_argument('-topp', help="Top-p sampling probability", type=float, default=1.0)
parser.add_argument('-topk', help="Top-k sampling limit", type=int, default=50)
parser.add_argument('--train-model', dest='train_model', action='store_true')
parser.add_argument('--no-train-model', dest='train_model', action='store_false')
parser.set_defaults(train_model=True)
parser.add_argument('--save-tuned', dest='save_tuned', action='store_true')
parser.add_argument('--no-save-tuned', dest='save_tuned', action='store_false')
parser.add_argument('-loglevel', type=int, default=0)
parser.set_defaults(save_tuned=True)

args = parser.parse_args()
random.seed(args.seed)

# hugging face loging
login(token = args.hf)

# gets model simple name from huggingface namespace
model_name = args.model.split('/')[-1].lower().strip()

# load movie lens 100k dataset
data_ml_100k = util.read_json("../data/ml_100k.json")

id_list = list(range(0, len(data_ml_100k)))
assert(len(id_list) == 943)

movie_idx = util.build_index_dict(data_ml_100k)
user_matrix_sim = util.build_user_similarity_matrix(data_ml_100k, movie_idx)
pop_dict = util.build_movie_popularity_dict(data_ml_100k)
item_sim_matrix = util.build_item_similarity_matrix(data_ml_100k)

# create a dataset sample by collaborative filtering
cand_ids = util.get_candidate_ids_list(data_ml_100k, id_list, user_matrix_sim, args.nsu, args.nci)

# build training arguments object
training_arguments = util.create_training_arguments(model_name, args)

# load model zero-shot nir prompt results
ds = util.create_dataset(model_name, args, cand_ids)
# ds.save_to_disk(f'../datasets/ml100k-sample264-{model_name}')

results = util.train_model_cv(model_name, ds, args)
os.makedirs(f"../results", exist_ok=True)
with open(f"../results/ml100k-ft-cv-su{args.nsu}-ci{args.nci}-{model_name}.pkl", 'wb') as fp:
    pickle.dump(results, fp)