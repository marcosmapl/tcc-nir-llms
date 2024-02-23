MOUNT_GOOGLE_DRIVE = True
HUGGING_FACE_LOGIN = True
INSTALL_LIBS = True
SAVE_DATASETS = True
SAVE_MODELS = True
TRAIN_MODEL = True

import argparse
# import bitsandbytes as bnb
# import json
# import numpy as np
# import os
# import pandas as pd
# import pickle
# import plotly.express as px
# import random
# import re
# import torch

# from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
# from datasets import load_dataset, Dataset, DatasetDict
# from trl import SFTTrainer

# from peft import (
#     LoraConfig,
#     get_peft_model,
#     prepare_model_for_kbit_training,
#     AutoPeftModelForCausalLM,
#     PeftModel
# )

# from transformers import (
#     AutoModelForCausalLM,
#     AutoTokenizer,
#     TrainingArguments,
#     BitsAndBytesConfig,
#     DataCollatorForLanguageModeling,
#     HfArgumentParser,
#     TrainingArguments,
#     pipeline,
#     set_seed,
#     logging
# )

# random_seed = 2023
# random.seed(random_seed)

# from huggingface_hub import login
# access_token_read = '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'
# login(token = access_token_read)

parser = argparse.ArgumentParser()
parser.add_argument("--verbose", help="display a square of a given number", type=int)
parser.add_argument("--model", help="display a square of a given number", type=str)
args = parser.parse_args()

print(args)