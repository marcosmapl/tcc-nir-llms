import argparse
import bitsandbytes as bnb
import pandas as pd
import random
import torch
import pickle
import time

# from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from datasets import Dataset, DatasetDict
from huggingface_hub import login
from transformers.generation.utils import top_k_top_p_filtering
from trl import SFTTrainer

from peft import (
    LoraConfig,
    get_peft_model
)

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)

parser = argparse.ArgumentParser()
# parser.add_argument("-mdl", help="HuggingFace model name", type=str, default=None)
parser.add_argument("-hf", help="HuggingFace auth token", type=str, default=None)
parser.add_argument("-ll", help="Length limit", type=int, default=8)
parser.add_argument("-ncd", help="Number of similar candidates", type=int, default=19)
parser.add_argument("-nuf", help="Number of similar users for filtering", type=int, default=12)
# parser.add_argument("-train-data", help="Train data file path.", type=str, default=None)
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
parser.add_argument("-sv-steps", help="Determines the frequency of saving the model, i.e., save model checkpoint every N steps.", type=int, default=200)
parser.add_argument("-sv_ttl", help="Limits the total amount of checkpoints and deletes older checkpoints.", type=int, default=2)
parser.add_argument("-nte", help="Specifies the number of training epochs.", type=int, default=1)
parser.add_argument('--fp16', dest='fp16', action='store_true')
parser.add_argument('--no-fp16', dest='fp16', action='store_false')
parser.set_defaults(fp16=True)
parser.add_argument('--bf16', dest='bf16', action='store_true')
parser.add_argument('--no-bf16', dest='bf16', action='store_false')
parser.set_defaults(bf16=True)
parser.add_argument("-pdtbs", help="Defines the batch size per GPU for training.", type=int, default=1)
# parser.add_argument("-pdebs", help="Defines the batch size per GPU for evaluation.", type=int, default=1)
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
# parser.add_argument("-test-size", help="Train/Test data split percent (float).", type=float, default=0.2)
parser.add_argument("-mtk", help="Max new tokens.", type=int, default=1024)
parser.add_argument('--train-model', dest='train_model', action='store_true')
parser.add_argument('--no-train-model', dest='train_model', action='store_false')
parser.set_defaults(train_model=True)
# parser.add_argument("-checkpoint", help="Check point number.", type=int, default=0)
parser.add_argument('--save-tuned', dest='save_tuned', action='store_true')
parser.add_argument('--no-save-tuned', dest='save_tuned', action='store_false')
parser.set_defaults(save_tuned=True)

args = parser.parse_args()
random.seed(args.seed)

# hugging face loging
login(token = args.hf)

with open('data/cand_ids.pkl', 'rb') as fp:
    cand_ids = pickle.load(fp)

hf_models = [
    # 'meta-llama/Llama-2-7b-chat-hf',
    'meta-llama/Llama-2-13b-chat-hf'
]

for namespace in hf_models:
    model_name = namespace.split('/')[-1].lower().strip()
    with open(f'results/ml100k-sample943-{model_name}.pkl', 'rb') as fp:
        results_data = pickle.load(fp)

    data = []
    for i in range(943):
        input_text = results_data[i]['input_3'].split('### QUESTION:')[0].strip()
        input_text = input_text + '\n\n### QUESTION: Can you recommend a movie from the "Candidate movie set" similar to the "Five most featured movie"?'
        input_text = input_text + '\n\n###ANSWER: ' + results_data[i]['gt'] + ' [end-gen]'
        data.append((i, results_data[i]['input_3'], results_data[i]['gt'], input_text))
    
    df = pd.DataFrame(data, columns=['id', 'prompt', 'ground_truth', 'input_text'])
    print(df.shape)
    ds = DatasetDict({
        'train': Dataset.from_pandas(df[~df.index.isin(cand_ids)]),
        'validation': Dataset.from_pandas(df.loc[cand_ids])
    })
    print(ds)
    ds.save_to_disk(f'datasets/{namespace}')

    # bits and bytes quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=args.use_4bit,
        bnb_4bit_quant_type=args.b4b_qtype,
        bnb_4bit_compute_dtype=getattr(torch, args.b4b_cdtype),
        bnb_4bit_use_double_quant=args.nested_quantization,
    )

    # check if GPU supports bffloat16
    if getattr(torch, args.b4b_cdtype) == torch.float16 and args.use_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print("Your GPU supports bfloat16: accelerate training with bf16=True")
            print("=" * 80)

    # lora config
    peft_config = LoraConfig(
        lora_alpha=args.lora_a,
        lora_dropout=args.lora_d,
        r=args.lora_r,
        bias='none',
        task_type='CAUSAL_LM',
    )

    # build training arguments object
    training_arguments = TrainingArguments(
        output_dir=f'train/{model_name}',
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

    # load base model on GPU
    model = AutoModelForCausalLM.from_pretrained(
        namespace,
        quantization_config=bnb_config,
        device_map=args.device
    )
    model.config.use_cache = True
    model.config.pretraining_tp = 1

    # load model tokenizer
    tokenizer = AutoTokenizer.from_pretrained(namespace, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"additional_special_tokens" : ['[end-gen]']})

    # resize model tokens size to match with tokenizer
    model.resize_token_embeddings(len(tokenizer))

    # create trainer object
    trainer = SFTTrainer(
        model=model,
        train_dataset=ds['train'],
        peft_config=peft_config,
        dataset_text_field='input_text',
        max_seq_length=args.max_seq_len,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=args.packing,
    )

    # cast normalization layers.
    for name, module in trainer.model.named_modules():
        if "norm" in name:
            module = module.to(torch.float16)

    # WANDB: 452415af745934fe1163b7f81dee7094e2f268a1
    # MODEL FINE TUNING
    if args.train_model:
        trainer.train()

    # save fine tuned model to disk
    if args.save_tuned:
        trainer.model.save_pretrained(f"models/{namespace}")

    lora_config = LoraConfig.from_pretrained(f"models/{namespace}")

    model = AutoModelForCausalLM.from_pretrained(
        namespace,
        quantization_config=bnb_config,
        device_map=args.device
    )

    tuned_model = get_peft_model(model, lora_config)
    tuned_model.print_trainable_parameters()
    tuned_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(namespace, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"additional_special_tokens" : ['[end-gen]']})

    # resize model embedding
    tuned_model.resize_token_embeddings(len(tokenizer))

    results_data = dict()
    results_data['namespace'] = namespace
    results_data['model_name'] = model_name
    results_data['args'] = args
    results_data['start_time'] = time.time()
    print(f"START TIME: {results_data['start_time']}")

    for i in range(ds['validation'].shape[0]):
        response = dict()
        response['id'] = ds['validation'][i]['id']
        response['input'] = ds['validation'][i]['prompt']
        response['gt'] = ds['validation'][i]['ground_truth']

        inputs = tokenizer(ds['validation'][i]['prompt'], return_tensors="pt").to("cuda:0")
        outputs = tuned_model.generate(**inputs, max_new_tokens=1024, eos_token_id=[tokenizer.get_vocab()["[end-gen]"]])
        generated = tokenizer.decode(outputs[0], skip_special_tokens=False)
        #   print(generated)
        response['output'] = generated.split('[end-gen]')[0]
        results_data[i] = response
        print(f'PROMPT SAMPLE: {i}')
        print(f"\tOUTPUT: {response['output']}\n")
        print(f"\tGROUND TRUTH: {response['gt']}\n\n")

    results_data['end_time'] = time.time()
    results_data['total_time'] = results_data['end_time'] - results_data['start_time']
    print(f"END TIME: {results_data['end_time']}")
    print(f"Total execution time: {(results_data['total_time'] / 60):.2f} min")

    with open(f"results/ml100k-fine-tuning-{model_name}.pkl", 'wb') as fp:
        pickle.dump(results_data, fp)
