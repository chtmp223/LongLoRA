import os
import sys
import math
import torch
import argparse
import textwrap
import transformers
from peft import PeftModel
from transformers import GenerationConfig, TextStreamer
from llama_attn_replace import replace_llama_attn
import torch.nn.functional as F

PROMPT_DICT = {
    "prompt_no_input": (
#        "Below is an instruction that describes a task. "
 #       "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
    "prompt_no_input_llama2": (
        "[INST] <<SYS>>\n"
        "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\n"
        "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n"
        "<</SYS>> \n\n {instruction} [/INST]"
    ),
    "prompt_llama2": "[INST]{instruction}[/INST]"
}

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--material', type=str, default="")
    parser.add_argument('--base_model', type=str, default="/data1/pretrained-models/llama-7b-hf")
    parser.add_argument('--cache_dir', type=str, default="./cache")
    parser.add_argument('--context_size', type=int, default=-1, help='context size during fine-tuning')
    parser.add_argument('--flash_attn', type=bool, default=False, help='')
    parser.add_argument('--temperature', type=float, default=0.6, help='')
    parser.add_argument('--top_p', type=float, default=0.9, help='')
    parser.add_argument('--max_gen_len', type=int, default=512, help='')
    args = parser.parse_args()
    return args

def read_txt_file(material_txt):
    if not material_txt.split(".")[-1]=='txt':
        raise ValueError("Only support txt or pdf file.")
    content = ""
    with open(material_txt) as f:
        for line in f.readlines():
            content += line
    return content

def build_generator(
    model, tokenizer, temperature=0.6, top_p=0.9, max_gen_len=4096, use_cache=True
):
    def response(prompt):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        streamer = TextStreamer(tokenizer)
        
        # Enable output_scores to get logits
        output = model.generate(
            **inputs,
            max_new_tokens=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            use_cache=use_cache,
            output_scores=True,  # This will return logits
            return_dict_in_generate=True,  # Ensures output is returned in a dict
            streamer=streamer,
            do_sample=False # Change to True to use temperature and top_p
        )
        
        # Decode the generated ids to tokens
        decoded_output = [tokenizer.decode(g, skip_special_tokens=True) for g in output.sequences]
        token_log_probs = []

        # Calculate log probabilities from logits directly
        for i, logits in enumerate(output.scores):
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)  # Compute log probabilities directly from logits
            top_i = output.sequences[0][i+1]  # Get the generated token id at each step
            log_prob = log_probs[0][top_i].item()  # Get the log probability of the generated token
            token_log_probs.append((tokenizer.decode([top_i]), log_prob))
        
        print(decoded_output, token_log_probs)

        return decoded_output, token_log_probs

    return response

def main(args):
    if args.flash_attn:
        replace_llama_attn(inference=True)

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        args.base_model,
        cache_dir=args.cache_dir,
    )

    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and args.context_size > orig_ctx_len:
        scaling_factor = float(math.ceil(args.context_size / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}

    # Load model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.base_model,
        config=config,
        cache_dir=args.cache_dir,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.resize_token_embeddings(32001)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.base_model,
        cache_dir=args.cache_dir,
        model_max_length=args.context_size if args.context_size > orig_ctx_len else orig_ctx_len,
        padding_side="right",
        use_fast=False,
    )

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    model.eval()

    respond = build_generator(model, tokenizer, temperature=args.temperature, top_p=args.top_p,
                              max_gen_len=args.max_gen_len, use_cache=True)

    material = read_txt_file(args.material)
    prompt_no_input = PROMPT_DICT["prompt_llama2"]
    prompt = prompt_no_input.format_map({"instruction": material})

    output = respond(prompt=prompt)

if __name__ == "__main__":
    args = parse_config()
    main(args)
