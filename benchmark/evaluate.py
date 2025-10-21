import os
import json
from tqdm import trange
from pathlib import Path
from dotenv import load_dotenv
from argparse import ArgumentParser

from models import LVLM, API_LVLM, VLLM_LVLM
from config import EVAL_PROMPTS

# useful for evaluating the self moderation method
RespKey2EvalKey = {
    "response": "eval_resp",
    "resp_2": "eval_2",
    "resp_3": "eval_3",
}

def eval_response(eval_model:LVLM, chat_args:dict, resp_list:list[dict], save_path:Path, batch_size:int, label2text:dict, resp_k = "response", eval_k = "eval_resp"):
    assert isinstance(save_path, Path) and save_path.parent.exists()

    resp_list_ = []
    for item in resp_list:
        if item[resp_k].startswith("Request failed after"):
            item[eval_k] = ""
            continue
        if item.get(eval_k) is None or item[eval_k] not in ["respond", "reject", "abstain"]:
            resp_list_.append(item)

    for i in trange(0, len(resp_list_), batch_size, ncols=120):
        resp_batch = resp_list_[i:i+batch_size]

        input_batch = []
        for resp in resp_batch:
            input_batch.append(EVAL_PROMPTS.format(label2text[resp['label']], resp['question'], resp[resp_k]))

        eval_resp_batch = eval_model.chat(input_batch, **chat_args)
        
        for item, eval_resp in zip(resp_batch, eval_resp_batch):
            if "reject" in eval_resp.lower():
                item[eval_k] = "reject"
            elif "respond" in eval_resp.lower():
                item[eval_k] = "respond"
            else:
                item[eval_k] = eval_resp
        
        with open(save_path, 'w') as f:
            for item in resp_list:
                f.write(json.dumps(item) + "\n")
    
    return resp_list


def main():
    parser = ArgumentParser()
    parser.add_argument("--eval_model", type=str, default="gpt-4o", help="Evaluation model name or model id")
    parser.add_argument("--gpu_id", type=str, default=None, help="Device to use")
    parser.add_argument("--result_dir", type=str, required=True, help="Result saving directory")
    parser.add_argument("--result_file", type=str, required=True, nargs='+', help="Result file name")
    parser.add_argument("--label_file", type=str, default="data/label2text.json", help="Label to text file path")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for sampling")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--resp_k", type=str, default="response", help="Key to store response")

    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("EVAL_API_KEY")
    base_url = os.getenv("EVAL_BASE_URL")

    if "/" in args.eval_model and args.gpu_id is None:
        raise ValueError("Please specify the device to use for the model")

    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
        tensor_parallel_size = len(args.gpu_id.split(","))

    result_dir = Path(args.result_dir)
    os.makedirs(result_dir / "eval", exist_ok=True)

    with open(args.label_file) as f:
        label2text = json.load(f)

    assert isinstance(args.result_file, list)
    for result_file in args.result_file:
        assert (result_dir / result_file).exists()

    eval_model_name = args.eval_model.split("/")[-1]
        
    if "/" in args.eval_model:
        eval_model = VLLM_LVLM(model_id=args.eval_model, tensor_parallel_size=tensor_parallel_size)
        eval_chat_args = {"max_new_tokens": args.max_new_tokens, "temperature": args.temperature}
    else:
        eval_model = API_LVLM(model_id=args.eval_model, api_key=api_key, base_url=base_url)
        eval_chat_args = {"max_new_tokens": args.max_new_tokens, "temperature": args.temperature, "max_retries": 1, "retry_delay": 2.0}

    for result_file in args.result_file:
        model_name = Path(result_file).stem.split("__")
        model_name = model_name[1] if len(model_name) > 1 else model_name[0]
        assert "/" not in model_name

        with open(result_dir / result_file, 'r') as f:
            resp_list = [json.loads(line) for line in f]

        print(f"Evaluating {model_name}...")
        
        if args.resp_k == "response":
            save_name = f"eval/{model_name}__{eval_model_name}.jsonl"
        else:
            save_name = f"eval/{model_name}__{eval_model_name}_{args.resp_k}.jsonl"
        eval_response(eval_model, eval_chat_args, resp_list, result_dir / save_name, args.batch_size, label2text, args.resp_k, RespKey2EvalKey[args.resp_k])

if __name__ == "__main__":
    main()