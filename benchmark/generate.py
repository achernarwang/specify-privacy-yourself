import os
import json
from PIL import Image
from tqdm import trange
from pathlib import Path
from copy import deepcopy
from dotenv import load_dotenv
from transformers import set_seed
from argparse import ArgumentParser

from models import LVLM, API_LVLM, VLLM_LVLM
from config import REASONING_API_MODELS, INSTRUCT_TEMPLATES


def gen_eval_file(benchmark_file_path, save_path, add_distractors):
    assert isinstance(benchmark_file_path, Path) and benchmark_file_path.exists()
    assert isinstance(save_path, Path) and save_path.parent.exists()

    with open(benchmark_file_path) as f:
        benchmark_file_content = json.load(f)
    benchmark_list = benchmark_file_content["benchmark_list"]

    input_list = []

    def append_input_question_item(benchmark_item, is_privacy, scenario_item, scenario_is_privacy):
        input_question_item = {
            "benchmark_id": benchmark_item["benchmark_id"],
            "label": benchmark_item["label"],
            "is_privacy": is_privacy,
            "question_id": benchmark_item["question_item"]["question_id"],
            "image_path": benchmark_item["question_item"]["image_path"],
            "question": benchmark_item["question_item"]["question"],
            "ref_answer": benchmark_item["question_item"]["gt"],
        }

        if "label2text" in benchmark_item:
            input_question_item["label2text"] = benchmark_item["label2text"]

        if "instruct_templates" in benchmark_item:
            input_question_item["instruct_templates"] = benchmark_item["instruct_templates"]

        if add_distractors:
            input_question_item["other_labels"] = deepcopy(benchmark_item["other_labels"])
            input_question_item["other_labels"][benchmark_item["label"]] = is_privacy

        if scenario_item is not None and scenario_is_privacy is not None:
            input_question_item["scenario"] = {
                "id": scenario_item["scenario"],
                "is_privacy": scenario_is_privacy,
                "2nd_template": scenario_item["2nd_person_template"],
            }
        
        input_list.append(input_question_item)
    
    for benchmark_item in benchmark_list:
        append_input_question_item(benchmark_item, 1, None, None)
        append_input_question_item(benchmark_item, 0, None, None)
        for scenario_item in benchmark_item["non_privacy_scenarios"]:
            append_input_question_item(benchmark_item, 1, scenario_item, scenario_is_privacy=0)
        for scenario_item in benchmark_item["privacy_scenarios"]:
            append_input_question_item(benchmark_item, 1, scenario_item, scenario_is_privacy=1)
            append_input_question_item(benchmark_item, 0, scenario_item, scenario_is_privacy=1)

    with open(save_path, 'w') as f:
        for input_question_item in input_list:
            input_question_item["input_question"] = ""
            input_question_item["response"] = ""
            f.write(json.dumps(input_question_item) + "\n")
    
    return input_list

def gen_response(model:LVLM, chat_args:dict, input_list:list[dict], label2text:dict, save_path:Path, batch_size:int, images_dir:Path, seed:int=42):
    assert isinstance(images_dir, Path) and images_dir.exists()
    assert isinstance(save_path, Path) and save_path.parent.exists()

    input_list_ = [item for item in input_list if item["response"] == "" or item["response"].startswith("Request failed after")]

    set_seed(seed)

    for i in trange(0, len(input_list_), batch_size, ncols=120):
        input_item_batch = input_list_[i:i+batch_size]
        
        image_batch, question_batch = [], []
        for item in input_item_batch:
            if "label2text" in item:
                label2text_ = item["label2text"]
            else:
                assert isinstance(label2text[item["label"]], str)
                label2text_ = label2text

            if "instruct_templates" in item:
                instruct_templates = item["instruct_templates"]
            else:
                instruct_templates = INSTRUCT_TEMPLATES

            if "other_labels" in item:
                input_question = '\n'.join([instruct_templates[v].format(label2text_[k]) for k, v in item['other_labels'].items()])
            else:
                input_question = f"{instruct_templates[item['is_privacy']].format(label2text_[item['label']])}"
            
            if "scenario" in item:
                input_question += f"\n\n{item['scenario']['2nd_template'].format(item['question'])}"
            else:
                input_question += f"\n\n{item['question']}"
            
            item["input_question"] = input_question
            
            image_batch.append(Image.open(images_dir / item['image_path']).convert('RGB'))
            question_batch.append(item["input_question"])

        response_batch = model.visual_chat(images=image_batch, questions=question_batch, **chat_args)

        for item, response in zip(input_item_batch, response_batch):
            item['response'] = response

        with open(save_path, 'w') as f:
            for item in input_list:
                f.write(json.dumps(item) + "\n")

    return input_list

def main():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model name or model id")
    parser.add_argument("--gpu_id", type=str, default=None, help="Device to use")
    parser.add_argument("--benchmark_file", type=str, default="data/questions.json", help="Benchmark file path")
    parser.add_argument("--label_file", type=str, default="data/label2text.json", help="Label to text file path")
    parser.add_argument("--image_dir", type=str, default="data", help="Root directory of images")
    parser.add_argument("--result_dir", type=str, required=True, help="Result saving directory")
    parser.add_argument("--add_distractors", action="store_true", help="Add distractors to the input question")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for sampling")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")

    args = parser.parse_args()

    load_dotenv()
    
    if "/" in args.model and args.gpu_id is None:
        raise ValueError("Please specify the device to use for the model")
    
    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
        tensor_parallel_size = len(args.gpu_id.split(","))
    
    result_dir = Path(args.result_dir)
    model_name = args.model.split("/")[-1]
    if "checkpoint" in model_name:
        model_name = args.model.split("/")[-2] + "_ck" + model_name.split("-")[-1]

    os.makedirs(result_dir / "resp", exist_ok=True)

    # read input list
    benchmark_file:str = args.benchmark_file
    if benchmark_file.endswith(".jsonl"):
        with open(benchmark_file) as f:
            input_list = [json.loads(line) for line in f]
    else:
        input_list = gen_eval_file(Path(benchmark_file), result_dir / f"resp/{model_name}.jsonl", args.add_distractors)
    print(len(input_list))

    with open(args.label_file) as f:
        label2text = json.load(f)

    # generate response
    print(f"Testing {model_name}...")
    if "/" in args.model:
        model = VLLM_LVLM(model_id=args.model, tensor_parallel_size=tensor_parallel_size)
        chat_args = {"max_new_tokens": args.max_new_tokens, "temperature": args.temperature}
    else:
        model = API_LVLM(model_id=args.model, enable_async=True)
        if args.model in REASONING_API_MODELS:
            chat_args = {"max_new_tokens": None, "temperature": 1, "max_retries": 1, "retry_delay": 2.0}
        else:
            chat_args = {"max_new_tokens": args.max_new_tokens, "temperature": args.temperature, "max_retries": 1, "retry_delay": 2.0}
    
    gen_response(model, chat_args, input_list, label2text, result_dir / f"resp/{model_name}.jsonl", args.batch_size, Path(args.image_dir))

if __name__ == "__main__":
    main()