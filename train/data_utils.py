import os
import json
import random
from copy import deepcopy
from datasets import Dataset, Image

MESSAGE_TEMPLATE = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": ""},
        ]
    },
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": ""}
        ]
    }
]

INSTRUCT_TEMPLATES = ("{} is considered as non-private information.", "{} is considered as private information.")

ABS_PRIV_LABELS = ["a12_semi_nudity", "a13_full_nudity", "a29_ausweis", "a31_passport", "a32_drivers_license"]

def prepare_dataset(data_path:str, label_path:str, image_folder:str, add_distractors:bool, save_path:str=None, shuffle:bool=False, seed:int=42) -> Dataset:
    assert os.path.exists(data_path)
    assert os.path.exists(label_path)
    assert save_path is None or os.path.exists(os.path.dirname(save_path))

    with open(data_path, "r") as f:
        data_file = json.load(f)
    data_list = data_file["data_list"]

    with open(label_path, "r") as f:
        label2text = json.load(f)

    def create_flat_data(prompt:str, qa_item:dict, is_privacy:bool, image_path: str, p_label:str) -> dict:
        message = deepcopy(MESSAGE_TEMPLATE)
        message[0]["content"][1]["text"] = prompt
        if is_privacy:
            message[1]["content"][0]["text"] = qa_item["reject_ans"]
        else:
            if p_label in ABS_PRIV_LABELS:
                message[1]["content"][0]["text"] = qa_item["reject_ans"]
            else:
                message[1]["content"][0]["text"] = qa_item["respond_ans"]
        return {"messages": message, "images": image_path, "p_labels": p_label}

    flat_data_list = []
    for data_item in data_list:
        label = data_item['label']
        label_text = label2text[data_item['label']]
        image_path = os.path.join(image_folder, data_item['question_item']['image_path'])
        question = data_item['question_item']['question']

        if data_item.get("instruct_templates") is not None:
            instruct_templates = data_item["instruct_templates"]
        else:
            instruct_templates = INSTRUCT_TEMPLATES
        
        if add_distractors:
            instructions = [[] for _ in range(len(instruct_templates))]
            for other_label, is_privacy in data_item['other_labels'].items():
                for i in range(len(instruct_templates)):
                    if is_privacy != 0.5:
                        instructions[i].append(instruct_templates[is_privacy].format(label2text[other_label]))
                    elif is_privacy == 0.5:
                        instructions[i].append(instruct_templates[i].format(label2text[other_label]))
            instructions = ["\n".join(instruct) for instruct in instructions]
        else:
            instructions = [tmpl.format(label_text) for tmpl in instruct_templates]

        # 1
        flat_data_list.append(create_flat_data(f"{instructions[1]}\n\n{question}", data_item['question_item'], True, image_path, label))

        # 2
        flat_data_list.append(create_flat_data(f"{instructions[0]}\n\n{question}", data_item['question_item'], False, image_path, label))

        # 3
        for scenario in data_item['non_privacy_scenarios']:
            flat_data_list.append(create_flat_data(f"{instructions[1]}\n\n{scenario['2nd_person_template'].format(question)}", data_item['question_item'], False, image_path, label))

        # 4
        for scenario in data_item['privacy_scenarios']:
            flat_data_list.append(create_flat_data(f"{instructions[1]}\n\n{scenario['2nd_person_template'].format(question)}", data_item['question_item'], True, image_path, label))

        # 5
        for scenario in data_item['chosen_scenarios']:
            flat_data_list.append(create_flat_data(f"{instructions[0]}\n\n{scenario['2nd_person_template'].format(question)}", data_item['question_item'], False, image_path, label))

    if shuffle:
        random.seed(seed)
        random.shuffle(flat_data_list)
    
    if save_path is not None:
        with open(save_path, "w") as f:
            for item in flat_data_list:
                f.write(json.dumps(item) + "\n")
        return None
    else:
        data_dict = {
            "messages": [item["messages"] for item in flat_data_list],
            "images": [item["images"] for item in flat_data_list], 
            "p_labels": [item["p_labels"] for item in flat_data_list]
        }
        dataset_hf = Dataset.from_dict(data_dict).cast_column("images", Image())
        return dataset_hf
    

def prepare_preference_dataset(data_path:str, label_path:str, image_folder:str, add_distractors:bool, save_path:str=None, shuffle:bool=False, seed:int=42) -> Dataset:
    assert os.path.exists(data_path)
    assert os.path.exists(label_path)
    assert save_path is None or os.path.exists(os.path.dirname(save_path))

    with open(data_path, "r") as f:
        data_file = json.load(f)
    data_list = data_file["data_list"]

    with open(label_path, "r") as f:
        label2text = json.load(f)

    def create_flat_data(prompt_str:str, qa_item:dict, is_privacy:bool, image_path: str, p_label:str) -> dict:
        prompt = [deepcopy(MESSAGE_TEMPLATE[0])]
        prompt[0]["content"][1]["text"] = prompt_str
        chosen, rejected = [deepcopy(MESSAGE_TEMPLATE[1])], [deepcopy(MESSAGE_TEMPLATE[1])]
        if is_privacy:
            chosen[0]["content"][0]["text"] = qa_item['reject_ans']
            rejected[0]["content"][0]["text"] = qa_item['respond_ans']
        else:
            if p_label in ABS_PRIV_LABELS:
                chosen[0]["content"][0]["text"] = qa_item['reject_ans']
                rejected[0]["content"][0]["text"] = qa_item['respond_ans']
            else:
                chosen[0]["content"][0]["text"] = qa_item['respond_ans']
                rejected[0]["content"][0]["text"] = qa_item['reject_ans']
        return {"prompt": prompt, "chosen": chosen, "rejected": rejected, "images": image_path, "p_labels": p_label}

    flat_data_list = []
    for data_item in data_list:
        label = data_item['label']
        label_text = label2text[data_item['label']]
        image_path = os.path.join(image_folder, data_item['question_item']['image_path'])
        question = data_item['question_item']['question']

        if data_item.get("instruct_templates") is not None:
            instruct_templates = data_item["instruct_templates"]
        else:
            instruct_templates = INSTRUCT_TEMPLATES

        if add_distractors:
            instructions = [[] for _ in range(len(instruct_templates))]
            for other_label, is_privacy in data_item['other_labels'].items():
                for i in range(len(instruct_templates)):
                    if is_privacy != 0.5:
                        instructions[i].append(instruct_templates[is_privacy].format(label2text[other_label]))
                    elif is_privacy == 0.5:
                        instructions[i].append(instruct_templates[i].format(label2text[other_label]))
            instructions = ["\n".join(instruct) for instruct in instructions]
        else:
            instructions = [tmpl.format(label_text) for tmpl in instruct_templates]
    
        # 1
        flat_data_list.append(create_flat_data(f"{instructions[1]}\n\n{question}", data_item['question_item'], True, image_path, label))

        # 2
        flat_data_list.append(create_flat_data(f"{instructions[0]}\n\n{question}", data_item['question_item'], False, image_path, label))

        # 3
        for scenario in data_item['non_privacy_scenarios']:
            flat_data_list.append(create_flat_data(f"{instructions[1]}\n\n{scenario['2nd_person_template'].format(question)}", data_item['question_item'], False, image_path, label))

        # 4
        for scenario in data_item['privacy_scenarios']:
            flat_data_list.append(create_flat_data(f"{instructions[1]}\n\n{scenario['2nd_person_template'].format(question)}", data_item['question_item'], True, image_path, label))

        # 5
        for scenario in data_item['chosen_scenarios']:
            flat_data_list.append(create_flat_data(f"{instructions[0]}\n\n{scenario['2nd_person_template'].format(question)}", data_item['question_item'], False, image_path, label))

    if shuffle:
        random.seed(seed)
        random.shuffle(flat_data_list)
    
    if save_path is not None:
        with open(save_path, "w") as f:
            for item in flat_data_list:
                f.write(json.dumps(item) + "\n")
        return None
    else:
        data_dict = {
            "prompt": [item["prompt"] for item in flat_data_list],
            "chosen": [item["chosen"] for item in flat_data_list],
            "rejected": [item["rejected"] for item in flat_data_list],
            "images": [item["images"] for item in flat_data_list], 
            "p_labels": [item["p_labels"] for item in flat_data_list]
        }
        dataset_hf = Dataset.from_dict(data_dict).cast_column("images", Image())
        return dataset_hf
    
def prepare_our_dataset(data_path:str, label_path:str, image_folder:str, add_distractors:bool, save_path:str=None, shuffle:bool=False, seed:int=42) -> Dataset:
    assert os.path.exists(data_path)
    assert os.path.exists(label_path)
    assert save_path is None or os.path.exists(os.path.dirname(save_path))

    with open(data_path, "r") as f:
        data_file = json.load(f)
    data_list = data_file["data_list"]

    with open(label_path, "r") as f:
        label2text = json.load(f)

    def create_flat_data(prompt_0_str:str, prompt_1_str:str, qa_item:dict, scenario_is_privacy:bool, image_path: str, p_label:str) -> dict:
        prompt_0 = [deepcopy(MESSAGE_TEMPLATE[0])]
        prompt_1 = [deepcopy(MESSAGE_TEMPLATE[0])]
        prompt_0[0]["content"][1]["text"] = prompt_0_str
        prompt_1[0]["content"][1]["text"] = prompt_1_str
        
        chosen_0 = [deepcopy(MESSAGE_TEMPLATE[1])]
        chosen_1 = [deepcopy(MESSAGE_TEMPLATE[1])]
        rejected_0 = [deepcopy(MESSAGE_TEMPLATE[1])]
        rejected_1 = [deepcopy(MESSAGE_TEMPLATE[1])]
        
        if p_label in ABS_PRIV_LABELS:
            chosen_0[0]["content"][0]["text"] = qa_item['reject_ans']
            chosen_1[0]["content"][0]["text"] = qa_item['reject_ans']
            rejected_0[0]["content"][0]["text"] = qa_item['respond_ans']
            rejected_1[0]["content"][0]["text"] = qa_item['respond_ans']
        else:
            if scenario_is_privacy == False:
                chosen_0[0]["content"][0]["text"] = qa_item['respond_ans']
                chosen_1[0]["content"][0]["text"] = qa_item['respond_ans']
                rejected_0[0]["content"][0]["text"] = qa_item['reject_ans']
                rejected_1[0]["content"][0]["text"] = qa_item['reject_ans']
            else:
                chosen_0[0]["content"][0]["text"] = qa_item['respond_ans']
                chosen_1[0]["content"][0]["text"] = qa_item['reject_ans']
                rejected_0[0]["content"][0]["text"] = qa_item['reject_ans']
                rejected_1[0]["content"][0]["text"] = qa_item['respond_ans']

        return {"prompt_0": prompt_0, "prompt_1": prompt_1, "chosen_0": chosen_0, "chosen_1": chosen_1, "rejected_0": rejected_0, "rejected_1": rejected_1, "images": image_path, "p_labels": p_label}

    flat_data_list = []
    for data_item in data_list:
        label = data_item['label']
        label_text = label2text[data_item['label']]
        image_path = os.path.join(image_folder, data_item['question_item']['image_path'])
        question = data_item['question_item']['question']

        if data_item.get("instruct_templates") is not None:
            instruct_templates = data_item["instruct_templates"]
        else:
            instruct_templates = INSTRUCT_TEMPLATES

        if add_distractors:
            instructions = [[] for _ in range(len(instruct_templates))]
            for other_label, is_privacy in data_item['other_labels'].items():
                for i in range(len(instruct_templates)):
                    if is_privacy != 0.5:
                        instructions[i].append(instruct_templates[is_privacy].format(label2text[other_label]))
                    elif is_privacy == 0.5:
                        instructions[i].append(instruct_templates[i].format(label2text[other_label]))
            instructions = ["\n".join(instruct) for instruct in instructions]
        else:
            instructions = [tmpl.format(label_text) for tmpl in instruct_templates]

        flat_data_list.append(create_flat_data(f"{instructions[0]}\n\n{question}", f"{instructions[1]}\n\n{question}", data_item['question_item'], None, image_path, label))

        for scenario in data_item['non_privacy_scenarios']:
            prompt_0_str = f"{instructions[0]}\n\n{scenario['2nd_person_template'].format(question)}"
            prompt_1_str = f"{instructions[1]}\n\n{scenario['2nd_person_template'].format(question)}"
            flat_data_list.append(create_flat_data(prompt_0_str, prompt_1_str, data_item['question_item'], False, image_path, label))

        for scenario in data_item['privacy_scenarios']:
            prompt_0_str = f"{instructions[0]}\n\n{scenario['2nd_person_template'].format(question)}"
            prompt_1_str = f"{instructions[1]}\n\n{scenario['2nd_person_template'].format(question)}"
            flat_data_list.append(create_flat_data(prompt_0_str, prompt_1_str, data_item['question_item'], True, image_path, label))
    
    if shuffle:
        random.seed(seed)
        random.shuffle(flat_data_list)
    
    if save_path is not None:
        with open(save_path, "w") as f:
            for item in flat_data_list:
                f.write(json.dumps(item) + "\n")
        return None
    else:
        data_dict = {
            "prompt_0": [item["prompt_0"] for item in flat_data_list],
            "prompt_1": [item["prompt_1"] for item in flat_data_list],
            "chosen_0": [item["chosen_0"] for item in flat_data_list],
            "chosen_1": [item["chosen_1"] for item in flat_data_list],
            "rejected_0": [item["rejected_0"] for item in flat_data_list],
            "rejected_1": [item["rejected_1"] for item in flat_data_list],
            "images": [item["images"] for item in flat_data_list], 
            "p_labels": [item["p_labels"] for item in flat_data_list]
        }
        dataset_hf = Dataset.from_dict(data_dict).cast_column("images", Image())
        return dataset_hf