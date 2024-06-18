import os, json

def get_subjects_prompts_info(info_file_path):
    with open(info_file_path, 'r') as file:
        data_info = json.load(file)["subjects"]
    
    prompts = {"prompt_object": data_info["prompt_object"], 
               "prompt_live": data_info["prompt_live"]}
    subjects = data_info["subject_with_cls"].keys()
    prompts_info = dict()

    for subject in subjects:
        if data_info["subject_with_cls"][subject] in ["cat", "dog"]:
            prompts_info[subject] = {"class": data_info["subject_with_cls"][subject], "prompt_type": "prompt_live"}
        else:
            prompts_info[subject] = {"class": data_info["subject_with_cls"][subject], "prompt_type": "prompt_object"}
    
    return prompts_info, prompts