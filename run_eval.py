import os
import io
import numpy as np
import glob
from PIL import Image
import base64
from openai import OpenAI
import google.generativeai as genai
import pandas as pd


from eval import (
    eval_frame_level_progress_prediction,
    eval_frame_level_reasoning,
    eval_video_qa
)



# Download the dataset linked on the GitHub page and set the dataset_dir variable to the path where the dataset is stored
dataset_dir = "/path/to/data/"

# set api key
API_KEY = ...


method = 'rover'
# method = 'gvl'

model_name="gemini-1.5-pro"
# model_name="gpt-4o"

llm_evaluator_eval2_model_name=model_name
llm_evaluator_eval3_model_name=model_name


task="PnPCounterToCab"
# task="OpenSingleDoor"
# ...


if 'gpt' in model_name:
    client = OpenAI(api_key=API_KEY)
elif 'gemini' in model_name:
    genai.configure(api_key=API_KEY)
    google_model = genai.GenerativeModel(model_name = model_name)



camera_view = "wrist view"
# camera_view = "front view"
# camera_view = "right side view"
# camera_view = "left side view"


with open('prompt/prompting.json', 'r') as json_file:
    prompts_dict = json.load(json_file)


prompt_image0_append_gvl, prompt_image_current_prepend_gvl, prompt_image_prev_prepend_gvl, prompt_image0_append, prompt_image_current_prepend, prompt_image_prev_prepend, prompt_image_prev_append_template, prompt0_template, task_prepend, prompt0_template_gvl, decomposition_examples = \
     (prompts_dict[k] for k in ["prompt_image0_append_gvl", "prompt_image_current_prepend_gvl", "prompt_image_prev_prepend_gvl", "prompt_image0_append", "prompt_image_current_prepend", "prompt_image_prev_prepend", "prompt_image_prev_append_template", "prompt0_template", "task_prepend", "prompt0_template_gvl", "decomposition_examples"])


def encode_image(image_path):
    with Image.open(image_path) as img:
        width, height = img.size
        left_half = img.crop((0, 0, width // 2, height))  
        buffered = io.BytesIO()
        left_half.save(buffered, format=img.format)
        buffered.seek(0)
        return base64.b64encode(buffered.read()).decode('utf-8')


def flatten(nested):
    for item in nested:
        if isinstance(item, list):
            yield from flatten(item)
        else:
            yield item


def rover(task_description_i, frame_file_list, current_start_idx=0, subtask_completion_threshold=100, include_last_subtask_frame=False):
    current_idx = current_start_idx+1
    prompt0 = prompt0_template.format(task_description=task_description_i, decomposition_examples=decomposition_examples, camera_view=camera_view.upper())
    base64_image0 = encode_image(frame_file_list[current_start_idx])
    base64_image_current = encode_image(frame_file_list[current_idx])
    base64_image_prev=None
    prev_progress=None
    progress_list=[]
    frame_description_list=[]
    response_text_list=[]
    subtask_list=[]
    subtask_progress_list = []
    subtask_frame_descriptions_list = []
    subtask_idx = -1
    if "gpt" in model_name:
        messages_content = [
                        {"type": "text", "text": prompt0},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image0}", "detail": "high"}},
                        {"type": "text", "text": prompt_image0_append},
                        {"type": "text", "text": prompt_image_current_prepend},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image_current}", "detail": "high"}}
                    ]
    elif "gemini" in model_name:
        messages_content = [
                        prompt0,
                        {'mime_type':'image/jpeg', 'data': base64_image0},
                        prompt_image0_append,
                        prompt_image_current_prepend,
                        {'mime_type':'image/jpeg', 'data': base64_image_current}
                    ]
    while True:
        # 
        if current_idx >= len(frame_file_list):
            return current_idx, subtask_list, subtask_progress_list, subtask_frame_descriptions_list, base64_image_current
        # 
        if "gpt" in model_name:
            response = client.chat.completions.create(temperature=0.0,model=model_name,messages=[{"role": "user","content": messages_content}])
            response_text=response.choices[0].message.content
        elif "gemini" in model_name:
            response = google_model.generate_content(messages_content, generation_config=genai.GenerationConfig(max_output_tokens=100,temperature=0,top_k=1))
            response = response.to_dict()
            response_text = response['candidates'][0]['content']['parts'][0]['text']
        response_text_list.append(response_text)
        print('')
        print(current_idx)
        if 'completion percentage: ' in response_text:
            current_frame_description = response_text.split('Frame description: ')[1].split('\n')[0]
            current_idx += 1
            current_progress = response_text.split('completion percentage: ')[1].split('%')[0]
            current_progress = int(current_progress)
            frame_description_list.append(current_frame_description)
            progress_list.append(current_progress)
            print(current_frame_description)
            print(current_progress)
            if current_progress < subtask_completion_threshold and current_idx < len(frame_file_list):
                if not current_progress== prev_progress:
                    prev_progress = current_progress
                    base64_image_prev = base64_image_current
                # 
                base64_image_current = encode_image(frame_file_list[current_idx])
                if "gpt" in model_name:
                    messages_content = [
                                {"type": "text", "text": prompt0.split('IF THE GIVEN SUBTASK')[0] + prompt0.split('IF THE SUBTASK IS NOT DECOMPOSABLE (SEE EXAMPLES ABOVE), ')[-1]},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image0}", "detail": "high"}},
                                {"type": "text", "text": prompt_image0_append},
                                {"type": "text", "text": prompt_image_prev_prepend},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image_prev}", "detail": "high"}},
                                {"type": "text", "text": prompt_image_prev_append_template.format(prev_progress=prev_progress)},
                                {"type": "text", "text": prompt_image_current_prepend},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image_current}", "detail": "high"}}
                    ]
                elif "gemini" in model_name:
                    messages_content = [
                                    prompt0.split('IF THE GIVEN SUBTASK')[0] + prompt0.split('IF THE SUBTASK IS NOT DECOMPOSABLE (SEE EXAMPLES ABOVE), ')[-1],
                                    {'mime_type':'image/jpeg', 'data': base64_image0},
                                    prompt_image0_append,
                                    prompt_image_prev_prepend,
                                    {'mime_type':'image/jpeg', 'data': base64_image_prev},
                                    prompt_image_prev_append_template.format(prev_progress=prev_progress),
                                    prompt_image_current_prepend,
                                    {'mime_type':'image/jpeg', 'data': base64_image_current}
                                ]
            else:
                print(progress_list)
                subtask_progress_list = progress_list
                subtask_frame_descriptions_list = frame_description_list
                return current_idx, subtask_list, subtask_progress_list, subtask_frame_descriptions_list, base64_image_current
        elif 'robot needs to' in response_text:
            if 'New subtasks: ' in response_text and len(subtask_list)==0:
                new_subtask_list = response_text.split('New subtasks: ')[1].split('\n')[0].replace("'", "").replace("[", "").replace("]", "").strip().split(', ')
                new_subtask_list = [[x] for x in new_subtask_list]
                new_subtask_progress_list = [[] for x in new_subtask_list]
                new_subtask_frame_descriptions_list = [[] for x in new_subtask_list]
                subtask_list = subtask_list + [new_subtask_list]
                subtask_progress_list = subtask_progress_list + [new_subtask_progress_list]
                subtask_frame_descriptions_list = subtask_frame_descriptions_list + [new_subtask_frame_descriptions_list]
                print('subtask_list = subtask_list + [new_subtask_list]')
                print(subtask_list)
                print(subtask_progress_list)
            # 
            new_subtask = response_text.split('robot needs to: ')[1].split('\n')[0].split('.')[0].lower()
            subtask_idx += 1
            if subtask_idx >= len(subtask_list[0]):
                subtask_list[0].append([new_subtask])
            # 
            current_idx, new_subtask_list2, new_subtask_progress_list2, new_subtask_frame_descriptions_list2, new_base64_image_current2  = rover(new_subtask, frame_file_list, current_idx-1)
            # 
            subtask_list[0][subtask_idx].append(new_subtask_list2)
            subtask_progress_list[0][subtask_idx] = new_subtask_progress_list2
            subtask_frame_descriptions_list[0][subtask_idx] = new_subtask_frame_descriptions_list2
            #
            print('subtask_list[0][subtask_idx].append(new_subtask_list2)')
            print(subtask_list)
            print(subtask_progress_list)
            # 
            if current_idx < len(frame_file_list):
                # 
                if "gpt" in model_name:
                    messages_content = [{"type": "text", "text": task_prepend}]
                    for i in range(len(response_text_list)):
                        messages_content.append({"type": "text", "text": response_text_list[i].replace('The robot needs to: ', 'The robot has completed: ')})
                    # 
                    if include_last_subtask_frame:
                        messages_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{new_base64_image_current2}", "detail": "high"}},)
                elif "gemini" in model_name:
                    messages_content = [task_prepend]
                    for i in range(len(response_text_list)):
                        messages_content.append(response_text_list[i].replace('The robot needs to: ', 'The robot has completed: '))
                    if include_last_subtask_frame:
                        messages_content.append({'mime_type':'image/jpeg', 'data': new_base64_image_current2})
            else:
                return current_idx, subtask_list, subtask_progress_list, subtask_frame_descriptions_list, base64_image_current
        elif 'task complete' in response_text.lower():
            print('Task complete')
            return current_idx, subtask_list, subtask_progress_list, subtask_frame_descriptions_list, base64_image_current




def gvl(task_description_i, frame_file_list, image_file_num_list, try_count_max=4):
    #
    prompt0 = prompt0_template_gvl.format(task_description=task_description_i, camera_view=camera_view.upper())
    base64_image0 = encode_image(frame_file_list[0])
    image_file_num_list_idx_shuffled = list(range(len(image_file_num_list)-1))
    image_file_num_list_idx_shuffled = np.random.permutation(image_file_num_list_idx_shuffled).tolist()
    # 
    current_progress = 0
    current_frame_description = None
    progress_list = []
    frame_descriptions_list = []
    response_text_list=[]
    response_failed = False
    for current_idx in range(len(image_file_num_list_idx_shuffled)):
        base64_image_current = encode_image(frame_file_list[image_file_num_list_idx_shuffled[current_idx]])        
        # 
        try_count=0
        while try_count < try_count_max and not response_failed:
            try:
                if "gpt" in model_name:
                    if current_idx==0:
                        messages_content = [
                                        {"type": "text", "text": prompt0},
                                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image0}", "detail": "high"}},
                                        {"type": "text", "text": prompt_image0_append_gvl.format(current_progress=current_progress)},
                                        {"type": "text", "text": prompt_image_current_prepend_gvl.format(current_idx=current_idx+1)},
                                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image_current}", "detail": "high"}}
                                    ]
                    else:
                        messages_content = messages_content + [
                                        {"type": "text", "text": prompt_image_prev_prepend_gvl.format(current_progress=current_progress, current_frame_description=current_frame_description)},
                                        {"type": "text", "text": prompt_image_current_prepend_gvl.format(current_idx=current_idx+1)},
                                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image_current}", "detail": "high"}}
                                    ]
                    # 
                    response = client.chat.completions.create(temperature=0.0,model=model_name,messages=[{"role": "user","content": messages_content}])
                    response_text=response.choices[0].message.content
                elif "gemini" in model_name:
                    if current_idx==0:
                        messages_content = [
                                        prompt0,
                                        {'mime_type':'image/jpeg', 'data': base64_image0},
                                        prompt_image0_append_gvl.format(current_progress=current_progress),
                                        prompt_image_current_prepend_gvl.format(current_idx=current_idx+1),
                                        {'mime_type':'image/jpeg', 'data': base64_image_current}
                                    ]
                    else:
                        messages_content = messages_content + [
                                        prompt_image_prev_prepend_gvl.format(current_progress=current_progress, current_frame_description=current_frame_description),
                                        prompt_image_current_prepend_gvl.format(current_idx=current_idx+1),
                                        {'mime_type':'image/jpeg', 'data': base64_image_current}
                                    ]
                    response = google_model.generate_content(messages_content, generation_config=genai.GenerationConfig(max_output_tokens=100,temperature=0,top_k=1))
                    response = response.to_dict()
                    response_text = response['candidates'][0]['content']['parts'][0]['text']
                # 
                # 
                current_frame_description = response_text.split('Frame description: ')[1].split('\n')[0]
                current_progress = response_text.split('completion percentage: ')[1].split('%')[0]
                current_progress = int(current_progress)
                # 
                response_text_list.append(response_text)
                frame_descriptions_list.append(current_frame_description)
                progress_list.append(current_progress)
                break
            except Exception as e:
                print(f"Error: {e}")
                try_count += 1
                print(f'Response: {response_text}')
        if try_count >= try_count_max:
            response_failed = True
            response_text_list.append('')
            frame_descriptions_list.append('') 
            progress_list.append(0)
        # 
        print(current_idx)
        print(image_file_num_list_idx_shuffled[current_idx])
        print(frame_descriptions_list[-1])
        print(progress_list[-1])
    # 
    return progress_list, frame_descriptions_list, image_file_num_list_idx_shuffled 



def process_gvl_output(progress_list, frame_descriptions_list, image_file_num_list_idx_shuffled):
    final_progress_list = []
    frame_descriptions_list = []
    for i in range(len(image_file_num_list_idx_shuffled)):
        final_progress_list.append(progress_list[image_file_num_list_idx_shuffled.index(i)])
        frame_descriptions_list.append(frame_descriptions_list[image_file_num_list_idx_shuffled.index(i)])
    # 
    while len(final_progress_list) < len(frame_file_list)-1:
        final_progress_list.append(final_progress_list[-1])
    # 
    while len(frame_descriptions_list) < len(frame_file_list)-1:
        frame_descriptions_list.append(frame_descriptions_list[-1])
    # 
    return final_progress_list, frame_descriptions_list


def process_rover_output(subtask_list, subtask_progress_list, subtask_frame_descriptions_list):
    final_progress_list = depth_first_traverse_rover_output_structure(subtask_list[0], subtask_progress_list[0], len(subtask_list[0]))
    while len(final_progress_list) < len(frame_file_list)-1:
        final_progress_list.append(final_progress_list[-1])
    frame_descriptions_list= list(flatten(subtask_frame_descriptions_list))
    while len(final_progress_list) < len(frame_file_list)-1:
        final_progress_list.append(final_progress_list[-1])
    # 
    while len(frame_descriptions_list) < len(frame_file_list)-1:
        frame_descriptions_list.append(frame_descriptions_list[-1])
    return final_progress_list, frame_descriptions_list


def depth_first_traverse_rover_output_structure(subtask_list_i, subtask_progress_list_i, subtask_count, subtask_subtask_idx=None, layer=0):
    indent_str = '----'*2*(layer+1) + ' '
    layer_tag = f'\n{indent_str}layer - {layer}'
    # print(layer_tag)
    # print(indent_str + str(subtask_list_i))
    # print(indent_str + str(subtask_progress_list_i))
    adjusted_progress_list=[]
    for subtask_idx in range(len(subtask_list_i)):
        print(layer_tag)
        print(indent_str + str(subtask_list_i[subtask_idx][0]))
        print(indent_str + f'subtask_count: {subtask_count}')
        if subtask_subtask_idx is None:
            progress_val_start= (subtask_idx/subtask_count)*100
        else:
            progress_val_start= (subtask_subtask_idx/subtask_count)*100
        print(indent_str + f'progress_val_start: {progress_val_start}')
        if len(subtask_list_i[subtask_idx])>1:
            if len(subtask_list_i[subtask_idx][1])>0:
                subtask_subtask_count = len(subtask_list_i[subtask_idx][1][0])
                for i in range(subtask_subtask_count):
                    if subtask_subtask_idx is None:
                        adjusted_progress_list2 = depth_first_traverse_rover_output_structure([subtask_list_i[subtask_idx][1][0][i]], subtask_progress_list_i[subtask_idx][0], subtask_subtask_count, subtask_subtask_idx=i, layer=layer+1)
                    else:
                        adjusted_progress_list2 = depth_first_traverse_rover_output_structure([subtask_list_i[subtask_idx][1][0][i]], subtask_progress_list_i[subtask_subtask_idx][0], subtask_subtask_count, subtask_subtask_idx=i, layer=layer+1)
                    # 
                    adjusted_progress_list = adjusted_progress_list + [progress_val_start + (x/subtask_count) for x in adjusted_progress_list2]
            else:
                print(indent_str + 'raw progress values for this subtask')
                if subtask_subtask_idx is None:
                    # print(indent_str + f'subtask_idx: {subtask_idx}')
                    
                    print(indent_str + str(subtask_progress_list_i[subtask_idx]))
                else:
                    # print(indent_str + f'subtask_subtask_idx: {subtask_subtask_idx}')
                    print(indent_str + str(subtask_progress_list_i[subtask_subtask_idx]))
                # 
                if subtask_subtask_idx is None:
                    adjusted_progress_list = adjusted_progress_list + [progress_val_start + (x/subtask_count) for x in subtask_progress_list_i[subtask_idx]]
                else:
                    adjusted_progress_list = adjusted_progress_list + [progress_val_start + (x/subtask_count) for x in subtask_progress_list_i[subtask_subtask_idx]]
                # 
        else:
            print(indent_str + 'raw progress values for this subtask')
            print(indent_str + str([]))
    # 
    # print(indent_str + 'here5')
    # print(indent_str + f'subtask_count: {subtask_count}')
    print(indent_str + 'scaled progress values for this subtask')
    print(indent_str + str(adjusted_progress_list))
    return adjusted_progress_list



def get_perturb_info(perturb_info_file):
    idx_start, idx_final, idx_start_contact, idx_contact, idx_start_contact_expert, idx_contact_expert, task_description, dist_list, step_label_list, gripper_target_dist_list, env_dist_list, obj_is_touching_gripper_list, obj_is_only_touching_gripper_list = None, None, None, None, None, None, '', [], [], [], [], [], []
    with open(perturb_info_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'idx_start:' in line:
                idx_start = int(line.split(':')[-1].strip())
                idx_start = idx_start
            elif 'idx_final' in line:
                idx_final = int(line.split(':')[-1].strip())
                idx_final = idx_final
            elif 'idx_start_contact:' in line:
                if not 'None' in line:
                    idx_start_contact = int(line.split(':')[-1].strip())
                else:
                    idx_start_contact = None
                idx_start_contact = idx_start_contact
            elif 'idx_contact:' in line:
                if not 'None' in line:
                    idx_contact = int(line.split(':')[-1].strip())
                else:
                    idx_contact = None
                idx_contact = idx_contact
            elif 'idx_start_contact_expert:' in line:
                idx_start_contact_expert = int(line.split(':')[-1].strip())
                idx_start_contact_expert = idx_start_contact_expert
            elif 'idx_contact_expert:' in line:
                idx_contact_expert = int(line.split(':')[-1].strip())
                idx_contact_expert = idx_contact_expert
            elif line.startswith("language"):
                # Extract the value after the colon and strip whitespace
                task_description = line.split(":")[1].strip().lower()
            elif line.startswith('dist_list:'):
                dist_list = line.split(': ')[-1].strip()
                dist_list = dist_list[1:-1].split(', ')
                dist_list = [float(x) for x in dist_list]
                dist_list = dist_list
            elif 'step_label' in line:
                step_label_list = line.split(': ')[-1].strip()
                step_label_list = step_label_list[1:-1].split(', ')
                step_label_list = [x[1:-1] for x in step_label_list]
                step_label_list = step_label_list
            elif 'gripper_target_dist_list' in line:
                gripper_target_dist_list = line.split(': ')[-1].strip()
                gripper_target_dist_list = gripper_target_dist_list[1:-1].split(', ')
                gripper_target_dist_list = [float(x) for x in gripper_target_dist_list]
                gripper_target_dist_list = gripper_target_dist_list
            elif 'env_dist_list' in line:
                env_dist_list = line.split(': ')[-1].strip()
                env_dist_list = env_dist_list[1:-1].split(', ')
                env_dist_list = [float(x) for x in env_dist_list]
                env_dist_list = env_dist_list
            elif 'obj_is_touching_gripper_list' in line:
                obj_is_touching_gripper_list = line.split(': ')[-1].strip()
                if not 'None' in obj_is_touching_gripper_list:
                    obj_is_touching_gripper_list = obj_is_touching_gripper_list[1:-1].split(', ')
                    obj_is_touching_gripper_list = obj_is_touching_gripper_list
                else:
                    obj_is_touching_gripper_list=None
            elif 'obj_is_only_touching_gripper_list' in line:
                obj_is_only_touching_gripper_list = line.split(': ')[-1].strip()
                if not 'None' in obj_is_only_touching_gripper_list:
                    obj_is_only_touching_gripper_list = obj_is_only_touching_gripper_list[1:-1].split(', ')
                    obj_is_only_touching_gripper_list = obj_is_only_touching_gripper_list
                else:
                    obj_is_only_touching_gripper_list=None
    return idx_start, idx_final, idx_start_contact, idx_contact, idx_start_contact_expert, idx_contact_expert, task_description, dist_list, step_label_list, gripper_target_dist_list, env_dist_list, obj_is_touching_gripper_list, obj_is_only_touching_gripper_list



episode_dir_list = glob.glob(f"{dataset_dir}/{task}/*")
episode_dir_list = [x for x in episode_dir_list if not x.endswith('.mp4')]

len(episode_dir_list)
downsample_to = 30

#### collect results for task 1
progress_corr_list = []
progress_dist_list = []
#### collect results for task 2
reasoning_error_rate_list = []
reasoning_success_rate_list = []
reasoning_inconclusive_rate_list = []
#### collect results for task 3
qa_accuracy_list = []
qa_precision_list = []
qa_recall_list = []
qa_frame_diff_list = []

if method == 'rover':
    final_idx_list = []
    subtask_list_list = []
    subtask_progress_list_list = []
    subtask_frame_descriptions_list_list = []


gt_progress_list_list = []
final_progress_list_list=[]
frame_descriptions_list_list = []
for episode_dir_idx in range(0,len(episode_dir_list[:2])):
    episode_dir = episode_dir_list[episode_dir_idx]
    #
    episode_dir = [x for x in episode_dir_list if 'lev6' in x][episode_dir_idx]
    print('\n\n\n\n\n*********************************')
    print(episode_dir)
    level_i = episode_dir.split('/')[-1].split('_')[-1].split('-')[0].split('.')[0]
    demo_num_i = int(episode_dir.split('/')[-1].split('_demo_')[-1].split('_')[0])
    # 
    # 
    ######################################################################################
    frame_file_list_all= glob.glob(f"{episode_dir}/frames/frame_*.jpg")
    len(frame_file_list_all)
    # downsample to 30 frames
    image_file_num_list_all = [x.split('/')[-1].split('_')[1].split('.')[0] for x in frame_file_list_all]
    image_file_num_list_all = [int(x) for x in image_file_num_list_all]
    image_file_num_list_all= sorted(image_file_num_list_all)
    # 
    image_file_num_list = np.linspace(0, image_file_num_list_all[-1], downsample_to).astype(int).tolist()
    frame_file_list = [f"{episode_dir}/frames/frame_{x}.jpg" for x in image_file_num_list]
    ######################################################################################
    gt_progress_file = f'{episode_dir}/task_progress.txt'
    gt_progress_list = []
    with open(gt_progress_file, 'r') as f:
        for line in f:
            if line.strip():
                gt_progress_list.append(float(line.strip()))
    # 
    gt_progress_list = [round(x) for x in gt_progress_list]
    gt_progress_list = [gt_progress_list[i] for i in image_file_num_list]
    # 
    len(frame_file_list)
    len(gt_progress_list)
    # 
    perturb_info_file = f'{episode_dir}/pertub_info.txt'
    idx_start_i, idx_final_i, idx_start_contact_i, idx_contact_i, idx_start_contact_expert_i, idx_contact_expert_i, task_description_i, dist_list_i, step_label_list_i, gripper_target_dist_list_i, env_dist_list_i, obj_is_touching_gripper_list_i, obj_is_only_touching_gripper_list_i = get_perturb_info(perturb_info_file)
    ######################################################################################
    # if True:
    if method == 'rover':
        final_idx, subtask_list, subtask_progress_list, subtask_frame_descriptions_list, _ = rover(task_description_i, frame_file_list)
        final_progress_list, frame_descriptions_list = process_rover_output(subtask_list, subtask_progress_list, subtask_frame_descriptions_list)
    elif method == 'gvl':
        progress_list, frame_descriptions_list, image_file_num_list_idx_shuffled = gvl(task_description_i, frame_file_list, image_file_num_list)
        final_progress_list, frame_descriptions_list = process_gvl_output(progress_list, frame_descriptions_list, image_file_num_list_idx_shuffled)
    len(final_progress_list), len(frame_descriptions_list)
    # adding 0 for the initial frame
    final_progress_list = [0] + final_progress_list
    # adding empty string for the initial frame
    frame_descriptions_list = [''] + frame_descriptions_list
    # 
    if method == 'rover':
        final_idx_list.append(final_idx)
        subtask_list_list.append(subtask_list)
        subtask_progress_list_list.append(subtask_progress_list)
        subtask_frame_descriptions_list_list.append(subtask_frame_descriptions_list)
    # 
    gt_progress_list_list.append(gt_progress_list)
    final_progress_list_list.append(final_progress_list)
    frame_descriptions_list_list.append(frame_descriptions_list)
    ######################################################################################
    # 
    corr, dist = eval_frame_level_progress_prediction(gt_progress_list, final_progress_list)
    progress_corr_list.append(corr)
    progress_dist_list.append(dist)
    # 
    error_rate, success_rate, inconclusive_rate = eval_frame_level_reasoning(frame_descriptions_list)
    reasoning_error_rate_list.append(error_rate)
    reasoning_success_rate_list.append(success_rate)
    reasoning_inconclusive_rate_list.append(inconclusive_rate)
    # 
    qa_accuracy, qa_precision, qa_recall, qa_frame_diff = eval_video_qa(frame_descriptions_list)
    qa_accuracy_list.append(qa_accuracy)
    qa_precision_list.append(qa_precision)
    qa_recall_list.append(qa_recall)
    qa_frame_diff_list.append(qa_frame_diff)



corr_list
dist_list
final_idx_list
subtask_list_list
subtask_progress_list_list
subtask_frame_descriptions_list_list
gt_progress_list_list
final_progress_list_list


res_df = pd.DataFrame({
    'episode_dir': [x.split('/')[-1] for x in episode_dir_list],
    'corr': progress_corr_list,
    'dist': progress_dist_list,
    'reasoning_error_rate': reasoning_error_rate_list,
    'reasoning_success_rate': reasoning_success_rate_list,
    'reasoning_inconclusive_rate': reasoning_inconclusive_rate_list,
    'qa_accuracy': qa_accuracy_list,
    'qa_precision': qa_precision_list,
    'qa_recall': qa_recall_list,
    'qa_frame_diff': qa_frame_diff_list
})


print(res_df)
