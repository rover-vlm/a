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

from rover_model import (
    rover,
    process_rover_output,
)

from gvl_model import (
    gvl,
    process_gvl_output,
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
    if method == 'rover':
        final_idx, subtask_list, subtask_progress_list, subtask_frame_descriptions_list, _ = rover(model_name, task_description_i, camera_view, frame_file_list)
        final_progress_list, frame_descriptions_list = process_rover_output(subtask_list, subtask_progress_list, subtask_frame_descriptions_list)
    elif method == 'gvl':
        progress_list, frame_descriptions_list, image_file_num_list_idx_shuffled = gvl(model_name, task_description_i, camera_view, frame_file_list, image_file_num_list)
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