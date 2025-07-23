import os
import io
import numpy as np
import glob
from PIL import Image
import base64
from openai import OpenAI
import google.generativeai as genai


################ Eval of frame-level progress prediction 

def eval_frame_level_progress_prediction(gt_progress_list, final_progress_list):
    # pearson correlation
    corr=np.corrcoef(final_progress_list, gt_progress_list)[0, 1]
    # l2 distance
    dist= np.linalg.norm(np.array(final_progress_list) - np.array(gt_progress_list))
    return corr, dist



################ Eval of frame-level reasoning 


def eval_frame_level_reasoning(frame_descriptions_list):
    response_text_yes_no_numb_list = []
    for i in range(len(image_file_num_list)):
        ground_truth_state_info = get_state_annotation(idx_start_i, image_file_num_list[i], task_description_i, level_i, task, idx_start_contact_i, idx_contact_i, step_label_list_i, image_file_num_list_all, gripper_target_dist_list_i, env_dist_list_i, obj_is_touching_gripper_list_i, obj_is_only_touching_gripper_list_i)
        current_frame_description = frame_descriptions_list[i]
        if not current_frame_description=='':
            question = llm_evaluator_eval2_question_template.format(task_description=task_description_i)
            prompt =  question + '\n\n\nGround truth information about frame:\n' + ground_truth_state_info + '\n\n\nFrame description:\n' + current_frame_description
            print(prompt)
            if "gpt" in llm_evaluator_eval2_model_name:
                response = client.chat.completions.create(temperature=0.0,model=model_name,messages=[{"role": "user","content": [{"type": "text", "text": prompt}]}])
                response_text=response.choices[0].message.content
            elif "gemini" in llm_evaluator_eval2_model_name:
                google_model = genai.GenerativeModel(model_name = model_name, system_instruction = 'You are a helpful AI assistant.')
                response = google_model.generate_content( [prompt], generation_config = genai.GenerationConfig(max_output_tokens=100,temperature=0,top_k=1))
                response = response.to_dict()
                response_text = response['candidates'][0]['content']['parts'][0]['text']
            # 
            response_text_yes_no = response_text.split('Final Answer: ')[-1].strip().lower() 
            if 'false' in response_text_yes_no:
                response_text_yes_no_numb = -1
            elif 'true' in response_text_yes_no:
                response_text_yes_no_numb = 1
            else:
                response_text_yes_no_numb = 0
            #
            print('response_text')
            print(response_text)
        else:
            response_text_yes_no_numb = -1
        response_text_yes_no_numb_list.append(response_text_yes_no_numb)  
        print(response_text_yes_no_numb_list)
    # 
    error_rate = sum([1 for x in response_text_yes_no_numb_list if x == -1]) / len(response_text_yes_no_numb_list)
    success_rate = sum([1 for x in response_text_yes_no_numb_list if x == 1]) / len(response_text_yes_no_numb_list)
    inconclusive_rate = sum([1 for x in response_text_yes_no_numb_list if x == 0]) / len(response_text_yes_no_numb_list)
    # 
    return error_rate, success_rate, inconclusive_rate


llm_evaluator_eval2_question_template = "A robot was given the task of '{task_description}'. We capture a video of the robot attempting the task. Below is the description of a frame within that video, along with ground truth information about the robot and environment state at that frame. Given the ground truth information, your task is to determine the accuracy of the frame description. Please classify the frame description as True, False, or Inconclusive. Please end your response with 'Final Answer: {{final_classification}}'."


def get_obj_name(task_description, task):
    if 'PnP' in task or task in ['CoffeeSetupMug', 'CoffeeServeMug', 'MicrowaveThawing-2', 'MicrowaveThawing']:
        obj_name = task_description.lower().split('pick the ')[-1].split(' from')[0]
    elif task in ['OpenSingleDoor', 'OpenDrawer'] or task in ['CloseSingleDoor', 'CloseDrawer']:
        if 'drawer' in task_description:
            obj_name = 'drawer'
        else:
            if 'microwave' in task_description:
                obj_name = 'microwave door'
            elif 'cabinet' in task_description:
                obj_name = 'cabinet door'
            else:
                obj_name = 'door'
    elif task in ['TurnOnSinkFaucet', 'TurnOffSinkFaucet', 'PreSoakPan-3']:
        obj_name = 'sink handle'
    elif task in ['TurnSinkSpout']:
        obj_name = 'sink spout'
    elif task in ['TurnOnStove', 'TurnOffStove']:
        obj_name = 'stove knob'
    elif task in ['TurnOnMicrowave', 'TurnOffMicrowave']:
        if 'start' in task_description:
            obj_name = 'microwave start button'
        elif 'stop' in task_description:
            obj_name = 'microwave stop button'
    elif task in ['PrepareCoffee-1']:
        obj_name = 'coffee mug'
    elif task in ['PreSoakPan-1']:
        obj_name = 'pan'
    elif task in ['PreSoakPan-2']:
        obj_name = 'sponge'
    elif task in ['MicrowaveThawing-1', 'MicrowaveThawing-3']:
        obj_name = 'microwave door'
    elif task in ['MicrowaveThawing-4']:
        obj_name = 'microwave start button'
    elif task in ['CoffeePressButton', 'PrepareCoffee-2']:
        obj_name = 'coffee machine start button'
    return obj_name



def get_state_annotation(idx_start, current_idx, task_description, level_i, task, idx_start_contact_i, idx_contact_i, step_label_list_i, frame_idx_list_i, gripper_target_dist_list_i, env_dist_list_i, obj_is_touching_gripper_list_i, obj_is_only_touching_gripper_list_i):
    # frame_idx_list_i = sorted(frame_idx_list_i)
    # 
    current_step_label = step_label_list_i[current_idx]
    # 
    # current_frame_idx = frame_idx_list_i[current_idx] - idx_start
    current_frame_idx = frame_idx_list_i[current_idx]
    # 
    current_frame_gripper_target_dist = gripper_target_dist_list_i[current_frame_idx]
    current_frame_env_dist = env_dist_list_i[current_frame_idx]
    current_frame_obj_is_touching_gripper = obj_is_touching_gripper_list_i[current_frame_idx]
    if not task.split('-')[0] in ['MicrowaveThawing', 'RestockPantry', 'ArrangeVegetables', 'PrepareCoffee', 'PreSoakPan']:
        if not obj_is_only_touching_gripper_list_i is None and len(obj_is_only_touching_gripper_list_i) > 0:
            current_frame_obj_is_only_touching_gripper = obj_is_only_touching_gripper_list_i[current_frame_idx]
        else:
            current_frame_obj_is_only_touching_gripper = None
    # 
    # previous_frame_idx = frame_idx_list_i[current_idx-1] - idx_start
    previous_frame_idx = frame_idx_list_i[current_idx-1] 
    # 
    previous_frame_gripper_target_dist = gripper_target_dist_list_i[previous_frame_idx]
    previous_frame_env_dist = env_dist_list_i[previous_frame_idx]
    previous_frame_obj_is_touching_gripper = obj_is_touching_gripper_list_i[previous_frame_idx]
    if not task.split('-')[0] in ['MicrowaveThawing', 'RestockPantry', 'ArrangeVegetables', 'PrepareCoffee', 'PreSoakPan']:
        if not obj_is_only_touching_gripper_list_i is None and len(obj_is_only_touching_gripper_list_i) > 0:
            previous_frame_obj_is_only_touching_gripper = obj_is_only_touching_gripper_list_i[previous_frame_idx]
        else:
            previous_frame_obj_is_only_touching_gripper = None
    # 
    annot_contact = None
    annot_contact_prev = None
    if task in ['MicrowaveThawing-1']:
        door_type = 'microwave door'
        obj = get_obj_name(task_description, 'MicrowaveThawing-2')
        if level_i in ['lev2', 'lev3', 'lev4', 'lev5']:
            annot_obj1 = f'The {door_type} is in the process of being opened.\nThe {obj} is still on the counter.'
        else:
            annot_obj1 = f'The {door_type} is closed.\nThe {obj} is still on the counter.'
        # 
        annot_final = f'{annot_obj1}\n'
        if current_frame_gripper_target_dist < previous_frame_gripper_target_dist:
            annot_gripper1 = f'The robot gripper is moving towards the {door_type} handle.'
            annot_final = annot_final + f'{annot_gripper1}\n'
        elif current_frame_gripper_target_dist > previous_frame_gripper_target_dist:
            annot_gripper1 = f'The robot gripper is moving away from the {door_type} handle.'
            annot_final = annot_final + f'{annot_gripper1}\n'
    elif task in ['MicrowaveThawing-2']:
        door_type = 'microwave door'
        obj = get_obj_name(task_description, 'MicrowaveThawing-2')
        if level_i in ['lev3', 'lev4', 'lev5']:
            annot_obj1 = f'The {door_type} has been opened.\nThe {obj} is in the process of being put in the microwave.'
        else:
            annot_obj1 = f'The {door_type} has been opened.\nThe {obj} is still on the counter.'
        # 
        annot_final = f'{annot_obj1}\n'
        if current_frame_gripper_target_dist < previous_frame_gripper_target_dist:
            annot_gripper1 = f'The robot gripper is moving towards the {obj}.'
            annot_final = annot_final + f'{annot_gripper1}\n'
        elif current_frame_gripper_target_dist > previous_frame_gripper_target_dist:
            annot_gripper1 = f'The robot gripper is moving away from the {obj}.'
            annot_final = annot_final + f'{annot_gripper1}\n'
    elif task in ['MicrowaveThawing-3']:
        obj = get_obj_name(task_description, 'MicrowaveThawing-2')
        door_type = 'microwave door'
        if level_i in ['lev4', 'lev5']:
            annot_obj1 = f'The {door_type} has been opened.\nThe {obj} has been put in the microwave.\nThe {door_type} is in the process of being closed.'
        else:
            annot_obj1 = f'The {door_type} has been opened.\nThe {obj} has been put in the microwave.\nThe {door_type} is still open.'
        # 
        annot_final = f'{annot_obj1}\n'
        if current_frame_gripper_target_dist < previous_frame_gripper_target_dist:
            annot_gripper1 = f'The robot gripper is moving towards the {door_type}.'
            annot_final = annot_final + f'{annot_gripper1}\n'
        elif current_frame_gripper_target_dist > previous_frame_gripper_target_dist:
            annot_gripper1 = f'The robot gripper is moving away from the {door_type}.'
            annot_final = annot_final + f'{annot_gripper1}\n'
    elif task in ['MicrowaveThawing-4']:
        obj = get_obj_name(task_description, 'MicrowaveThawing-2')
        door_type = 'microwave door'
        if level_i in ['lev5']:
            annot_obj1 = f'The {door_type} has been opened.\nThe {obj} has been put in the microwave.\nThe {door_type} has been closed.\nThe robot is in the process of pressing the microwave start button.'
        else:
            annot_obj1 = f'The {door_type} has been opened.\nThe {obj} has been put in the microwave.\nThe {door_type} has been closed.\nThe robot has still not pressed the microwave start button.'
        # 
        annot_final = f'{annot_obj1}\n'
        if current_frame_gripper_target_dist < previous_frame_gripper_target_dist:
            annot_gripper1 = f'The robot gripper is moving towards the microwave start button.'
            annot_final = annot_final + f'{annot_gripper1}\n'
        elif current_frame_gripper_target_dist > previous_frame_gripper_target_dist:
            annot_gripper1 = f'The robot gripper is moving away from the microwave start button.'
            annot_final = annot_final + f'{annot_gripper1}\n'
    elif task in ['PreSoakPan-1']:
        if level_i in ['lev2', 'lev3', 'lev4']:
            annot_obj1 = f'The robot is in the process of putting the pan in the sink.\nThe sponge is still on the counter.\nThe sink is not turned on.'
        else:
            annot_obj1 = f'The pan is still on the counter.\nThe sponge is still on the counter.\nThe sink is not turned on.'
        # 
        annot_final = f'{annot_obj1}\n'
        if current_frame_gripper_target_dist < previous_frame_gripper_target_dist:
            annot_gripper1 = f'The robot gripper is moving towards the pan.'
            annot_final = annot_final + f'{annot_gripper1}\n'
        elif current_frame_gripper_target_dist > previous_frame_gripper_target_dist:
            annot_gripper1 = f'The robot gripper is moving away from the pan.'
            annot_final = annot_final + f'{annot_gripper1}\n'
    elif task in ['PreSoakPan-2']:
        if level_i in ['lev3', 'lev4']:
            annot_obj1 = f'The pan has been put in the sink.\nThe robot is in the process of putting the sponge in the pan (which is in the sink).\nThe sink is not turned on.'
        else:
            annot_obj1 = f'The pan has been put in the sink.\nThe sponge is still on the counter.\nThe sink is not turned on.'
        # 
        annot_final = f'{annot_obj1}\n'
        if current_frame_gripper_target_dist < previous_frame_gripper_target_dist:
            annot_gripper1 = f'The robot gripper is moving towards the sponge.'
            annot_final = annot_final + f'{annot_gripper1}\n'
        elif current_frame_gripper_target_dist > previous_frame_gripper_target_dist:
            annot_gripper1 = f'The robot gripper is moving away from the sponge.'
            annot_final = annot_final + f'{annot_gripper1}\n'
    elif task in ['PreSoakPan-3']:
        if level_i in ['lev4']:
            annot_obj1 = f'The pan has been put in the sink.\nThe sponge has been put in the pan (which is in the sink).\nThe robot is in the process of turning on the sink.'
        else:
            annot_obj1 = f'The pan has been put in the sink.\nThe sponge has been put in the pan (which is in the sink).\nThe sink is not turned on.'
        # 
        annot_final = f'{annot_obj1}\n'
        if current_frame_gripper_target_dist < previous_frame_gripper_target_dist:
            annot_gripper1 = f'The robot gripper is moving towards the sink handle.'
            annot_final = annot_final + f'{annot_gripper1}\n'
        elif current_frame_gripper_target_dist > previous_frame_gripper_target_dist:
            annot_gripper1 = f'The robot gripper is moving away from the sink handle.'
            annot_final = annot_final + f'{annot_gripper1}\n'
    elif task in ['PrepareCoffee-1']:
        if level_i in ['lev2', 'lev3']:
            annot_obj1 = f'The robot is in the process of putting the mug in the coffee machine.\nThe coffee machine is off.'
        else:
            annot_obj1 = f'The mug is in the cabinet.\nThe coffee machine is off.'
        # 
        annot_final = f'{annot_obj1}\n'
        if current_frame_gripper_target_dist < previous_frame_gripper_target_dist:
            annot_gripper1 = f'The robot gripper is moving towards the mug.'
            annot_final = annot_final + f'{annot_gripper1}\n'
        elif current_frame_gripper_target_dist > previous_frame_gripper_target_dist:
            annot_gripper1 = f'The robot gripper is moving away from the mug.'
            annot_final = annot_final + f'{annot_gripper1}\n'
    elif task in ['PrepareCoffee-2']:
        if level_i in ['lev3']:
            annot_obj1 = f'The mug is in the coffee machine.\nThe robot is in the process of turning on the coffee machine.'
        else:
            annot_obj1 = f'The mug is in the coffee machine.\nThe coffee machine is off.'
        # 
        annot_final = f'{annot_obj1}\n'
        if current_frame_gripper_target_dist < previous_frame_gripper_target_dist:
            annot_gripper1 = f'The robot gripper is moving towards the coffee machine start button.'
            annot_final = annot_final + f'{annot_gripper1}\n'
        elif current_frame_gripper_target_dist > previous_frame_gripper_target_dist:
            annot_gripper1 = f'The robot gripper is moving away from the coffee machine start button.'
            annot_final = annot_final + f'{annot_gripper1}\n'
    elif task in ['RestockPantry-1', 'ArrangeVegetables-1']:
        if 'Vegetable' in task:
            obj = 'vegetable'
            preposition1 = 'in'
            location1 = 'sink'
            preposition2 = 'on'
            location2 = 'cutting board'
        else:
            obj = 'can'
            preposition1 = 'on'
            location1 = 'counter'
            preposition2 = 'in'
            location2 = 'cabinet'
        # 
        if level_i in ['lev2', 'lev3']:
            annot_obj1 = f'The robot is in the process of putting the first {obj} {preposition2} the {location2}.\nThe second {obj} is still {preposition1} the {location1}.'
        else:
            annot_obj1 = f'The first {obj} is still {preposition1} the {location1}.\nThe second {obj} is still {preposition1} the {location1}.'
        # 
        annot_final = f'{annot_obj1}\n'
        if current_frame_gripper_target_dist < previous_frame_gripper_target_dist:
            annot_gripper1 = f'The robot gripper is moving towards the first {obj}.'
            annot_final = annot_final + f'{annot_gripper1}\n'
        elif current_frame_gripper_target_dist > previous_frame_gripper_target_dist:
            annot_gripper1 = f'The robot gripper is moving away from the first {obj}.'
            annot_final = annot_final + f'{annot_gripper1}\n'
    elif task in ['RestockPantry-2', 'ArrangeVegetables-2']:
        if 'Vegetable' in task:
            obj = 'vegetable'
            preposition1 = 'in'
            location1 = 'sink'
            preposition2 = 'on'
            location2 = 'cutting board'
        else:
            obj = 'can'
            preposition1 = 'on'
            location1 = 'counter'
            preposition2 = 'in'
            location2 = 'cabinet'
        # 
        if level_i in ['lev3']:
            annot_obj1 = f'The first {obj} has been put {preposition2} the {location2}.\nThe robot is in the process of putting the second {obj} {preposition2} the {location2}.'
        else:
            annot_obj1 = f'The first {obj} has been put {preposition2} the {location2}.\nThe second {obj} is still {preposition1} the {location1}.'
        # 
        annot_final = f'{annot_obj1}\n'
        if current_frame_gripper_target_dist < previous_frame_gripper_target_dist:
            annot_gripper1 = f'The robot gripper is moving towards the second {obj}.'
            annot_final = annot_final + f'{annot_gripper1}\n'
        elif current_frame_gripper_target_dist > previous_frame_gripper_target_dist:
            annot_gripper1 = f'The robot gripper is moving away from the second {obj}.'
            annot_final = annot_final + f'{annot_gripper1}\n'
    elif 'PnP' in task or task in ['CoffeeSetupMug', 'CoffeeServeMug', 'MicrowaveThawing-2']:
        # obj = task_description.split('pick the ')[-1].split(' from')[0]
        obj = get_obj_name(task_description, task)
        location1 = task_description.split('from the ')[-1].split(' and')[0]
        if 'place it in the' in task_description:
            location2 = task_description.split('in the ')[-1]
        elif 'place it on the' in task_description:
            location2 = task_description.split('on the ')[-1]
        elif 'place it under the' in task_description:
            location2 = task_description.split('on the ')[-1]
        elif 'place it on ' in task_description:
            location2 = task_description.split('place it on ')[-1]
        # 
        if idx_start_contact_i is None or current_frame_idx < idx_start_contact_i:
            if location1 in ['sink', 'microwave', 'cabinet']:
                annot_obj1 = f'The {obj} is in the {location1}.'
            # elif location1 in ['stove', 'counter']:
            else:
                annot_obj1 = f'The {obj} is on the {location1}.'
            # 
            annot_contact = f'The robot is not in contact with the {obj}.'
            annot_holding = f'The robot is not holding the {obj}.'
            # 
            if current_frame_gripper_target_dist < previous_frame_gripper_target_dist:
                annot_gripper1 = f'The robot gripper is moving towards the {obj}.'
            elif current_frame_gripper_target_dist > previous_frame_gripper_target_dist:
                annot_gripper1 = f'The robot gripper is moving away from the {obj}.'
            # 
            annot_obj2 = f'The {obj} is not moving.'
            # 
            annot_final = f'{annot_obj1}\n{annot_contact}\n{annot_holding}\n{annot_gripper1}\n{annot_obj2}'
            # annot_final = f'{annot_obj1}\n{annot_contact}\n{annot_holding}\n{annot_gripper1}\n{annot_obj2}'
        #
        else:
            # 
            # if location1 in ['sink', 'microwave', 'cabinet']:
            #     annot_obj1 = f'The {obj} is in the {location1}.'
            # elif location1 in ['stove', 'counter']:
            #     annot_obj1 = f'The {obj} is on the {location1}.'
            # 
            if current_frame_obj_is_touching_gripper == 'True':
                annot_contact = f'The robot is in contact with the {obj}.'
            else:
                annot_contact = f'The robot is not in contact with the {obj}.'
            # 
            if current_frame_obj_is_only_touching_gripper == 'True':
                annot_contact = f'The robot is in contact with the {obj}.'
                annot_holding = f'The robot is holding the {obj}.'
            else:
                annot_holding = f'The robot is not holding the {obj}.'
            # 
            # if current_frame_gripper_target_dist < previous_frame_gripper_target_dist:
            #     annot_gripper1 = f'The robot gripper is moving towards the {obj}.'
            # elif current_frame_gripper_target_dist > previous_frame_gripper_target_dist:
            #     annot_gripper1 = f'The robot gripper is moving away from the {obj}.'
            # 
            # 
            if current_frame_env_dist < previous_frame_env_dist:
                annot_obj2 = f'The {obj} is moving closer to the {location2}.'
            else:
                annot_obj2 = f'The {obj} is moving further away from the {location2}.'
            # 
            # annot_obj2 = f'The {obj} is not moving.'
            # 
            annot_final = f'{annot_contact}\n{annot_holding}\n{annot_obj2}'
            # annot_final = f'{annot_obj1}\n{annot_contact}\n{annot_holding}\n{annot_gripper1}\n{annot_obj2}'
            # annot_final = f'{annot_obj1}\n{annot_contact}\n{annot_holding}\n{annot_gripper1}\n{annot_obj2}'
    elif task in ['OpenSingleDoor', 'OpenDrawer', 'MicrowaveThawing-1']:
        door_type = get_obj_name(task_description, task)
        # 
        if idx_start_contact_i is None or current_frame_idx < idx_start_contact_i:
            annot_obj1 = f'The {door_type} is closed.'
            # 
            annot_contact = f'The robot is not in contact with the {door_type} handle.'
            # 
            annot_final = f'{annot_obj1}\n{annot_contact}\n'
            if current_frame_gripper_target_dist < previous_frame_gripper_target_dist:
                annot_gripper1 = f'The robot gripper is moving towards the {door_type} handle.'
                annot_final = annot_final + f'{annot_gripper1}\n'
            elif current_frame_gripper_target_dist > previous_frame_gripper_target_dist:
                annot_gripper1 = f'The robot gripper is moving away from the {door_type} handle.'
                annot_final = annot_final + f'{annot_gripper1}\n'
        else:
            annot_final = ''
            if '1' in level_i.split('-')[0].split('.')[0] or '2' in level_i.split('-')[0].split('.')[0] or '3' in level_i.split('-')[0].split('.')[0]:
                annot_obj1 = f'The {door_type} is closed.'
                annot_final = f'{annot_obj1}\n'
            else:
                if current_frame_idx - idx_start_contact_i > 50:
                    annot_obj1 = f'The {door_type} is in the process of being opened.'
                    annot_final = f'{annot_obj1}\n'
            if current_frame_obj_is_touching_gripper == 'True':
                annot_contact = f'The robot is in contact with the {door_type} handle.'
                annot_final = annot_final + f'{annot_contact}\n'
            else:
                annot_contact = f'The robot is not in contact with the {door_type} handle.'
                annot_final = annot_final + f'{annot_contact}\n'
                # 
                if annot_contact_prev is None or ' not ' in annot_contact_prev:
                    if current_frame_gripper_target_dist < previous_frame_gripper_target_dist:
                        annot_gripper1 = f'The robot gripper is moving towards the {door_type} handle.'
                    elif current_frame_gripper_target_dist > previous_frame_gripper_target_dist:
                        annot_gripper1 = f'The robot gripper is moving away from the {door_type} handle.'
                    # 
                    annot_final = annot_final + f'{annot_gripper1}\n'
            # 
    elif task in ['CloseSingleDoor', 'CloseDrawer', 'MicrowaveThawing-3']:
        door_type = get_obj_name(task_description, task)
        # 
        if idx_start_contact_i is None or current_frame_idx < idx_start_contact_i:
            annot_obj1 = f'The {door_type} is open.'
            # 
            annot_contact = f'The robot is not in contact with the {door_type}.'
            # 
            annot_final = f'{annot_obj1}\n{annot_contact}\n'
            if current_frame_gripper_target_dist < previous_frame_gripper_target_dist:
                annot_gripper1 = f'The robot gripper is moving towards the {door_type}.'
                annot_final = annot_final + f'{annot_gripper1}\n'
            elif current_frame_gripper_target_dist > previous_frame_gripper_target_dist:
                annot_gripper1 = f'The robot gripper is moving away from the {door_type}.'
                annot_final = annot_final + f'{annot_gripper1}\n'
        else:
            annot_final = ''
            if '1' in level_i.split('-')[0].split('.')[0] or '2' in level_i.split('-')[0].split('.')[0] or '3' in level_i.split('-')[0].split('.')[0]:
                annot_obj1 = f'The {door_type} is open.'
                annot_final = f'{annot_obj1}\n'
            else:
                if current_frame_idx - idx_start_contact_i > 50:
                    annot_obj1 = f'The {door_type} is in the process of being closed.'
                    annot_final = f'{annot_obj1}\n'
            if current_frame_obj_is_touching_gripper == 'True':
                annot_contact = f'The robot is in contact with the {door_type}.'
                annot_final = annot_final + f'{annot_contact}\n'
            else:
                annot_contact = f'The robot is not in contact with the {door_type}.'
                annot_final = annot_final + f'{annot_contact}\n'
                # 
                if annot_contact_prev is None or ' not ' in annot_contact_prev:
                    if current_frame_gripper_target_dist < previous_frame_gripper_target_dist:
                        annot_gripper1 = f'The robot gripper is moving towards the {door_type}.'
                    elif current_frame_gripper_target_dist > previous_frame_gripper_target_dist:
                        annot_gripper1 = f'The robot gripper is moving away from the {door_type}.'
                    # 
                    annot_final = annot_final + f'{annot_gripper1}\n'
            # 
    # 
    elif task in ['TurnOnSinkFaucet', 'TurnOffSinkFaucet'] or task in ['TurnOnStove', 'TurnOffStove']:
        if 'Sink' in task:
            obj_body = 'sink'
        elif 'Stove' in task:
            obj_body = 'stove'
        elif 'Microwave' in task:
            obj_body = 'microwave'
        elif 'Coffee' in task:
            obj_body = 'coffee machine'
        # 
        # sink handle or stove knob or button type
        obj = get_obj_name(task_description, task)
        # 
        if idx_start_contact_i is None or current_frame_idx < idx_start_contact_i:
            if 'Off' in task:
                annot_obj1 = f'The {obj_body} is on.'
            else:
                annot_obj1 = f'The {obj_body} is off.' 
            # 
            annot_contact = f'The robot is not in contact with the {obj}.'
            # 
            annot_final = f'{annot_obj1}\n{annot_contact}\n'
            if current_frame_gripper_target_dist < previous_frame_gripper_target_dist:
                annot_gripper1 = f'The robot gripper is moving towards the {obj}.'
                annot_final = annot_final + f'{annot_gripper1}\n'
            elif current_frame_gripper_target_dist > previous_frame_gripper_target_dist:
                annot_gripper1 = f'The robot gripper is moving away from the {obj}.'
                annot_final = annot_final + f'{annot_gripper1}\n'
        else:
            annot_final = ''
            if '1' in level_i.split('-')[0].split('.')[0] or '2' in level_i.split('-')[0].split('.')[0] or '3' in level_i.split('-')[0].split('.')[0]:
                if 'Off' in task:
                    annot_obj1 = f'The {obj_body} is on.'
                else:
                    annot_obj1 = f'The {obj_body} is off.'
                annot_final = f'{annot_obj1}\n'
            else:
                if current_frame_idx - idx_start_contact_i > 50:
                    if 'Off' in task:
                        annot_obj1 = f'The {obj} is in the process of being turned off.'
                    else:
                        annot_obj1 = f'The {obj} is in the process of being turned on.'
                    # 
                    annot_final = f'{annot_obj1}\n'
            if current_frame_obj_is_touching_gripper == 'True':
                annot_contact = f'The robot is in contact with the {obj}.'
                annot_final = annot_final + f'{annot_contact}\n'
            else:
                annot_contact = f'The robot is not in contact with the {obj}.'
                annot_final = annot_final + f'{annot_contact}\n'
                # 
                if annot_contact_prev is None or ' not ' in annot_contact_prev:
                    if current_frame_gripper_target_dist < previous_frame_gripper_target_dist:
                        annot_gripper1 = f'The robot gripper is moving towards the {obj}.'
                    elif current_frame_gripper_target_dist > previous_frame_gripper_target_dist:
                        annot_gripper1 = f'The robot gripper is moving away from the {obj}.'
                    # 
                    annot_final = annot_final + f'{annot_gripper1}\n'
            # 
    elif task in ['TurnSinkSpout']:
        # sink spout
        obj = get_obj_name(task_description, task)
        # 
        if idx_start_contact_i is None or current_frame_idx < idx_start_contact_i:
            annot_obj1 = f'The {obj} has not been turned.'
            # 
            annot_contact = f'The robot is not in contact with the {obj}.'
            # 
            annot_final = f'{annot_obj1}\n{annot_contact}\n'
            if current_frame_gripper_target_dist < previous_frame_gripper_target_dist:
                annot_gripper1 = f'The robot gripper is moving towards the {obj}.'
                annot_final = annot_final + f'{annot_gripper1}\n'
            elif current_frame_gripper_target_dist > previous_frame_gripper_target_dist:
                annot_gripper1 = f'The robot gripper is moving away from the {obj}.'
                annot_final = annot_final + f'{annot_gripper1}\n'
        else:
            annot_final = ''
            if '1' in level_i.split('-')[0].split('.')[0] or '2' in level_i.split('-')[0].split('.')[0] or '3' in level_i.split('-')[0].split('.')[0]:
                annot_obj1 = f'The {obj} has not been turned.'
                annot_final = f'{annot_obj1}\n'
            else:
                if current_frame_idx - idx_start_contact_i > 50:
                    annot_obj1 = f'The {obj} is in the process of being turned.'
                    annot_final = f'{annot_obj1}\n'
            if current_frame_obj_is_touching_gripper == 'True':
                annot_contact = f'The robot is in contact with the {obj}.'
                annot_final = annot_final + f'{annot_contact}\n'
            else:
                annot_contact = f'The robot is not in contact with the {obj}.'
                annot_final = annot_final + f'{annot_contact}\n'
                # 
                if annot_contact_prev is None or ' not ' in annot_contact_prev:
                    if current_frame_gripper_target_dist < previous_frame_gripper_target_dist:
                        annot_gripper1 = f'The robot gripper is moving towards the {obj}.'
                    elif current_frame_gripper_target_dist > previous_frame_gripper_target_dist:
                        annot_gripper1 = f'The robot gripper is moving away from the {obj}.'
                    # 
                    annot_final = annot_final + f'{annot_gripper1}\n'
    elif task in ['TurnOnMicrowave', 'TurnOffMicrowave', 'CoffeePressButton', 'MicrowaveThawing-4']:
        if 'Sink' in task:
            obj_body = 'sink'
        elif 'Stove' in task:
            obj_body = 'stove'
        elif 'Microwave' in task:
            obj_body = 'microwave'
        elif 'Coffee' in task:
            obj_body = 'coffee machine'
        # 
        # sink handle or stove knob or button type
        obj = get_obj_name(task_description, task)
        # 
        if idx_start_contact_i is None or current_frame_idx < idx_start_contact_i:
            if 'Off' in task:
                annot_obj1 = f'The {obj_body} is on.'
            else:
                annot_obj1 = f'The {obj_body} is off.' 
            # 
            annot_contact = f'The robot is not in contact with the {obj}.'
            # 
            annot_final = f'{annot_obj1}\n{annot_contact}\n'
            if current_frame_gripper_target_dist < previous_frame_gripper_target_dist:
                annot_gripper1 = f'The robot gripper is moving towards the {obj}.'
                annot_final = annot_final + f'{annot_gripper1}\n'
            elif current_frame_gripper_target_dist > previous_frame_gripper_target_dist:
                annot_gripper1 = f'The robot gripper is moving away from the {obj}.'
                annot_final = annot_final + f'{annot_gripper1}\n'
        else:
            annot_final = ''
            if '1' in level_i.split('-')[0].split('.')[0] or '2' in level_i.split('-')[0].split('.')[0]:
                if 'Off' in task:
                    annot_obj1 = f'The {obj_body} is on.'
                else:
                    annot_obj1 = f'The {obj_body} is off.'
                annot_final = f'{annot_obj1}\n'
            else:
                if current_frame_idx - idx_start_contact_i > 50:
                    if 'Off' in task:
                        annot_obj1 = f'The {obj} is in the process of being turned off.'
                    else:
                        annot_obj1 = f'The {obj} is in the process of being turned on.'
                    # 
                    annot_final = f'{annot_obj1}\n'
            if current_frame_obj_is_touching_gripper == 'True':
                annot_contact = f'The robot is in contact with the {obj}.'
                annot_final = annot_final + f'{annot_contact}\n'
            else:
                annot_contact = f'The robot is not in contact with the {obj}.'
                annot_final = annot_final + f'{annot_contact}\n'
                # 
                if annot_contact_prev is None or ' not ' in annot_contact_prev:
                    if current_frame_gripper_target_dist < previous_frame_gripper_target_dist:
                        annot_gripper1 = f'The robot gripper is moving towards the {obj}.'
                    elif current_frame_gripper_target_dist > previous_frame_gripper_target_dist:
                        annot_gripper1 = f'The robot gripper is moving away from the {obj}.'
                    # 
                    annot_final = annot_final + f'{annot_gripper1}\n'
            # 
    # 
    # 
    annot_contact_prev = annot_contact
    # 
    # 
    return annot_final



################ Eval of Video QA  



def eval_video_qa(frame_descriptions_list):
    keypoint1 = None
    keypoint2 = None
    keypoint3 = None
    if task in ['MicrowaveThawing', 'RestockPantry', 'ArrangeVegetables', 'PrepareCoffee', 'PreSoakPan']:
        keypoint1 = len(gripper_target_dist_list_i[0])
        if len(gripper_target_dist_list_i) > 1:
            keypoint2 = keypoint1 + len(gripper_target_dist_list_i[1])
        if len(gripper_target_dist_list_i) > 2:
            keypoint3 = keypoint2 + len(gripper_target_dist_list_i[2])
    # 
    step_list = get_step_list(task)
    step_list
    # 
    response_text_yes_no_bin_qa_list = []
    response_text_frame_number_qa_list = []
    response_text_qa_list = []
    groundtruth_qa_list = []
    groundtruth_frame_number_qa_list = []
    for step_idx in range(len(step_list)): 
        # step_idx=0
        response_text_frame_number_qa = None
        step_i_template = step_list[step_idx]
        if 'PnP' in task or task in ['CoffeeServeMug', 'CoffeeSetupMug']:
            obj = get_obj_name(task_description_i, task)
            location1 = task_description_i.split('from the ')[-1].split(' and')[0]
            if 'place it in the' in task_description_i:
                location2 = task_description_i.split('in the ')[-1]
            elif 'place it on the' in task_description_i:
                location2 = task_description_i.split('on the ')[-1]
            elif 'place it under the' in task_description_i:
                location2 = task_description_i.split('on the ')[-1]
            elif 'place it on' in task_description_i:
                location2 = task_description_i.split('place it on ')[-1]
            else:
                assert False, f"location2 not found in task_description: {task_description_i}"
            step_i_format = step_i_template.format(obj=obj, location2=location2)
        # elif task in ['OpenSingleDoor', 'OpenDrawer'] or task in ['CloseSingleDoor', 'CloseDrawer']:
        else:
            if not task in ['RestockPantry', 'ArrangeVegetables', 'PrepareCoffee', 'PreSoakPan']:
                obj = get_obj_name(task_description_i, task)
                step_i_format = step_i_template.format(obj=obj)
            else:
                step_i_format = step_i_template
        # 
        frame_description_concat = ''
        for i in range(1,len(frame_descriptions_list)):
            frame_desription_i = frame_descriptions_list[i]
            frame_desription_i = frame_desription_i.split('escription: ')[-1]
            frame_desription_i = frame_desription_i.strip()
            frame_desription_i = f'Frame{i}: ' + frame_desription_i
            frame_desription_i = frame_desription_i.split('\n')[0] + '\n'
            frame_description_concat += frame_desription_i 
            # frame_description_concat += frame_desription_i + ' '
        print(frame_description_concat)
        question = llm_evaluator_eval3_question_template.format(task_description=task_description_i, step=step_i_format)
        prompt =  question + '\n\n' + frame_description_concat
        # print(question)
        print(prompt)
        if "gpt" in llm_evaluator_eval3_model_name:
            response = client.chat.completions.create(temperature=0.0,model=model_name,messages=[{"role": "user","content": [{"type": "text", "text": prompt}]}])
            response_text=response.choices[0].message.content
        elif "gemini" in llm_evaluator_eval3_model_name:
            google_model = genai.GenerativeModel(model_name = model_name, system_instruction = 'You are a helpful AI assistant.')
            response = google_model.generate_content( [prompt], generation_config = genai.GenerationConfig(max_output_tokens=100,temperature=0,top_k=1))
            response = response.to_dict()
            response_text = response['candidates'][0]['content']['parts'][0]['text']
        # 
        print(response_text)
        response_text_yes_no = response_text.split('Final Answer: ')[-1].strip().lower()
        if 'yes' in response_text_yes_no:
            response_text_yes_no_bin_qa = 1
            response_text_frame_number_qa  = response_text_yes_no.split('frame number: ')[-1].strip()
            if ' ' in response_text_frame_number_qa:
                response_text_frame_number_qa = response_text_frame_number_qa.split(' ')[0]
            response_text_frame_number_qa = int(response_text_frame_number_qa)
        else:
            response_text_yes_no_bin_qa = 0
            response_text_frame_number_qa ='NA'
        # 
        response_text_yes_no_bin_qa_list.append(response_text_yes_no_bin_qa)
        response_text_frame_number_qa_list.append(response_text_frame_number_qa)
        response_text_qa_list.append(response_text)
        # 
        groundtruth_qa = get_groundtruth_qa(task, step_list[step_idx], level_i)
        groundtruth_qa_list.append(groundtruth_qa)
        # 
        if groundtruth_qa == 1:
            groundtruth_frame_number_qa = get_groundtruth_frame_number(task, step_list[step_idx], level_i, image_file_num_list, idx_start_i, idx_final_i, idx_start_contact_i, idx_contact_i, idx_start_contact_expert_i, idx_contact_expert_i, keypoint1, keypoint2, keypoint3)
        else:
            groundtruth_frame_number_qa = 'NA'
        # 
        groundtruth_frame_number_qa_list.append(groundtruth_frame_number_qa)
        # 
    # 
    qa_accuracy = accuracy_score(groundtruth_qa_list, response_text_yes_no_bin_qa_list)
    qa_precision = precision_score(groundtruth_qa_list, response_text_yes_no_bin_qa_list)
    qa_recall = recall_score(groundtruth_qa_list, response_text_yes_no_bin_qa_list)
    qa_frame_diff = avg_frame_diff(groundtruth_frame_number_qa_list, response_text_frame_number_qa_list)
    # 
    return qa_accuracy, qa_precision, qa_recall, qa_frame_diff




llm_evaluator_eval3_question_template = "A robot was given the task of '{task_description}'. We capture a video of the robot attempting the task and describe each frame of the video below. Based on the below descriptions, did the robot {step} during the video? If yes, please identify the earliest frame number where this is indicated. Please end your response with 'Final Answer: Yes, Frame Number: {{}}' or 'Final Answer: No, Frame Number: NA'."


def get_step_list(task):
    if 'PnP' in task or task in ['CoffeeServeMug', 'CoffeeSetupMug']:
        step1="contact the {obj}"
        step2="pick up the {obj}"
        step3="drop the {obj}"		
        step4="place the {obj} in the {location2}"
        step_list = [step1, step2, step3, step4]
    elif task in ['OpenSingleDoor', 'OpenDrawer']:
        step1="contact the {obj} handle"
        step2="start opening the {obj}"		
        step3="finish opening the {obj}"
        step_list = [step1, step2, step3]
    elif task in ['CloseSingleDoor', 'CloseDrawer']:
        step1="contact the {obj}"
        step2="start closing the {obj}"		
        step3="finish closing the {obj}"
        step_list = [step1, step2, step3]
    elif task in ['TurnOnSinkFaucet', 'TurnOffSinkFaucet'] or task in ['TurnOnStove', 'TurnOffStove']:
        # obj_name = 'sink handle'
        # obj_name = 'stove knob'
        step1="contact the {obj}"
        step2="start turning the {obj}"		
        if 'Off' in task:
            step3="finish turning the {obj} to the on position"
        else:
            step3="finish turning the {obj} to the off position"
        step_list = [step1, step2, step3]
    elif task in ['TurnSinkSpout']:
        # obj_name = 'sink spout'
        step1="contact the {obj}"
        step2="start turning the {obj}"		
        step3="finish turning the {obj}"
        step_list = [step1, step2, step3]
    elif task in ['TurnOnMicrowave', 'TurnOffMicrowave'] or task in ['CoffeePressButton']:
        step1="successfully press the {obj}"
        step_list = [step1]
    elif task in ['MicrowaveThawing']:
        step1="open the microwave door"
        step2="put the {obj} in the microwave"
        step3="close the microwave door"
        step4="press the microwave start button"
        step_list = [step1, step2, step3, step4]
    elif task in ['RestockPantry']:
        step1="pick up the first can from the counter"
        step2="place the first can in the cabinet"
        step3="pick up the second can from the counter"
        step4="place the second can in the cabinet"
        step_list = [step1, step2, step3, step4]
    elif task in ['ArrangeVegetables']:
        step1="pick up the first vegetable from the sink"
        step2="place the first vegetable on the cutting board"
        step3="pick up the second vegetable from the sink"
        step4="place the second vegetable on the cutting board"
        step_list = [step1, step2, step3, step4]
    elif task in ['PrepareCoffee']:
        step1="pick up the mug from the cabinet"
        step2="place the mug vegetable in the coffee machine"
        step3="turn on the coffee machine"
        step_list = [step1, step2, step3]
    elif task in ['PreSoakPan']:
        step1="pick up the pan from the counter"
        step2="place the pan in the sink"
        step3="pick up the sponge from the counter"
        step4="place the sponge in the pan"
        step5="turn on the sink"
        step_list = [step1, step2, step3, step4, step5]
    return step_list



def get_groundtruth_qa(task, step_template, level):
    level_int = level.split('-')[0].split('.')[0]
    if 'PnP' in task or task in ['CoffeeServeMug', 'CoffeeSetupMug']: 
        if 'contact' in step_template:
            if '1' in level_int or '2' in level_int:
                groundtruth = 0
            else:
                groundtruth = 1
        elif 'pick' in step_template:
            if '1' in level_int or '2' in level_int or '3' in level_int:
                groundtruth = 0
            else:
                groundtruth = 1
        elif 'drop' in step_template:
            if '1' in level_int or '2' in level_int or '3' in level_int or '5' in level_int or '6' in level_int or '7' in level_int:
                groundtruth = 0
            else:
                groundtruth = 1
        elif 'place' in step_template:
            if '1' in level_int or '2' in level_int or '3' in level_int or '4' in level_int or '5' in level_int or '6' in level_int:
                groundtruth = 0
            else:
                groundtruth = 1
    elif task in ['OpenSingleDoor', 'OpenDrawer']:
        if '6' in level_int:
            level_int = 'lev4'
        elif '7' in level_int:
            level_int = 'lev5'
        if 'contact' in step_template:
            if '1' in level_int or '2' in level_int:
                groundtruth = 0
            else:
                groundtruth = 1
        elif 'start opening' in step_template:
            if '1' in level_int or '2' in level_int or '3' in level_int:
                groundtruth = 0
            else:
                groundtruth = 1
        elif 'finish opening' in step_template:
            if '1' in level_int or '2' in level_int or '3' in level_int or '4' in level_int:
                groundtruth = 0
            else:
                groundtruth = 1
    elif task in ['CloseSingleDoor', 'CloseDrawer']:
        if '6' in level_int:
            level_int = 'lev4'
        elif '7' in level_int:
            level_int = 'lev5'
        if 'contact' in step_template:
            if '1' in level_int or '2' in level_int:
                groundtruth = 0
            else:
                groundtruth = 1
        elif 'start closing' in step_template:
            if '1' in level_int or '2' in level_int or '3' in level_int:
                groundtruth = 0
            else:
                groundtruth = 1
        elif 'finish closing' in step_template:
            if '1' in level_int or '2' in level_int or '3' in level_int or '4' in level_int:
                groundtruth = 0
            else:
                groundtruth = 1
    elif task in ['TurnOnSinkFaucet', 'TurnOffSinkFaucet'] or task in ['TurnOnStove', 'TurnOffStove'] or task in ['TurnSinkSpout']:
        if '6' in level_int:
            level_int = 'lev4'
        elif '7' in level_int:
            level_int = 'lev5'
        if 'contact' in step_template:
            if '1' in level_int or '2' in level_int:
                groundtruth = 0
            else:
                groundtruth = 1
        elif 'start turning' in step_template:
            if '1' in level_int or '2' in level_int or '3' in level_int:
                groundtruth = 0
            else:
                groundtruth = 1
        elif 'finish turning' in step_template:
            if '1' in level_int or '2' in level_int or '3' in level_int or '4' in level_int:
                groundtruth = 0
            else:
                groundtruth = 1
    elif task in ['TurnOnMicrowave', 'TurnOffMicrowave'] or task in ['CoffeePressButton']:
        if True:
            if '1' in level_int or '2' in level_int:
                groundtruth = 0
            else:
                groundtruth = 1
    # 
    elif task in ['MicrowaveThawing']:
        if 'open' in step_template:
            if '1' in level_int:
                groundtruth = 0
            else:
                groundtruth = 1
        elif 'put' in step_template:
            if '1' in level_int or '2' in level_int:
                groundtruth = 0
            else:
                groundtruth = 1
        elif 'close' in step_template:
            if '1' in level_int or '2' in level_int or '3' in level_int:
                groundtruth = 0
            else:
                groundtruth = 1
        elif 'press' in step_template:
            if '1' in level_int or '2' in level_int or '3' in level_int or '4' in level_int:
                groundtruth = 0
            else:
                groundtruth = 1
    elif task in ['RestockPantry', 'ArrangeVegetables']:
        if 'pick up the first' in step_template:
            if '1' in level_int:
                groundtruth = 0
            else:
                groundtruth = 1
        elif 'place the first' in step_template:
            if '1' in level_int:
                groundtruth = 0
            else:
                groundtruth = 1
        elif 'pick up the second' in step_template:
            if '1' in level_int or '2' in level_int:
                groundtruth = 0
            else:
                groundtruth = 1
        elif 'place the second' in step_template:
            if '1' in level_int or '2' in level_int:
                groundtruth = 0
            else:
                groundtruth = 1
    elif task in ['PreSoakPan']:
        if 'pick up the pan' in step_template:
            if '1' in level_int:
                groundtruth = 0
            else:
                groundtruth = 1
        elif 'place the pan' in step_template:
            if '1' in level_int:
                groundtruth = 0
            else:
                groundtruth = 1
        elif 'pick up the sponge' in step_template:
            if '1' in level_int or '2' in level_int:
                groundtruth = 0
            else:
                groundtruth = 1
        elif 'place the sponge' in step_template:
            if '1' in level_int or '2' in level_int:
                groundtruth = 0
            else:
                groundtruth = 1
        elif 'turn on' in step_template:
            if '1' in level_int or '2' in level_int or '3' in level_int:
                groundtruth = 0
            else:
                groundtruth = 1
    elif task in ['PrepareCoffee']:
        if 'pick up the mug' in step_template:
            if '1' in level_int:
                groundtruth = 0
            else:
                groundtruth = 1
        elif 'place the mug' in step_template:
            if '1' in level_int:
                groundtruth = 0
            else:
                groundtruth = 1
        elif 'turn on the' in step_template:
            if '1' in level_int or '2' in level_int:
                groundtruth = 0
            else:
                groundtruth = 1
    return groundtruth



def get_groundtruth_frame_number(task, step_template, level, frame_idx_list, idx_start, idx_final, idx_start_contact, idx_contact, idx_start_contact_expert, idx_contact_expert, keypoint1, keypoint2, keypoint3):
    frame_idx_list = sorted(frame_idx_list)
    level_int = level.split('-')[0].split('.')[0]
    if 'PnP' in task: 
        if 'contact' in step_template:
            if '1' in level_int or '2' in level_int:
                groundtruth_idx = 'NA'
            else:
                groundtruth_idx = idx_start_contact
        elif 'pick' in step_template:
            if '1' in level_int or '2' in level_int or '3' in level_int:
                groundtruth_idx = 'NA'
            else:
                groundtruth_idx = idx_contact
        elif 'drop' in step_template:
            if '1' in level_int or '2' in level_int or '3' in level_int or '5' in level_int or '6' in level_int or '7' in level_int:
                groundtruth_idx = 'NA'
            else:
                groundtruth_idx = idx_contact + 30
        elif 'place' in step_template:
            if '1' in level_int or '2' in level_int or '3' in level_int or '4' in level_int or '5' in level_int or '6' in level_int:
                groundtruth_idx = 'NA'
            else:
                groundtruth_idx = idx_final - 60
    elif task in ['OpenSingleDoor', 'OpenDrawer']:
        if '6' in level_int:
            level_int = 'lev4'
        elif '7' in level_int:
            level_int = 'lev5'
        if 'contact' in step_template:
            if '1' in level_int or '2' in level_int:
                groundtruth_idx = 'NA'
            else:
                groundtruth_idx = idx_start_contact
        elif 'start opening' in step_template:
            if '1' in level_int or '2' in level_int or '3' in level_int:
                groundtruth_idx = 'NA'
            else:
                groundtruth_idx = idx_contact
        elif 'finish opening' in step_template:
            if '1' in level_int or '2' in level_int or '3' in level_int or '4' in level_int:
                groundtruth_idx = 'NA'
            else:
                groundtruth_idx = idx_final - 20
    elif task in ['CloseSingleDoor', 'CloseDrawer']:
        if '6' in level_int:
            level_int = 'lev4'
        elif '7' in level_int:
            level_int = 'lev5'
        if 'contact' in step_template:
            if '1' in level_int or '2' in level_int:
                groundtruth_idx = 'NA'
            else:
                groundtruth_idx = idx_start_contact
        elif 'start closing' in step_template:
            if '1' in level_int or '2' in level_int or '3' in level_int:
                groundtruth_idx = 'NA'
            else:
                groundtruth_idx = idx_contact
        elif 'finish closing' in step_template:
            if '1' in level_int or '2' in level_int or '3' in level_int or '4' in level_int:
                groundtruth_idx = 'NA'
            else:
                groundtruth_idx = idx_final - 20
    elif task in ['TurnOnSinkFaucet', 'TurnOffSinkFaucet'] or task in ['TurnOnStove', 'TurnOffStove'] or task in ['TurnSinkSpout']:
        if '6' in level_int:
            level_int = 'lev4'
        elif '7' in level_int:
            level_int = 'lev5'
        if 'contact' in step_template:
            if '1' in level_int or '2' in level_int:
                groundtruth_idx = 'NA'
            else:
                groundtruth_idx = idx_start_contact
        elif 'start turning' in step_template:
            if '1' in level_int or '2' in level_int or '3' in level_int:
                groundtruth_idx = 'NA'
            else:
                groundtruth_idx = idx_contact
        elif 'finish turning' in step_template:
            if '1' in level_int or '2' in level_int or '3' in level_int or '4' in level_int:
                groundtruth_idx = 'NA'
            else:
                groundtruth_idx = idx_final - 20
    elif task in ['TurnOnMicrowave', 'TurnOffMicrowave'] or task in ['CoffeePressButton']:
        if True:
            if '1' in level_int or '2' in level_int:
                groundtruth_idx = 'NA'
            else:
                groundtruth_idx = idx_final - 20
    elif task in ['MicrowaveThawing']:
        if 'open' in step_template:
            if '1' in level_int:
                groundtruth_idx = 'NA'
            else:
                groundtruth_idx = keypoint1 - 50
        elif 'put' in step_template:
            if '1' in level_int or '2' in level_int:
                groundtruth_idx = 'NA'
            else:
                groundtruth_idx = keypoint2 - 50
        elif 'close' in step_template:
            if '1' in level_int or '2' in level_int or '3' in level_int:
                groundtruth_idx = 'NA'
            else:
                groundtruth_idx = keypoint3 - 50
        elif 'press' in step_template:
            if '1' in level_int or '2' in level_int or '3' in level_int or '4' in level_int:
                groundtruth_idx = 'NA'
            else:
                groundtruth_idx = idx_final - 50
    elif task in ['RestockPantry', 'ArrangeVegetables']:
        if 'pick up the first' in step_template:
            if '1' in level_int:
                groundtruth_idx = 'NA'
            else:
                groundtruth_idx = keypoint1 - 200
        elif 'place the first' in step_template:
            if '1' in level_int:
                groundtruth_idx = 'NA'
            else:
                groundtruth_idx = keypoint1 - 50
        elif 'pick up the second' in step_template:
            if '1' in level_int or '2' in level_int:
                groundtruth_idx = 'NA'
            else:
                groundtruth_idx = idx_final - 200
        elif 'place the second' in step_template:
            if '1' in level_int or '2' in level_int:
                groundtruth_idx = 'NA'
            else:
                groundtruth_idx = idx_final - 50
    elif task in ['PreSoakPan']:
        if 'pick up the pan' in step_template:
            if '1' in level_int:
                groundtruth_idx = 'NA'
            else:
                groundtruth_idx = keypoint1 - 200
        elif 'place the pan' in step_template:
            if '1' in level_int:
                groundtruth_idx = 'NA'
            else:
                groundtruth_idx = keypoint1 - 50
        elif 'pick up the sponge' in step_template:
            if '1' in level_int or '2' in level_int:
                groundtruth_idx = 'NA'
            else:
                groundtruth_idx = keypoint2 - 200
        elif 'place the sponge' in step_template:
            if '1' in level_int or '2' in level_int:
                groundtruth_idx = 'NA'
            else:
                groundtruth_idx = keypoint2 - 50
        elif 'turn on' in step_template:
            if '1' in level_int or '2' in level_int or '3' in level_int:
                groundtruth_idx = 'NA'
            else:
                groundtruth_idx = idx_final - 50
    elif task in ['PrepareCoffee']:
        if 'pick up the mug' in step_template:
            if '1' in level_int:
                groundtruth_idx = 'NA'
            else:
                groundtruth_idx = keypoint1 - 200
        elif 'place the mug' in step_template:
            if '1' in level_int:
                groundtruth_idx = 'NA'
            else:
                groundtruth_idx = keypoint1 - 50
        elif 'turn on the' in step_template:
            if '1' in level_int or '2' in level_int:
                groundtruth_idx = 'NA'
            else:
                groundtruth_idx = idx_final - 200
    # 
    if groundtruth_idx == 'NA' or groundtruth_idx is None:
        groundtruth_frame_number = 'NA'
    else:
        groundtruth_frame_number = 30
        for i in range(len(frame_idx_list)):
            if frame_idx_list[i] >= groundtruth_idx:
                groundtruth_frame_number = i
                break
    return groundtruth_frame_number



def accuracy_score(y_true, y_pred):
    return sum([1 for i in range(len(y_true)) if y_true[i] == y_pred[i]]) / len(y_true)

def precision_score(y_true, y_pred):
    if sum([x for x in y_pred if x==1]) == 0:
        return 'NA'
    tp = sum([1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 1])
    fp = sum([1 for i in range(len(y_true)) if y_true[i] == 0 and y_pred[i] == 1])
    if tp + fp == 0:
        return 0
    return tp / (tp + fp)

def recall_score(y_true, y_pred):
    if sum([x for x in y_true if x==1]) == 0:
        return 'NA'
    tp = sum([1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 1])
    fn = sum([1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 0])
    if tp + fn == 0:
        return 0
    return tp / (tp + fn)

def avg_frame_diff(y_true, y_pred):
    y_true2 = []
    y_pred2 = []
    diff_list = []
    for i in range(len(y_true)):
        if not y_true[i] == 'NA' and not y_pred[i] == 'NA':
            y_true2.append(y_true[i])
            y_pred2.append(y_pred[i])
            diff_list.append(y_pred[i] - y_true[i])
    # 
    if len(y_true2) == 0:
        return 'NA'
    else:
        return np.mean(diff_list)




