import os
import io
import numpy as np
import glob
from PIL import Image
import base64
from openai import OpenAI
import google.generativeai as genai
import pandas as pd


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


def rover(model_name, task_description_i, camera_view, frame_file_list, current_start_idx=0, subtask_completion_threshold=100, include_last_subtask_frame=False):
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
            current_idx, new_subtask_list2, new_subtask_progress_list2, new_subtask_frame_descriptions_list2, new_base64_image_current2  = rover(new_subtask, camera_view, frame_file_list, current_idx-1)
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


