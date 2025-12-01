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



def gvl(task_description_i, camera_view, frame_file_list, image_file_num_list, try_count_max=4):
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

