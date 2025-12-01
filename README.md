
# ROVER: Recursive Reasoning Over Videos with Vision-Language Models for Embodied Tasks

This repository contains the official implementation of methods for the NeurIPS 2025 paper:

**ROVER: Recursive Reasoning Over Videos with Vision-Language Models for Embodied Tasks**
Philip Schroeder, Ondrej Biza, Thomas Weng, Hongyin Luo, James Glass
---

## 📁 Quick Start

from rover_model import (
    rover,
    process_rover_output,
)
import google.generativeai as genai

API_KEY = ...
genai.configure(api_key=API_KEY)
google_model = genai.GenerativeModel(model_name = "gemini-2.5-pro")

frame_file_list = sorted(glob.glob("./test_video/frame_wrist_*.jpg"), key=lambda x: int(x.split('frame_wrist_')[-1].split('.jpg')[0]))
task_description = "Pick the cupcake from the counter and place it in the cabinet."
camera_view = "wrist view"

final_idx, subtask_list, subtask_progress_list, subtask_frame_descriptions_list, _ = rover(task_description, camera_view, frame_file_list)
final_progress_list, frame_descriptions_list = process_rover_output(subtask_list, subtask_progress_list, subtask_frame_descriptions_list)


---


## 📁 Demos 

Example videos showing ROVER frame-level reasoning and task progress prediction can be seen here:

👉 [https://rover-vlm.github.io](https://rover-vlm.github.io)

---

## 📁 Evaluation Dataset

The evaluation dataset used in our experiments can be downloaded here:

👉 [Download Evaluation Dataset](https://drive.google.com/drive/folders/1Tj5lpItYeQ7hMKenBfs6iZACY168id8Y?usp=share_link)

---
