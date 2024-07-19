# Shot_Detection
A shot detection repository, including AutoShot, containing pretained model

# How to use it in Kaggle/Colab notebook
## 1. Link of the dataset
Data part 1:
   - Phuc Nguyen: [AIC_VideoB1v1](https://www.kaggle.com/datasets/superheroinmordenday/c00-vidieo)
   - Tinh Anh: [AIC_VideoB1v2](https://www.kaggle.com/datasets/khitrnhxun/aic-videob1v2)

Data part 2:
   - Quan Tran(2 notebook): [AIC_VideoB2](https://www.kaggle.com/datasets/superheroinmordenday/aic-vidieob1v2)

Data part 3:
   - Leo Nguyen: [AIC_VideoB3v1](https://www.kaggle.com/datasets/khitrnhxun/aic-videob3-0)
   - Leo Nguyen: [AIC_VideoB3v2](https://www.kaggle.com/datasets/superheroinmordenday/aic-b2-v3)
   - Quang Hao: [AIC_VideoB3v3](https://www.kaggle.com/datasets/nguynlngnamanh/aic-videob3-2)

## 2. Step-by-step instruction
<br>

#### 2.1. Create Kaggle account
1. Create account Kaggle with your gmail account
2. Then go to setting -> go to `phone verify`, and add your phone number
![setting](/img/image.png)
![phone verify](/img/image-1.png)

3. Then click on the link dataset I have assigned for each of you, then open the notebook
![notebook](/img/image-2.png)

4. In the notebook, heads to setting. If your account is phone verified, it should pop up this internet -> turn it on. If not, press F5 multiple times.
![notebook setting](/img/image-3.png)

#### 2.2. Get Google Drive API
1. Heads to this [link](https://console.cloud.google.com/)
2. Create a project, name whatever you want
![project](/img/image-4.png)
![new project](/img/image-5.png)
3. Then heads to Google Drive API
![API library](/img/image-6.png)
![Google Drive](/img/image-7.png)
![Enable GoogleDrive](/img/image-8.png)

4. After that it should redirect you to the API& services interface

5. Go to OAuth consent screen
![OAuth consent screen](/img/image-9.png)

6. Choose `MAKE EXTERNAL`
- In the first page, enter your name, and your email to the field that must be filled ( optional field is not important to consider)
![info](/img/image-10.png)
![dev mail](/img/image-11.png)
add your mail and continue
- Pass `scopes`, save and continue
- Test users, add your email
- Summary -> done

7. Then heads to Credentials -> Create credentials -> create service accounts
![Service Account](/img/image-12.png)

8. Fill your info
![](/img/image-13.png)
- then select onwer in `Grant`
![alt text](/img/image-14.png)
 
- fill out your email 
![alt text](/img/image-15.png)

9. If everything is okay, you should have this service account
![alt text](/img/image-16.png)

10. Click on the account, then heads to keys
![alt text](/img/image-17.png)

11. Click on `Adds key` -> `Create New keys`

![alt text](/img/image-18.png)

Then your json key will be saved to your local machine

#### 2.3. Run the notebook

1. Go to this section, then upload your JSON credentials as dataset
![alt text](/img/image-19.png)

2. Then running the following command
- 2.1. 
```python
!git clone https://github.com/TinhAnhGitHub/Shot_Detection.git
%cd Shot_Detection
%pwd
```
```python
!python -m venv venv
!source venv/bin/activate
!pip install -q -r requirements.txt
!pip install einops==0.8.0
!pip install ffmpeg-python==0.2.0
```

```python
import threading
import time
import os
from typing import Dict, List, Any
import cv2
import json
import numpy as np
from process_video import VideoProcessor
from io_setup import setup_video_path
```

This is the important block
```python
class KaggleSessionKeeper(threading.Thread):
    def __init__(self, delay_min=175, delay_max=185):
        super(KaggleSessionKeeper, self).__init__()
        self.delay_min = delay_min
        self.delay_max = delay_max
        self.running = False
        self.program_running = True
        self.action_count = 0

    def start_actions(self):
        self.running = True

    def stop_actions(self):
        self.running = False

    def exit(self):
        self.stop_actions()
        self.program_running = False

    def run(self):
        while self.program_running:
            while self.running:
                # Perform a small action to keep the session active
                print("Keeping session active...")
                self.action_count += 1
                print(f"Action {self.action_count} performed")
                
                delay = np.random.uniform(self.delay_min, self.delay_max)
                time.sleep(delay)
            time.sleep(0.1)

def main_process():
    
    
    pretrained_model_weights_path = "/kaggle/working/Shot_Detection/AutoShot/model_weight/ckpt_0_200_0.pth"
    keyframe_output_dir = "/kaggle/working/output_sample"
    credentials_path ="/kaggle/input/apikeys/aic-dataset-429815-812ca38e92db.json"
    token_path = ''
    drive_folder_id = "11LHMCE9r0L3dT8J1d_QAD4mBy4f0C6ar"
    
    input_dir = "/kaggle/input/aic-videob1v2"
    all_video_paths = setup_video_path(input_dir)
    
    video_processor = VideoProcessor(
        pretrained_model_weights_path,
        keyframe_output_dir,
        credentials_path,
        token_path,
        drive_folder_id,
        user_service_account=True
    )
    
    
    video_processor.process_videos(all_video_paths)
    
    print("Main process completed")

# Initialize session keeper
session_keeper = KaggleSessionKeeper()
session_keeper.start()
session_keeper.start_actions()

# Start the main process
main_process()

# Stop the session keeper after the main process is done
session_keeper.exit()
session_keeper.join()

print("All processes completed.")
```

You need to fill out the configuration file
![alt text](/img/image-20.png)
1. Pretrained model path and keyframe_dir, you keep the same
2. Credentials part, heads to the JSON credential file you just uploaded, copy the path, and paste it in this variable
3. drive_foler_id: heads to your folder in google drive -> create a foler of your choice -> heads to share 
![alt text](/img/image-21.png)
- 3.1. First take your service account, and invite it. Make sure it in edit mode
- Then copy the link, then ID will be in the link:
`ttps://drive.google.com/drive/folders/[id]`
![alt text](/img/image-22.png)
From the 
- then paste it in the drive_folder_id variable

4. The input will be your root folder of dataset

5. Then run the notebook, and remember to check if the folder and the keyframes appear in your drive folder

## 3. Errors/ Issues

If any error occurs, feel free to leave in `Issues` in my repo