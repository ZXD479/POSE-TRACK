# POSE-TRACK


## 🏁 Table of Contents


- Environment Requirements
- Installing Dependencies
- Running the Project


## 💾 Environment Requirements

To ensure the project runs smoothly, please meet the following environment requirements:

- **Python **: 3.8 or higher
- **Conda**: It is recommended to use Conda for managing virtual environments.
- **CUDA**: If GPU acceleration is required, ensure you have a compatible CUDA version (e.g., CUDA 11.6).
- **PyTorch**: Install a PyTorch version compatible with your CUDA version.

## 🛠️  Installing Dependencies

Here are the steps to install the project dependencies:

1. **Clone the Repository**：
   ```bash
   
   git clone https://github.com/ZXD479/Pose-Track.git 
   cd Pose-Track
   conda activate -n posetrack python=3.8
   conda activate posetrack
   pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116 
   pip install -r requirements.txt
   python setup.py develop
   
# Reproduce on SportsMOT Dataset

## Data Preparation for Reproduction on SportsMOT Dataset

To reproduce results on the SportsMOT dataset, you need to download detection and embedding files and place them in the appropriate folders.

### 下载数据

Please download the detection and embedding files from the following links:
- [Google Drive 链接](https://drive.google.com/drive/folders/14gh9e5nQhqHsw77EfxZaUyn9NgPP0-Tq?usp=sharing)
- The pose_embedding location is at /home/zxd/project/Deep-EIoU-main/Deep-EIoU/pose_embedding

### File Structure

Place the downloaded files according to the following directory structure:

```
{pose-track Root}
   |——————pose-track
   └——————detection
   |        └——————v_-9kabh1K8UA_c008.npy
   |        └——————v_-9kabh1K8UA_c009.npy
   |        └——————...
   └——————embedding
            └——————v_-9kabh1K8UA_c008.npy
            └——————v_-9kabh1K8UA_c009.npy
            └——————...
   └——————pose_embedding
            └——————v_-9kabh1K8UA_c008.npy
            └——————v_-9kabh1K8UA_c009.npy
            └——————...

```


### 2. Run tracking on SportsMOT dataset
Run the following commands, you should see the tracking result for each sequences in the interpolation folder.


```
python tools/sport_track_pose.py 
```

The result file is saved in shiyan/
Please directly zip the tracking results and submit to the [SportsMOT evaluation server](https://codalab.lisn.upsaclay.fr/competitions/12424#participate).


### 3. Run tracking on SoccerNET-Tracking dataset

```
python tools/soccer_track_pose.py 
```

跑出来的结果文件会在以下目录放置：

```

{pose-track Root}
   |——————TrackEval
            └——————data
                    └——————gt
                          └——————mot_challenge
                                   └——————soccer-val

```
执行以下命令获取结果

```
python TrackEval/scripts/run_mot_challenge.py
```



