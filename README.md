# DriveCLIP
## Reference link: [Publication link](https://arxiv.org/abs/2306.10159)  
(please cite this paper or use it for reference)

Here is the codebase for running the frame based DriveCLIP framework.

### Datasets
```
1. StateFarm
2. SynDD1
```



1. Data folder structure should be like this: [0-7] action classes on SynDD1 dataset

```
     ./data
      ├── syn1fps_dash
      │   ├── 0
      │   ├── 1
      │   ├── 2
      │   ├── 3
      │   ├── 4
      │   ├── 5
      │   ├── 6
      │   └── 7
      ├── syn5fps_dash
      │   ├── 0
      │   ├── 1
      │   ├── 2
      │   ├── 3
      │   ├── 4
      │   ├── 5
      │   ├── 6
      │   └── 7
```

2. The subject splitting profile files are saved in `driverprofile` folder. For SynDD1 see `subject_splitting_profile.json` and for StateFarm see `driver_img_list.csv`.

```json
"fold0": {
    "train": [
        "35133",
        "65818",
        "42271",
        "19332",
        "79336",
        "56306",
        "25470",
        "24491",
        "76803",
        "24026"
    ],
    "val": [
        "49381",
        "38058"
    ],
    "test": [
        "76189",
        "61597"
    ]
}
```

## Steps to run the inference on the video file:

1. Download the video and save it in .MP4 format in a video/ folder
2. Create a conda environment and run the ``` requirements.txt``` file
3. Open ```inference.py``` and specify the CLIP backbone (model_name) and FPS (default=1FPS). Then run the python file from terminal by following the command:
        ```python inference.py --video video_path –frame Extracted_frame_directory_path```
For example:
        ```python inference.py –-video video/Dashboard_user_id_13522_NoAudio_5.MP4 --frame frame```

5. Prediction results will be stored in a .json file named ```frame_prediction.json``` This file consists of the following format:

        ``` { frame_01_path: {prediction label, [list of prediction prob. scores]}, 
              frame_02_path: {prediction label, [list of prediction prob. scores]}, ... } ```


## Steps to run frame-based experiments:
1. Set up the conda environment and run requirements.txt files from CLIP main repo
2. Upload .mp4 video files into synvid folder (see `synvid/video_list.csv`) and run the `frame.py` to extract frames at different fps. Store the frames in [0-7] folder structure shown above.
3. Check the driver profile folder to prepare the driver split
3. Feed the frames to CLIP model 
4. run the run files 

### Repo for model comparison: [Link](https://github.com/suzoosuagr/CLIP_Zahid.git)
### Repo for VideoCLIP: [Link](https://github.com/jiajingchen113322/DeepInsigth.git)
