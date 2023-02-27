# DriveCLIP



Here is the codebase for running the frame based DriveCLIP framework.


1. Data folder structure should be like this: [0-7] action classes

```
     ./data
      ├── syn10fps_dash
      │   ├── 0
      │   ├── 1
      │   ├── 2
      │   ├── 3
      │   ├── 4
      │   ├── 5
      │   ├── 6
      │   └── 7
      ├── syn15fps_dash
      │   ├── 0
      │   ├── 1
      │   ├── 2
      │   ├── 3
      │   ├── 4
      │   ├── 5
      │   ├── 6
      │   └── 7
```

2. The subject splitting profile is saved in [subject_splitting_profile.json](.driverprofile/subject_splitting_profile.json).

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

Steps:
1. Set up the conda environment and run requirements.txt files from CLIP main repo
2. Upload .mp4 video files into synvid folder and run the video process script to extract frames at different fps. Store the frames in [0-7] folder structure shown above.
3. Feed the frames to CLIP model 
4. run the run files 


We will release the majority voting experiments and end-to-end framework for the DriveCLIP model soon.
