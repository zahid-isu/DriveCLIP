# DriveCLIP



Here is the codebase for running the frame based DriveCLIP framework.

### Datasets
```
1. StateFarm
2. SynDD1
3. AUC
```

### Total 8 distracted Classes:
```
0  "driver is adjusting his or her hair while driving a car"
1  "driver is drinking water from a bottle while driving a car"
2  "driver is eating while driving a car"
3  "driver is picking something from floor while driving a car"
4  "driver is reaching behind to the backseat while driving a car"
5  "driver is singing a song with music and smiling while driving"
6  "driver is talking to the phone on hand while driving a car"
7  "driver is yawning while driving a car"
```


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
2. Upload .mp4 video files into synvid folder and run the `frame.py` to extract frames at different fps. Store the frames in [0-7] folder structure shown above.
3. Check the driver profile folder to prepare the driver split
3. Feed the frames to CLIP model 
4. run the run files 


We will release the majority voting experiments and end-to-end framework for the DriveCLIP model soon.
