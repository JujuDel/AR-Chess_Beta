
# Reminder

My GDrive with some data to download can be found [here](https://drive.google.com/drive/folders/18ZITzda-ByP9Mnxt3nFNfxxfl260T9S1?usp=sharing).

Folder architecture:
```
AR-Chess_Beta\ObjectDetection
├── roboflow_416x416_default  # Roboflow chess dataset in yolov5 format
├── yolov5                    # ultralytics/yolov5 repo
│   ├── ...
│   ├── runs                  # Fill this folder with the downloaded data
│   ├── ...
│   └── yolov5s.pt            # Pre-trained YoloV5s weights
├── chess.yaml                # Data file for the training
├── README.md                 #
└── *.mp4                     # Test the detection on it

```
### Training

```bash
(myEnv) $ python train.py --weights "./yolov5s.pt" --model "./model/yolov5s.yaml" --data "../data/chess.yaml" 
```

### Tensorboard

```bash
(myEnv) $ tensorboard --logdir=runs
```

### GPUStat

```bash
(myEnv) $ gpustat --watch -cp
```

### Predict

```bash
(myEnv) $ python detect.py --weights "./runs/default/train/weights/best.pt" --source "../data/final_test/shortest_game_magnus_cropped.mp4"
```
