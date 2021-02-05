
# Reminder

My GDrive with some data to download can be found [here](https://drive.google.com/drive/folders/18ZITzda-ByP9Mnxt3nFNfxxfl260T9S1?usp=sharing).

Folder architecture:
```
AR-Chess_Beta\ObjectDetection
├── data                          # Data folder
│   ├── _final_test                 # Videos to test the final model on
│   │   └── *.mp4
│   ├── 0_roboflow_416x416_default  # Roboflow chess dataset in yolov5 format
│   │   ├── test                      # Test set folder
│   │   ├── train                     # Train set folder
│   │   ├── valid                     # Validation set folder
│   │   ├── ...
│   │   └── labels_distribution.png   # Labels distribution within this set
│   ├── x_xxx                       # Other custom made dataset
│   └── chess.yaml                  # Data file for the training
├── utils                         # Some utils scripts
│   └── *.py
├── yolov5                        # Forked ultralytics/yolov5 repo
│   ├── ...
│   ├── runs                        # Outputs of the train, test and detect scripts
│   ├── ...
│   └── yolov5s.pt                  # Pre-trained YoloV5s weights
└── README.md

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
