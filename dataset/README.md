Please organize this folder as follow:
```
./
├── config.py
├── ARID
│   ├── raw
│   │   ├── train_data -> (soft link to actual training dataset)
│   │   │   ├── Run (folder)
│   │   │   ├── Sit
│   │   │   ├── Stand
│   │   │   ├── Turn
│   │   │   ├── Walk
│   │   │   ├── Wave
│   │   ├── test_data -> (soft link to actual validation/testing dataset)
│   │   │   ├── 0.mp4
│   │   │   ├── 1.mp4
│   │   │   ├── 2.mp4
│   │   │   ├── 3.mp4
│   │   │   ├── 4.mp4
│   │   │   ├── ...
│   │   └── list_cvt
│   │       ├── ARID1.1_t1_train_pub.csv
│   │       ├── ARID1.1_t1_validation_gt_pub.csv
│   │       └── mapping_table_t1.txt
├── __init__.py
├── README.md
```
