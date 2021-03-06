## Experiment result
### Results on Isolated Chinese Sign Language Dataset
Isolated Chinese Sign Language Dataset is a dataset for SLR, which has 500 action classes performed by 50 different signers
(repeat 5 times)

| Model         |   Pretrained    ｜   top-1 acc   |   top-5 acc   |
| ------------- | --------------- ｜ ------------- | ------------- |
| R3D           |     True        ｜     93.70     |     99.74     |
| R(2+1)D       |     True        ｜     97.43     |     99.93     |
| P3D(b=1)      |     None        ｜     77.75     |     96.85     |

### Analysis of comlexity
- Currently on Isolated Chinese Sign Language Dataset

| Model         |   Backbone     |       Total Params     |    FLOPs(b=2)    |
| ------------- | -------------- | ---------------------- | ---------------- |
| R3D           |   Resnet18     |     127.6M (466.36k)   |      12.39G      |
| R(2+1)D       |   Resnet18     |     120.5M (31.56M)    |     325.16G      |
| P3D           |   Resnet18     |     65.3M (8.73M)      |      72.21G      |