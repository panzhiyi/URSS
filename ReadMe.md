# Scribble-Supervised Semantic Segmentation by Random Walk on Neural Representation and Self-Supervision on Neural Eigenspace
### Zhiyi Pan
All the computations are carried out on NVIDIA TITAN RTX GPUs.

##### dataset

*scribble_shrink* and *scribble_drop* are available at [here](https://drive.google.com/drive/folders/1q2PvbQVOdIY9S-qjh85ohM66svzp9wnp).  The *scribble_sup* dataset can be downloaded on [jifengdai.org/downloads/scribble_sup/](https://jifengdai.org/downloads/scribble_sup/).

##### environment

```
pip install -r requirements.txt
```

##### baseline

Please modify the dataset file path in **train_seg_baseline.sh** and run:

```
sh train_seg_baseline.sh
```

##### first-stage training with Uncertainty Reduction on Neural Representation

Please modify the dataset file path in **train_seg_UR.sh** and run:

```
sh train_seg_UR.sh
```

the model will be saved in ./runs/ 

##### second-stage training to refine the model with soft self supervision loss

Please modify the model(obtained by first-stage training) file path in **train_seg_SS.sh** and run: 

```
sh train_seg_SS.sh
```

##### evaluate

Please modify the model(obtained by second-stage training) file path and save path in **evaluate.sh** and run: 

```
sh evaluate.sh
```

