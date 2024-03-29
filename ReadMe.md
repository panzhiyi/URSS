# Scribble-Supervised Semantic Segmentation by Uncertainty Reduction on Neural Representation and Self-Supervision on Neural Eigenspace
[Zhiyi Pan](https://github.com/panzhiyi), [Peng Jiang*](https://github.com/sdujump), Yunhai Wang, Changhe Tu, Anthony G. Cohn

The [paper](https://openaccess.thecvf.com/content/ICCV2021/html/Pan_Scribble-Supervised_Semantic_Segmentation_by_Uncertainty_Reduction_on_Neural_Representation_and_ICCV_2021_paper.html) has been accepted by ICCV2021.

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

All the computations are carried out on NVIDIA TITAN RTX GPUs.

