#/bin/bash
python ./train_seg_SS.py \
 train_seg_SS.sh \
 0,1 \
 50 \
 /path/to/data \
 VOC2012 \
 21 \
 4 \
 RW \
 1 \
 pascal_2012_scribble \
 16 \
 1e-5 \
 5e-4 \
 0.9 \
 50 \
 1 \
 1 \
 /path/to/stage1/model \
 random \
 P