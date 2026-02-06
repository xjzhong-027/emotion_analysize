#! /bin/bash
python ./main.py
  
"""
python main.py \
  --dataset_type 2015 \
  --text_model_name roberta \
  --image_model_name vit \
  --batch_size 8 \
  --epochs 60 \
  --alpha 0.6 \
  --beta 0.6 \
  --lr 2.0e-05
"""

"""
python main.py \
  --dataset_type 2017 \
  --text_model_name roberta-large \
  --image_model_name swin \
  --batch_size 16 \
  --epochs 80 \
  --alpha 0.515 \
  --beta 0.675 \
  --lr 1.55e-05
"""