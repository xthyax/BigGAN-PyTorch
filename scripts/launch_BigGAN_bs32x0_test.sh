#!/bin/bash
python train.py\
--data_root Dataset\Train \
--weight_root Model\Pretrained \
--logs_root Logs\log_GAN_1
--samples_root Logs\log_GAN_1\samples