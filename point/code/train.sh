#!/bin/bash
python solve.py --year=2012 --output='qilu' --train-img='/media/sensetime/64C43C54C43C2AA6/lmdb/train_im_full' --train-gt='/media/sensetime/64C43C54C43C2AA6/lmdb/train_gt_point' --val-img='/media/sensetime/64C43C54C43C2AA6/lmdb/val_im_full' --val-gt='/media/sensetime/64C43C54C43C2AA6/lmdb/val_gt_full' --expectation --location --constraint --classes --objectness 
