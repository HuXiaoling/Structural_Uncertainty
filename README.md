## Probabilistic-Unet_Pytorch

conda activate 3dunet

### Getting started - run dipha

%% You only need to run cmake & make once

* (dipha-graph-recon folder)

run the following commands in this folder to build dipha

> rm -rf build/ (this removes my build directory)

> mkdir build

> cd build

> cmake ..

> make

now that dipha is built, you are ready

## Training from scratch: 

### CREMI
```
CUDA_VISIBLE_DEVICES=0 python3 train_model.py   --params params/CREMI_train.json 
                                                --train_batch 24
```

### ISBI2013
```
CUDA_VISIBLE_DEVICES=1 python3 train_model.py --dataset ISBI2013 
                                              --params params/ISBI2013_train.json 
                                              --train_batch 24
```

### DRIVE
```
CUDA_VISIBLE_DEVICES=2 python3 train_model.py --dataset DRIVE 
                                              --params params/DRIVE_train.json 
                                              --train_batch 8
```

## Finetune from baseline: 

### CREMI
```
CUDA_VISIBLE_DEVICES=0 python3 train_model.py --params params/CREMI_train.json 
                                              --train_batch 24 
                                              --pretrain False 
                                              --resume baseline
```

### ISBI2013
```
CUDA_VISIBLE_DEVICES=1 python3 train_model.py --dataset ISBI2013 
                                              --params params/ISBI2013_train.json 
                                              --train_batch 24 
                                              --pretrain False 
                                              --resume baseline
```

### DRIVE
```
CUDA_VISIBLE_DEVICES=2 python3 train_model.py --dataset DRIVE 
                                              --params params/DRIVE_train.json 
                                              --train_batch 8 
                                              --pretrain False 
                                              --resume baseline
```

## Finetune from best: 

### CREMI
```
CUDA_VISIBLE_DEVICES=0 python3 train_model.py --params params/CREMI_train.json 
                                              --train_batch 24 
                                              --pretrain False 
                                              --resume best
```

### ISBI2013
```
CUDA_VISIBLE_DEVICES=1 python3 train_model.py --dataset ISBI2013 
                                              --params params/ISBI2013_train.json 
                                              --train_batch 24 
                                              --pretrain False 
                                              --resume best
```

### DRIVE
```
CUDA_VISIBLE_DEVICES=2 python3 train_model.py --dataset DRIVE 
                                              --params params/DRIVE_train.json 
                                              --train_batch 8 
                                              --pretrain False 
                                              --resume best
```

## Validation: 

### CREMI
```
CUDA_VISIBLE_DEVICES=0 numactl --physcpubind=40-50 python3 infer.py --params params/CREMI_validation.json
```

### ISBI2013
```
CUDA_VISIBLE_DEVICES=1 python3 infer.py --dataset ISBI2013 --params params/ISBI2013_validation.json
```

### DRIVE
```
CUDA_VISIBLE_DEVICES=2 python3 infer.py --dataset DRIVE --params params/DRIVE_validation.json
```
