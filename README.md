## Structural Uncertainty

This repository contains the implementation of our work "[Learning Probabilistic Topological Representations Using Discrete Morse Theory](https://openreview.net/pdf?id=cXMHQD-xQas)", **accepted to ICLR 2023 (Spotlight)**. 

## Getting started - compile dipha

**You only need to run cmake & make once**

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
python3 train_model.py  --params params/CREMI_train.json
                        --train_batch 24
```

### ISBI2013
```
python3 train_model.py  --dataset ISBI2013
                        --params params/ISBI2013_train.json
                        --train_batch 24
```

### DRIVE
```
python3 train_model.py  --dataset DRIVE 
                        --params params/DRIVE_train.json
                        --train_batch 8
```

## Finetune from baseline: 

### CREMI
```
python3 train_model.py  --params params/CREMI_train.json 
                        --train_batch 24
                        --pretrain False
                        --resume baseline
```

### ISBI2013
```
python3 train_model.py  --dataset ISBI2013 
                        --params params/ISBI2013_train.json 
                        --train_batch 24 
                        --pretrain False 
                        --resume baseline
```

### DRIVE
```
python3 train_model.py  --dataset DRIVE 
                        --params params/DRIVE_train.json 
                        --train_batch 8 
                        --pretrain False 
                        --resume baseline
```

## Finetune from best: 

### CREMI
```
python3 train_model.py  --params params/CREMI_train.json 
                        --train_batch 24 
                        --pretrain False 
                        --resume best
```

### ISBI2013
```
python3 train_model.py  --dataset ISBI2013 
                        --params params/ISBI2013_train.json 
                        --train_batch 24 
                        --pretrain False 
                        --resume best
```

### DRIVE
```
python3 train_model.py  --dataset DRIVE 
                        --params params/DRIVE_train.json 
                        --train_batch 8 
                        --pretrain False 
                        --resume best
```

## Validation: 

### CREMI
```
python3 infer.py  --params params/CREMI_validation.json
```

### ISBI2013
```
python3 infer.py --dataset ISBI2013 --params params/ISBI2013_validation.json
```

### DRIVE
```
python3 infer.py --dataset DRIVE --params params/DRIVE_validation.json
```

## Citation
Please consider citing our paper if you find it useful.
```

@inproceedings{hu2021topology,
  title={Topology-Aware Segmentation Using Discrete Morse Theory},
  author={Hu, Xiaoling and Wang, Yusu and Fuxin, Li and Samaras, Dimitris and Chen, Chao},
  booktitle={International Conference on Learning Representations},
  year={2021}
}

@inproceedings{hu2023learning,
  title={Learning Probabilistic Topological Representations Using Discrete Morse Theory},
  author={Hu, Xiaoling and Samaras, Dimitris and Chen, Chao},
  booktitle={International Conference on Learning Representations},
  year={2023}
}
