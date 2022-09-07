## Pytorch-MultiGPUs-Finetune - Pytorch with multigpus practices

This is the repository that tries to use multigpus to finetune the pretrained models.

## Usage (Training and Testing)

```
cifar_multigpus_finetune.py [-h]
                            [--arch {resnet152,regnet_x_32gf,wide_resnet101_2}]
                            [--dataset {cifar10,cifar100}]
                            [--dataset_path DATASET_PATH] 
                            [--cp CP]
                            [--E E] 
                            [--LR LR] 
                            [--BS BS] 
                            [--TBS TBS]
                            [--pretrained PRETRAINED]
                            [--finetune FINETUNE] 
                            [--mode {train,test}]
```

Here are the explanation of the arguments:

* ```arch```: A string. Specify the architecture of the model. Input can be [ resnet152 | regnet_x_32gf | wide_resnet101_2 ].
* ```dataset```: A string. Specify the datasets. Input can be [ cifar10 | cifar100 ].
* ```dataset_path```: A string. The path of the datasets. 
* ```cp```: A string. The path of the checkpoints. Only needed in testing phase.
* ```E```: An int. Specify the training epochs for training the model.
* ```LR```: A float. Specify the learning rate for training the model.
* ```BS```: An int. Specify the batch size on training dataset for training or testing the model.
* ```TBS```: An int. Specify the batch size on testing dataset for training or testing the model.
* ```pretrained```: A boolean. Specify whether to use pretrained model or not.
* ```finetune```: A boolean. Specify whether to finetune the model or not.
* ```mode```: A string. Training or testing the model. Input can be [ train | test ].

If you want to use multigpus to train your model, please specific which gpus can be used in the running script. For example:
```
CUDA_VISIBLE_DEVICES=0,1,2,3
```

You can train the wide resnet101 with GPU:0,1, then you can use the script (with pretrained && finetune) below (Take Cifar10 as example):

```python
CUDA_VISIBLE_DEVICES=0,1 python cifar_multigpus_finetune.py --arch wide_resnet101_2 --dataset cifar10 --dataset_path [please_input_the_path_of_the_datasets_here] --LR 0.001 --E 20 --BS 64 --pretrained True --finetune True --mode train
```

You can test your finetuned model by using the script below (Take Cifar10 as example):

```python
CUDA_VISIBLE_DEVICES=0,1 python cifar_multigpus_finetune.py --arch wide_resnet101_2 --dataset cifar10 --dataset_path [please_input_the_path_of_the_datasets_here] --cp [please_input_the_path_of_the_finetuned_weights_here] --BS 64 --TBS 64 --mode test
```

## Cifar10 Finetune Results (Train on 2 GTX2080Ti GPUs)

| Model | Epochs | Learning Rate | Batch Size | Training Data Acc. | Testing Data Acc. |
| :----: | :----: | :----: | :----: | :----: | :----: |
| wide_resnet101_2 | 20 | 0.001 | 64 | 99.964% | 98.29% |



## Note
You can add more pretrained model by yourself, please refer to the website of Pytorch.
https://pytorch.org/vision/stable/models.html 