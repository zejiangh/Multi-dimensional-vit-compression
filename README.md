# Multi-Dimensional Model Compression of Vision Transformer

This repository contains the PyTorch implementation of the paper [Multi-Dimensional Model Compression of Vision Transformer](https://openaccess.thecvf.com/content/CVPR2022W/EVW/papers/Hou_Multi-Dimensional_Vision_Transformer_Compression_via_Dependency_Guided_Gaussian_Process_Search_CVPRW_2022_paper.pdf).

### Highlight
Vision transformers (ViT) have recently attracted considerable attentions, but the huge computational cost remains an issue for practical deployment. Previous ViT pruning methods tend to prune the model along one dimension solely, which may suffer from excessive reduction and lead to sub-optimal model quality. In contrast, we advocate a multi-dimensional ViT compression paradigm, and propose to harness the redundancy reduction from attention head, neuron and sequence dimensions jointly. We firstly propose a statistical dependence based pruning criterion that is generalizable to different dimensions for identifying deleterious components. Moreover, we cast the multi-dimensional compression as an optimization, learning the optimal pruning policy across the three dimensions that maximizes the compressed model’s accuracy under a computational budget. The problem is solved by the Expected Improvement (EI) algorithm. Experimental results show that our method effectively reduces the computational cost of various ViT models.

* Dedependency based pruning criterion is implemented in `dependency_criterion.py`

* Expected Improvement based search algorithm is implemented in `expected_improvement.py`


### Dependency

* This repo was built upon the [DeiT repo](https://github.com/facebookresearch/deit). Installation and preparation follow that repo.

### Prepare the dataset

* We use ImageNet-1K. [Download the images](http://image-net.org/download-images).

* Extract the training data:
```Shell
mkdir train && mv ILSVRC2012_img_train.tar train/ && cd train
tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
cd ..
```

* Extract the validation data and move the images to subfolders:
```Shell
mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val && tar -xvf ILSVRC2012_img_val.tar
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
```

### Three steps in our method

Example of applying our method to compress DeiT-Small on ImageNet-1K using 8 GPUs:

#### Pretrain

```Shell
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py \
    --model deit_small_patch16_224 \
    --batch-size 128 \
    --data-path /data/imagenet \
    --output_dir ./logs \
    --dist-eval
```

* Change the `data-path` to the location where ImageNet dataset is stored.

#### Search:

```Shell
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py \
    --model deit_small_patch16_224 \
    --batch-size 128 \
    --data-path /data/imagenet \
    --output_dir ./logs \
    --dist-eval \
    --resume ./logs/deit_small.pth \
    --eval \
    --search \
    --search_epoch 100 \
    --search_flops 1.84 \
    --search_population 100
```

* Change the `resume` to the location where pretrained model is stored.
* The target FLOPs of the compressed model is specified in `--search_flops`
* Number of search iteratinos is specified in `--search_epoch`
* Number of initial sample population to fit GP is specified in `--search_population`

#### Prune and finetune:

```Shell
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py \
    --model deit_small_patch16_224 \
    --batch-size 128 \
    --data-path /data/imagenet \
    --output_dir ./logs \
    --dist-eval \
    --resume ./logs/deit_small.pth \
    --neuron_pruning \
    --head_pruning \
    --token_pruning \
    --prune_finetune
```

* Change the `resume` to the location where pretrained model is stored.

### Citation
If you find this repository helpful, please consider citing:
```Shell
@inproceedings{hou2022multi,
  title={Multi-Dimensional Vision Transformer Compression via Dependency Guided Gaussian Process Search},
  author={Hou, Zejiang and Kung, Sun-Yuan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={3669--3678},
  year={2022}
}
```


