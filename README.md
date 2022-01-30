# [ICLR 2022] Deep Ensembling with No Overhead for either Training or Testing: The All-Round Blessings of Dynamic Sparsity

<img src="https://github.com/Shiweiliuiiiiiii/FreeTickets/blob/main/FreeTickets.png" width="800" height="300">


**Deep Ensembling with No Overhead for either Training or Testing: The All-Round Blessings of Dynamic Sparsity**<br>
Shiwei Liu, Tianlong Chen, Zahra Atashgahi, Xiaohan Chen, Ghada Sokar, Elena Mocanu, Mykola Pechenizkiy, Zhangyang Wang, Decebal Constantin Mocanu<br>

https://openreview.net/forum?id=RLtqs6pzj1-

Abstract: *The success of deep ensembles on improving predictive performance, uncertainty, and out-of-distribution robustness has been extensively demonstrated in the machine learning literature. Albeit the promising results, naively training multiple deep neural networks and combining their predictions at test lead to prohibitive computational costs and memory requirements. Recently proposed efficient ensemble approaches reach the performance of the traditional deep ensembles with significantly lower costs. However, the training resources required by these approaches are still at least the same as training a single dense model. In this work, we draw a unique connection between sparse neural network training and deep ensembles, yielding a novel efficient ensemble learning framework called **FreeTickets**. Instead of training multiple dense networks and averaging them, we directly train sparse subnetworks from scratch and extract diverse yet accurate subnetworks during this efficient, sparse-to-sparse training. Our framework, FreeTickets, is defined as the ensemble of these relatively cheap sparse subnetworks. Despite being an ensemble method, FreeTickets has even fewer parameters and training FLOPs compared to a single dense model. This seemingly counter-intuitive outcome is due to the ultra training efficiency of dynamic sparse training. FreeTickets improves over the dense baseline in the following criteria: prediction accuracy, uncertainty estimation, out-of-distribution (OoD) robustness, and training/inference efficiency. Impressively, FreeTickets outperforms the naive deep ensemble with ResNet50 on ImageNet using around only 1/5 training FLOPs required by the latter.*

This code base is created by Shiwei Liu s.liu3@tue.nl during his Ph.D. at Eindhoven University of Technology.

## Requirements
Python 3.6, PyTorch v1.5.1, and CUDA v10.2.

## How to Run Experiments

### CIFAR-10/100 Experiments
To train Wide ResNet28-10 on CIFAR10/100 with DST ensemble at sparsity 0.8:

```bash
python main_DST.py --sparse --model wrn-28-10 --data cifar10 --seed 17 --sparse-init ERK \
--update-frequency 1000 --batch-size 128 --death-rate 0.5 --large-death-rate 0.8 \
--growth gradient --death magnitude --redistribution none --epochs 250 --density 0.2

```

To train Wide ResNet28-10 on CIFAR10/100 with EDST ensemble at sparsity 0.8:

```bash
python3 main_EDST.py --sparse --model wrn-28-10 --data cifar10 --nolrsche \
--decay-schedule constant --seed 17 --epochs-explo 150 --model-num 3 --sparse-init ERK \
--update-frequency 1000 --batch-size 128 --death-rate 0.5 --large-death-rate 0.8 \
--growth gradient --death magnitude --redistribution none --epochs 450 --density 0.2
```
[Training module] The training module is controlled by the following arguments:
* `--epochs-explo` - An integer that controls the training epochs of the exploration phase.
* `--model-num` - An integer, the number free tickets to produce.
* `--large-death-rate` - A float, the ratio of parameters to explore for each refine phase.
* `--density` - An float, the density (1-sparsity) level for each free ticket.

To train Wide ResNet28-10 on CIFAR10/100 with PF (prung and finetuning) ensemble at sparsity 0.8:

First, we need train a dense model with:

```bash
python3 main_individual.py  --model wrn-28-10 --data cifar10 --decay-schedule cosine --seed 18 \
--sparse-init ERK --update-frequency 1000 --batch-size 128 --death-rate 0.5 --large-death-rate 0.5 \
--growth gradient --death magnitude --redistribution none --epochs 250 --density 0.2
```

Then, perform pruning and finetuning with:

```bash
pretrain='results/wrn-28-10/cifar10/individual/dense/18.pt'
python3 main_PF.py --sparse --model wrn-28-10 --resume --pretrain $pretrain --lr 0.001 \
--fix --data cifar10 --nolrsche --decay-schedule constant --seed 18 
--epochs-fs 150 --model-num 3 --sparse-init pruning --update-frequency 1000 --batch-size 128 \
--death-rate 0.5 --large-death-rate 0.8 --growth gradient --death magnitude \
--redistribution none --epochs $epoch --density 0.2
```

After finish the training of various ensemble methods, run the following commands for test ensemble:

```bash
resume=results/wrn-28-10/cifar10/density_0.2/EDST/M=3/
python ensemble_freetickets.py --mode predict --resume $resume --dataset cifar10 --model wrn-28-10 \
--seed 18 --test-batch-size 128
```
* `--resume` - An folder path that contains the all the free tickets obtained during training.
* `--mode` - An str that control the evaluation mode, including: predict, disagreement, calibration, KD, and tsne.
