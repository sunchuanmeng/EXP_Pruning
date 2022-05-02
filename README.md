# EXP_Pruning
EXP way : Complete pruning using the expectation scaling factor

## Running Code

In this code, you can run our models on CIFAR-10 and ImageNet dataset. The code has been tested by Python 3.6, Pytorch 1.6 and CUDA 10.2 on Windows 10.

## parser
```shell
/data_dir/ : Dataset storage address
/dataset/ ： dataset - CIFAR10 or Imagenet
/lr/ ： initial learning rate
/lr_decay_step/ ： learning rate decay step
/resume/ ： load the model from the specified checkpoint
/resume_mask/ ： After the program is interrupted, the task file can be used to continue running
/job_dir/ ： The directory where the summaries will be stored
/epochs/ ： The num of epochs to fine-tune
/start_cov/ ： The num of conv to start prune
/compress_rate/ ： compress rate of each conv
/arch/ ： The architecture to prune
/pruning_way/ ： The chosen pruning method,A:bn priority pruning ; B:Full-layer redundant pruning
```

## Model Training

For the ease of reproducibility. we provide some of the experimental results and the corresponding pruned rate of every layer as belows:

##### 1. VGG-16

| Flops     | Accuracy  |way and Model                | Flops     | Accuracy  |way and Model                |
|-----------|-----------|-----------------------------|-----------|-----------|-----------------------------|
| 100%      | 93.96%    |[VGG16](https://drive.google.com/file/d/1q_uzAvsAPyQxdaeYWy9NkpnRxwWRr_zc/view?usp=sharing)
| 53.1%     | 93.64%    |[A](https://drive.google.com/file/d/1S4he_cv9NGbtT3HL13uQ5qZQ5_r_3W9N/view?usp=sharing)| 60.8%     | 93.73%    |[B](https://drive.google.com/file/d/198ei_zfehnHD0lidqqhn8eu03BlaE6Ag/view?usp=sharing)
| 53.1%     | 93.55%    |[B](https://drive.google.com/file/d/1Df7LM3kNULiqhT97TXgAlcvqETcJXwzK/view?usp=sharing)| 70.9%     | 93.25%    |[A](https://drive.google.com/file/d/1hxmyNi-nPra9QGfqxBdAWojIF5kG5uXi/view?usp=sharing)
| 60.8%     | 93.50%    |[A](https://drive.google.com/file/d/1qs1cFQBko9HdNno7XeybT7xVTPH-hAGl/view?usp=sharing)| 70.9%     | 93.54%    |[B](https://drive.google.com/file/d/19YHyQtdO_DerQBquDut8FVRGQpIUHF5j/view?usp=sharing)

##### 2. ResNet-56

| Flops     | Accuracy  |way and Model                |
|-----------|-----------|-----------------------------|
| 100%      | 93.26%    |[Res56](https://drive.google.com/file/d/1WE83j7rlKlCp-tslSL6hS-d_mJe4ZQ2r/view?usp=sharing)
| 53.5%     | 93.54%    |[A](https://drive.google.com/file/d/1WhW7O0-GDvZCLpwvXdCLVWK5kddgk94z/view?usp=sharing)
| 53.5%     | 93.82%    |[B](https://drive.google.com/file/d/1qs1cFQBko9HdNno7XeybT7xVTPH-hAGl/view?usp=sharing)
| 60.5%     | 93.48%    |[A](https://drive.google.com/file/d/1qs1cFQBko9HdNno7XeybT7xVTPH-hAGl/view?usp=sharing)
| 60.5%     | 93.63%    |[B](https://drive.google.com/file/d/198ei_zfehnHD0lidqqhn8eu03BlaE6Ag/view?usp=sharing)
| 71.9%     | 92.91%    |[A](https://drive.google.com/file/d/1hxmyNi-nPra9QGfqxBdAWojIF5kG5uXi/view?usp=sharing)
| 71.9%     | 93.03%    |[B](https://drive.google.com/file/d/19YHyQtdO_DerQBquDut8FVRGQpIUHF5j/view?usp=sharing)




