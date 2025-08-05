## baseline

| Dataset               | Segmentation Score (Dice) | Classification Score (Accuracy) |
|-----------------------|--------------------------|--------------------------------|
| BUS-BRA               | 0.699                    | 0.633                          |
| BUSIS                 | 0.817                    | -                              |
| BUSI                  | 0.547                    | 0.712                          |
| CAMUS                 | 0.883                    | -                              |
| DDTI                  | 0.491                    | -                              |
| Fetal_HC              | 0.900                    | -                              |
| KidneyUS              | 0.772                    | -                              |
| Appendix              | -                        | 0.551                          |
| Fatty-Liver           | -                        | 0.636                          |
| private_Thyroid       | 0.355                    | -                              |
| private_Kidney        | 0.673                    | -                              |
| private_Fetal_Head    | 0.845                    | -                              |
| private_Cardiac       | 0.727                    | -                              |
| private_Breast        | 0.815                    | 0.867                          |
| private_Breast_luminal| 0.696                    | 0.583                          |
| private_Liver         | -                        | 0.364                          |
| private_Appendix      | -                        | 0.500                          |

## trainable Loss weight
| Dataset               | Segmentation Score (Dice) | Classification Score (Accuracy) |
|-----------------------|--------------------------|--------------------------------|
| BUS-BRA               | 0.451                    | 0.633                          |
| BUSIS                 | 0.688                    | -                              |
| BUSI                  | 0.402                    | 0.712                          |
| CAMUS                 | 0.662                    | -                              |
| DDTI                  | 0.332                    | -                              |
| Fetal_HC              | 0.752                    | -                              |
| KidneyUS              | 0.586                    | -                              |
| Appendix              | -                        | 0.551                          |
| Fatty-Liver           | -                        | 0.636                          |
| private_Thyroid       | 0.192                    | -                              |
| private_Kidney        | 0.520                    | -                              |
| private_Fetal_Head    | 0.671                    | -                              |
| private_Cardiac       | 0.554                    | -                              |
| private_Breast        | 0.678                    | 0.867                          |
| private_Breast_luminal| 0.562                    | 0.583                          |
| private_Liver         | -                        | 0.364                          |
| private_Appendix      | -                        | 0.500                          |

## train on private datasets
| Dataset               | Segmentation (Dice) | Classification (Accuracy) |
|-----------------------|---------------------|--------------------------|
| BUS-BRA               | 0.2230              | 0.3670                   |
| BUSIS                 | 0.3831              | -                        |
| BUSI                  | 0.1590              | 0.2879                   |
| CAMUS                 | 0.3699              | -                        |
| DDTI                  | 0.0000              | -                        |
| Fetal_HC              | 0.6432              | -                        |
| KidneyUS              | 0.5849              | -                        |
| private_Thyroid       | 0.0136              | -                        |
| private_Kidney        | 0.5584              | -                        |
| private_Fetal_Head    | 0.6314              | -                        |
| private_Cardiac       | 0.4154              | -                        |
| private_Breast_luminal| 0.2214              | 0.5833                   |
| private_Breast        | 0.6065              | 0.1333                   |
| Appendix              | -                   | 0.5510                   |
| Fatty-Liver           | -                   | 0.6364                   |
| private_Liver         | -                   | 0.3636                   |
| private_Appendix      | -                   | 0.5000                   |

## 修改保存pth文件的逻辑，调整平衡采样的硬编码概率顺序，修改epoch和分割损失的系数后重新训的结果

| Dataset               | Segmentation (Dice) | Classification (Accuracy) |
|-----------------------|---------------------|--------------------------|
| BUS-BRA               | 0.6389              | 0.6330                   |
| BUSIS                 | 0.8345              | -                        |
| BUSI                  | 0.5347              | 0.7121                   |
| CAMUS                 | 0.8868              | -                        |
| DDTI                  | 0.4444              | -                        |
| Fetal_HC              | 0.9115              | -                        |
| KidneyUS              | 0.7565              | -                        |
| private_Thyroid       | 0.2713              | -                        |
| private_Kidney        | 0.7674              | -                        |
| private_Fetal_Head    | 0.8971              | -                        |
| private_Cardiac       | 0.6645              | -                        |
| private_Breast_luminal| 0.7173              | 0.5833                   |
| private_Breast        | 0.8921              | 0.8667                   |
| Appendix              | -                   | 0.5510                   |
| Fatty-Liver           | -                   | 0.6364                   |
| private_Liver         | -                   | 0.3636                   |
| private_Appendix      | -                   | 0.5000                   |

## 与baseline比较
| Dataset               | trial_uusic_3 (Dice) | 对比数据 (Dice) | Δ Dice | trial_uusic_3 (Acc) | 对比数据 (Acc) | Δ Acc |
|-----------------------|----------------------|-----------------|--------|---------------------|----------------|-------|
| BUS-BRA               | 0.6389               | 0.699           | -0.060 | 0.6330              | 0.633          | 0.000 |
| BUSIS                 | 0.8345               | 0.817           | +0.018 | -                   | -              | -     |
| BUSI                  | 0.5347               | 0.547           | -0.012 | 0.7121              | 0.712          | +0.001|
| CAMUS                 | 0.8868               | 0.883           | +0.004 | -                   | -              | -     |
| DDTI                  | 0.4444               | 0.491           | -0.047 | -                   | -              | -     |
| Fetal_HC              | 0.9115               | 0.900           | +0.012 | -                   | -              | -     |
| KidneyUS              | 0.7565               | 0.772           | -0.016 | -                   | -              | -     |
| private_Thyroid       | 0.2713               | 0.355           | -0.084 | -                   | -              | -     |
| private_Kidney        | 0.7674               | 0.673           | +0.094 | -                   | -              | -     |
| private_Fetal_Head    | 0.8971               | 0.845           | +0.052 | -                   | -              | -     |
| private_Cardiac       | 0.6645               | 0.727           | -0.063 | -                   | -              | -     |
| private_Breast        | 0.8921               | 0.815           | +0.077 | 0.8667              | 0.867          | -0.000|
| private_Breast_luminal| 0.7173               | 0.696           | +0.021 | 0.5833              | 0.583          | +0.000|
| Appendix              | -                    | -               | -      | 0.5510              | 0.551          | 0.000 |
| Fatty-Liver           | -                    | -               | -      | 0.6364              | 0.636          | +0.000|
| private_Liver         | -                    | -               | -      | 0.3636              | 0.364          | -0.000|
| private_Appendix      | -                    | -               | -      | 0.5000              | 0.500          | 0.000 |

## 增加Appendix，CAMUS，DDTI和Thyroid数据大小，同时调整分类任务的学习率，重新训练结果
### 和上一次实验以及baseline的比较
| Dataset               | uusic4 (Dice) | uusic3 (Dice) | 原始数据 (Dice) | Δ(u4 vs u3) | Δ(u4 vs 原始) | uusic4 (Acc) | uusic3 (Acc) | 原始数据 (Acc) | Δ(u4 vs u3) | Δ(u4 vs 原始) |
|-----------------------|---------------|---------------|------------------|-------------|----------------|--------------|--------------|-----------------|-------------|----------------|
| BUS-BRA               | 0.6945        | 0.6389        | 0.6990           | +0.0556     | -0.0045        | 0.6330       | 0.6330       | 0.6330          | 0.0000      | 0.0000         |
| BUSIS                 | 0.8528        | 0.8345        | 0.8170           | +0.0183     | +0.0358        | -            | -            | -               | -           | -              |
| BUSI                  | 0.5852        | 0.5347        | 0.5470           | +0.0505     | +0.0382        | 0.7121       | 0.7121       | 0.7120          | 0.0000      | +0.0001        |
| CAMUS                 | 0.9101        | 0.8868        | 0.8830           | +0.0233     | +0.0271        | -            | -            | -               | -           | -              |
| DDTI                  | 0.5591        | 0.4444        | 0.4910           | +0.1147     | +0.0681        | -            | -            | -               | -           | -              |
| Fetal_HC              | 0.9041        | 0.9115        | 0.9000           | -0.0074     | +0.0041        | -            | -            | -               | -           | -              |
| KidneyUS              | 0.7946        | 0.7565        | 0.7720           | +0.0381     | +0.0226        | -            | -            | -               | -           | -              |
| private_Thyroid       | 0.4141        | 0.2713        | 0.3550           | +0.1428     | +0.0591        | -            | -            | -               | -           | -              |
| private_Kidney        | 0.7943        | 0.7674        | 0.6730           | +0.0269     | +0.1213        | -            | -            | -               | -           | -              |
| private_Fetal_Head    | 0.8703        | 0.8971        | 0.8450           | -0.0268     | +0.0253        | -            | -            | -               | -           | -              |
| private_Cardiac       | 0.7461        | 0.6645        | 0.7270           | +0.0816     | +0.0191        | -            | -            | -               | -           | -              |
| private_Breast_luminal| 0.7587        | 0.7173        | 0.6960           | +0.0414     | +0.0627        | 0.5833       | 0.5833       | 0.5830          | 0.0000      | +0.0003        |
| private_Breast        | 0.8822        | 0.8921        | 0.8150           | -0.0099     | +0.0672        | 0.8667       | 0.8667       | 0.8670          | 0.0000      | -0.0003        |
| Appendix              | -             | -             | -                | -           | -              | 0.7872       | 0.5510       | 0.5510          | +0.2362     | +0.2362        |
| Fatty-Liver           | -             | -             | -                | -           | -              | 0.6364       | 0.6364       | 0.6360          | 0.0000      | +0.0004        |
| private_Liver         | -             | -             | -                | -           | -              | 0.3636       | 0.3636       | 0.3640          | 0.0000      | -0.0004        |
| private_Appendix      | -             | -             | -                | -           | -              | 0.5000       | 0.5000       | 0.5000          | 0.0000      | 0.0000         |



## 增加数据增强，和cardiac数据,分类和分割分别使用不同的学习率调度器 结果


# 目前暂时先调整分类部分

- debug3：lr=1e-5,weight_decay=0.05，无prompt 分类恢复正常
- debug4：lr=1e-5,weight_decay=0.01，无prompt 分类效果有一些提升
- debug5：使用全部的分类数据，单独训分类器，lr=1e-5,weight_decay=0.01，无prompt

| Dataset               | Classification Score (Accuracy) | Timestamp               |
|-----------------------|----------------------------------|-------------------------|
| Appendix              | 0.7943                          | 2025-07-24 01:36:28     |
| BUS-BRA               | 0.7234                          | 2025-07-24 01:36:38     |
| BUSI                  | 0.6667                          | 2025-07-24 01:36:42     |
| Fatty-Liver           | 0.6545                          | 2025-07-24 01:36:45     |
| private_Liver         | 0.3636                          | 2025-07-24 01:36:46     |
| private_Breast_luminal| 0.5833                          | 2025-07-24 01:36:47     |
| private_Breast        | 0.7333                          | 2025-07-24 01:36:48     |
| private_Appendix      | 0.3750                          | 2025-07-24 01:36:49     |

- debug6：使用全部的分类数据，单独训分类器，lr=1e-5,weight_decay=0.01，加入prompt

| Dataset               | Classification Score (Accuracy) | Timestamp               |
|-----------------------|----------------------------------|-------------------------|
| Appendix              | 0.7801                          | 2025-07-24 01:14:39     |
| BUS-BRA               | 0.6543                          | 2025-07-24 01:14:49     |
| BUSI                  | 0.7576                          | 2025-07-24 01:14:53     |
| Fatty-Liver           | 0.8727                          | 2025-07-24 01:14:56     |
| private_Liver         | 0.5455                          | 2025-07-24 01:14:57     |
| private_Breast_luminal| 0.5417                          | 2025-07-24 01:14:58     |
| private_Breast        | 0.8000                          | 2025-07-24 01:14:59     |
| private_Appendix      | 0.2500                          | 2025-07-24 01:15:00     |

-  ⭐[有效实验]cfff_debug_7:使用全部的分类数据，分割和分类，lr=1e-5,weight_decay=0.01，加入prompt： 分类和分割模型都训起来了

## Epoch 179 实验结果

### 分割任务性能 (Segmentation)
| Dataset               | Metric (Dice) | 
|-----------------------|---------------|
| BUS-BRA               | 0.5182        |
| BUSIS                 | 0.7302        |
| BUSI                  | 0.4053        |
| CAMUS                 | 0.9132        |
| DDTI                  | 0.6114        |
| Fetal_HC              | 0.8332        |
| KidneyUS              | 0.6489        |
| private_Thyroid       | 0.3021        |
| private_Kidney        | 0.6866        |
| private_Fetal_Head    | 0.7551        |
| private_Cardiac       | 0.6277        |
| private_Breast_luminal| 0.6077        |
| private_Breast        | 0.8335        |

### 分类任务性能 (Classification)
| Dataset               | Metric (Accuracy) |
|-----------------------|-------------------|
| BUS-BRA               | 0.7394           |
| BUSI                  | 0.7727           |
| private_Breast_luminal| 0.7083           |
| private_Breast        | 0.6667           |
| Appendix              | 0.7872           |
| Fatty-Liver           | 0.8000           |
| private_Liver         | 0.4545           |
| private_Appendix      | 0.7500           |
 
| Dataset               | Task Type       | Epoch 179 (Dice/Acc) | Epoch 266 (Dice/Acc) | Δ (Epoch266 - Epoch179) | Trend  |
|-----------------------|-----------------|-----------------------|-----------------------|--------------------------|--------|
| BUS-BRA               | Segmentation    | 0.5182               | 0.5993               | +0.0811                  | ↑↑     |
|                       | Classification  | 0.7394               | 0.7553               | +0.0159                  | ↑      |
| BUSIS                 | Segmentation    | 0.7302               | 0.7478               | +0.0176                  | ↑      |
| BUSI                  | Segmentation    | 0.4053               | 0.4698               | +0.0645                  | ↑↑     |
|                       | Classification  | 0.7727               | 0.7424               | -0.0303                  | ↓      |
| CAMUS                 | Segmentation    | 0.9132               | 0.9196               | +0.0064                  | ↑      |
| DDTI                  | Segmentation    | 0.6114               | 0.6512               | +0.0398                  | ↑↑     |
| Fetal_HC              | Segmentation    | 0.8332               | 0.8432               | +0.0100                  | ↑      |
| KidneyUS              | Segmentation    | 0.6489               | 0.6749               | +0.0260                  | ↑      |
| private_Thyroid       | Segmentation    | 0.3021               | 0.3503               | +0.0482                  | ↑↑     |
| private_Kidney        | Segmentation    | 0.6866               | 0.6801               | -0.0065                  | ↓      |
| private_Fetal_Head    | Segmentation    | 0.7551               | 0.7162               | -0.0389                  | ↓↓     |
| private_Cardiac       | Segmentation    | 0.6277               | 0.6689               | +0.0412                  | ↑↑     |
| private_Breast_luminal| Segmentation    | 0.6077               | 0.6851               | +0.0774                  | ↑↑     |
|                       | Classification  | 0.7083               | 0.6250               | -0.0833                  | ↓↓     |
| private_Breast        | Segmentation    | 0.8335               | 0.8166               | -0.0169                  | ↓      |
|                       | Classification  | 0.6667               | 0.6667               | 0.0000                   | →      |
| Appendix              | Classification  | 0.7872               | 0.7447               | -0.0425                  | ↓↓     |
| Fatty-Liver           | Classification  | 0.8000               | 0.8182               | +0.0182                  | ↑      |
| private_Liver         | Classification  | 0.4545               | 0.4545               | 0.0000                   | →      |
| private_Appendix      | Classification  | 0.7500               | 0.5000               | -0.2500                  | ↓↓↓    |

- train_debug_7:使用不同的学习率分别优化分类和分割  **不work**

72个epoch的结果,分类训崩,分割效果很差

| Dataset               | Segmentation (Dice) | Classification (Accuracy) |
|-----------------------|----------------------|--------------------------|
| BUS-BRA               | 0.4163               | 0.6330                   |
| BUSIS                 | 0.7202               | -                        |
| BUSI                  | 0.4455               | 0.7121                   |
| CAMUS                 | 0.8692               | -                        |
| DDTI                  | 0.3472               | -                        |
| Fetal_HC              | 0.8396               | -                        |
| KidneyUS              | 0.7056               | -                        |
| private_Thyroid       | 0.1247               | -                        |
| private_Kidney        | 0.6932               | -                        |
| private_Fetal_Head    | 0.7943               | -                        |
| private_Cardiac       | 0.7107               | -                        |
| private_Breast_luminal| 0.5333               | 0.5833                   |
| private_Breast        | 0.7691               | 0.8667                   |
| Appendix              | -                    | 0.7872                   |
| Fatty-Liver           | -                    | 0.6364                   |
| private_Liver         | -                    | 0.3636                   |
| private_Appendix      | -                    | 0.5000                   |




- trail_debug_8: 分类优化器优化骨干网络和分类头lr=0.00001，分割优化器优化仅与分割相关的部分lr=0.0001，同时加入图像标准化  **不work，val和test没加标准化**

- trail_debug_9: 用一个优化器优化，lr=0.00001，同时加入图像标准化, 调高DDTI数据集和Thyroid数据集的比例  **不work，val和test没加标准化**

- trail_debug_10：用预训练权重全部初始化，后续两个opt分别训练分类头和分类头。

| Dataset(seg)                | seg_result               | cls_result               |
|------------------------|--------------------------|--------------------------|
| BUS-BRA                | 0.5219973499214556       | 0.616                    |
| BUSIS                  | 0.7493430383312801       | -                        |
| BUSI                   | 0.5184983258470196       | 0.7674418604651163       |
| CAMUS                  | 0.8661335436637926       | -                        |
| DDTI                   | 0.32825335463227207      | -                        |
| Fetal_HC               | 0.8913143772439396       | -                        |
| KidneyUS               | 0.6325336490088369       | -                        |
| private_Thyroid        | 0.2084745896738239       | -                        |
| private_Kidney         | 0.7377713497704271       | -                        |
| private_Fetal_Head     | 0.7652719759000357       | -                        |
| private_Cardiac        | 0.793924406015925        | -                        |
| private_Breast_luminal | 0.6106602569711272       | 0.7021276595744681       |
| private_Breast         | 0.8034937683528423       | 0.7333333333333333       |
| Appendix               | -                        | 0.6678571428571428       |
| Fatty-Liver            | -                        | 0.6818181818181818       |
| private_Liver          | -                        | 0.7                      |
| private_Appendix       | -                        | 0.6923076923076923       |


| Dataset(cls)               | Epoch 179 (Dice) | trail_debug_10 (Dice) | Δ (Current - Epoch179) | Trend  |
|-----------------------|------------------|-----------------------|------------------------|--------|
| BUS-BRA               | 0.5182           | 0.5220                | +0.0038                | ↑      |
| BUSIS                 | 0.7302           | 0.7493                | +0.0191                | ↑      |
| BUSI                  | 0.4053           | 0.5185                | +0.1132                | ↑↑↑    |
| CAMUS                 | 0.9132           | 0.8661                | -0.0471                | ↓↓     |
| DDTI                  | 0.6114           | 0.3283                | -0.2831                | ↓↓↓    |
| Fetal_HC              | 0.8332           | 0.8913                | +0.0581                | ↑↑     |
| KidneyUS              | 0.6489           | 0.6325                | -0.0164                | ↓      |
| private_Thyroid       | 0.3021           | 0.2085                | -0.0936                | ↓↓     |
| private_Kidney        | 0.6866           | 0.7378                | +0.0512                | ↑↑     |
| private_Fetal_Head    | 0.7551           | 0.7653                | +0.0102                | ↑      |
| private_Cardiac       | 0.6277           | 0.7939                | +0.1662                | ↑↑↑    |
| private_Breast_luminal| 0.6077           | 0.6107                | +0.0030                | ↑      |
| private_Breast        | 0.8335           | 0.8035                | -0.0300                | ↓↓     |

| Dataset               | Epoch 179 (Acc) | trail_debug_10 (Acc) | Δ (Current - Epoch179) | Trend  |
|-----------------------|-----------------|-----------------------|------------------------|--------|
| BUS-BRA               | 0.7394          | 0.6160                | -0.1234                | ↓↓     |
| BUSI                  | 0.7727          | 0.7674                | -0.0053                | ↓      |
| private_Breast_luminal| 0.7083          | 0.7021                | -0.0062                | ↓      |
| private_Breast        | 0.6667          | 0.7333                | +0.0666                | ↑↑     |
| Appendix              | 0.7872          | 0.6679                | -0.1193                | ↓↓     |
| Fatty-Liver           | 0.8000          | 0.6818                | -0.1182                | ↓↓     |
| private_Liver         | 0.4545          | 0.7000                | +0.2455                | ↑↑↑    |
| private_Appendix      | 0.7500          | 0.6923                | -0.0577                | ↓↓     |

- trial_debug_11: 用一个优化器优化，lr=0.00001，同时在训练，验证过程都加入图像标准化, 调高DDTI数据集和Thyroid数据集的比例

- ⭐[有效实验]trial_debug_12: 用PG模型，用一个优化器优化，lr=0.00001，同时在训练，验证过程都加入图像标准化, 调高DDTI数据集和Thyroid数据集的比例

on PG-395-0.7075

分割任务性能 (Segmentation)

| Dataset               | Metric (Dice) | Timestamp           |
|-----------------------|---------------|---------------------|
| BUS-BRA               | 0.4771        | 2025-07-28 02:42:56 |
| BUSIS                 | 0.6174        | 2025-07-28 02:42:59 |
| BUSI                  | 0.4133        | 2025-07-28 02:43:02 |
| CAMUS                 | 0.9069        | 2025-07-28 02:43:16 |
| DDTI                  | 0.6095        | 2025-07-28 02:44:09 |
| Fetal_HC              | 0.8055        | 2025-07-28 02:44:14 |
| KidneyUS              | 0.6459        | 2025-07-28 02:44:17 |
| private_Thyroid       | 0.4137        | 2025-07-28 02:44:20 |
| private_Kidney        | 0.7466        | 2025-07-28 02:44:20 |
| private_Fetal_Head    | 0.7301        | 2025-07-28 02:44:21 |
| private_Cardiac       | 0.6910        | 2025-07-28 02:44:21 |
| private_Breast_luminal| 0.5588        | 2025-07-28 02:44:22 |
| private_Breast        | 0.7434        | 2025-07-28 02:44:23 |

分类任务性能 (Classification)

| Dataset               | Metric (Accuracy) | Timestamp           |
|-----------------------|-------------------|---------------------|
| Appendix              | 0.7679           | 2025-07-28 02:44:30 |
| BUS-BRA               | 0.7360           | 2025-07-28 02:44:37 |
| BUSI                  | 0.7054           | 2025-07-28 02:44:40 |
| Fatty-Liver           | 0.7000           | 2025-07-28 02:44:43 |
| private_Liver         | 0.7000           | 2025-07-28 02:44:43 |
| private_Breast_luminal| 0.6383           | 2025-07-28 02:44:44 |
| private_Breast        | 0.6000           | 2025-07-28 02:44:45 |
| private_Appendix      | 0.7692           | 2025-07-28 02:44:46 |

实验性能对比分析 (UniUSNet-BIBM vs PerceptGuide-MedIA)

分割任务性能对比 (Segmentation)

| Dataset               | UniUSNet (Dice) | PG (Dice) | Δ (PG - UniUSNet) | Trend |
|-----------------------|-----------------|-----------|-------------------|-------|
| BUS-BRA               | 0.5182          | 0.4771    | -0.0411           | ↓     |
| BUSIS                 | 0.7302          | 0.6174    | -0.1128           | ↓↓    |
| BUSI                  | 0.4053          | 0.4133    | +0.0080           | ↑     |
| CAMUS                 | 0.9132          | 0.9069    | -0.0063           | ↓     |
| DDTI                  | 0.6114          | 0.6095    | -0.0019           | →     |
| Fetal_HC              | 0.8332          | 0.8055    | -0.0277           | ↓     |
| KidneyUS              | 0.6489          | 0.6459    | -0.0030           | →     |
| private_Thyroid       | 0.3021          | 0.4137    | +0.1116           | ↑↑    |
| private_Kidney        | 0.6866          | 0.7466    | +0.0600           | ↑↑    |
| private_Fetal_Head    | 0.7551          | 0.7301    | -0.0250           | ↓     |
| private_Cardiac       | 0.6277          | 0.6910    | +0.0633           | ↑↑    |
| private_Breast_luminal| 0.6077          | 0.5588    | -0.0489           | ↓     |
| private_Breast        | 0.8335          | 0.7434    | -0.0901           | ↓↓    |

分类任务性能对比 (Classification)

| Dataset               | UniUSNet (Acc) | PG (Acc) | Δ (PG - UniUSNet) | Trend |
|-----------------------|----------------|----------|-------------------|-------|
| BUS-BRA               | 0.7394         | 0.7360   | -0.0034           | →     |
| BUSI                  | 0.7727         | 0.7054   | -0.0673           | ↓↓    |
| private_Breast_luminal| 0.7083         | 0.6383   | -0.0700           | ↓↓    |
| private_Breast        | 0.6667         | 0.6000   | -0.0667           | ↓↓    |
| Appendix              | 0.7872         | 0.7679   | -0.0193           | ↓     |
| Fatty-Liver           | 0.8000         | 0.7000   | -0.1000           | ↓↓    |
| private_Liver         | 0.4545         | 0.7000   | +0.2455           | ↑↑↑   |
| private_Appendix      | 0.7500         | 0.7692   | +0.0192           | ↑     |


**当前提交结果395_0.7075**
{ "Accuracy_Appendix": 0.6533333333333333, "AUC_Appendix": 0.7514245014245015, "T2_Appendix": 0.7023789173789174, "Accuracy_Breast": 0.7784090909090909, "AUC_Breast": 0.8610786224821312, "T2_Breast": 0.819743856695611, "Accuracy_Breast_luminal": 0.5625, "AUC_Breast_luminal": 0.4756470130967727, "T2_Breast_luminal": 0.5190735065483864, "Accuracy_Liver": 0.6528925619834711, "AUC_Liver": 0.8473239436619718, "T2_Liver": 0.7501082528227214, "DSC_Breast": 0.6464360328683867, "NSD_Breast": 0.05618100039300724, "T1_Breast": 0.7951275162376897, "DSC_Breast_luminal": 0.4784755261298462, "NSD_Breast_luminal": 0.02389913157674143, "T1_Breast_luminal": 0.7272881972765524, "DSC_Cardiac": 0.6345239392634457, "NSD_Cardiac": 0.02214575606013667, "T1_Cardiac": 0.8061890916016545, "DSC_Fetal_Head": 0.6976647592564753, "NSD_Fetal_Head": 0.02987472421187622, "T1_Fetal_Head": 0.8338950175222996, "DSC_Kidney": 0.6649285437841299, "NSD_Kidney": 0.03510571828722756, "T1_Kidney": 0.8149114127484511, "DSC_Thyroid": 0.3838297283646404, "NSD_Thyroid": 0.02628619574359305, "T1_Thyroid": 0.6787717663105237, "DSC_Breast_all": 0.5469038807270293, "NSD_Breast_all": 0.03705100405744231, "T1_Breast_all": 0.7549264383347936 }

**0730 top1的详细分数**
{ "Accuracy_Appendix": 0.5866666666666667, "AUC_Appendix": 0.717948717948718, "T2_Appendix": 0.6523076923076923, "Accuracy_Breast": 0.5909090909090909, "AUC_Breast": 0.6022092267706303, "T2_Breast": 0.5965591588398607, "Accuracy_Breast_luminal": 0.1171875, "AUC_Breast_luminal": 0.45400482521122815, "T2_Breast_luminal": 0.28559616260561405, "Accuracy_Liver": 0.7024793388429752, "AUC_Liver": 0.732112676056338, "T2_Liver": 0.7172960074496566, "DSC_Breast": 0.8419483796729123, "NSD_Breast": 0.24217687716406386, "T1_Breast": 0.7998857512544242, "DSC_Breast_luminal": 0.8351720041143568, "NSD_Breast_luminal": 0.1435258839520758, "T1_Breast_luminal": 0.8458230600811405, "DSC_Cardiac": 0.8391237172270091, "NSD_Cardiac": 0.06477038445713673, "T1_Cardiac": 0.8871766663849361, "DSC_Fetal_Head": 0.9245959844530885, "NSD_Fetal_Head": 0.11087514887916335, "T1_Fetal_Head": 0.9068604177869626, "DSC_Kidney": 0.868685870395864, "NSD_Kidney": 0.11325077083593489, "T1_Kidney": 0.8777175497799645, "DSC_Thyroid": 0.6774622108728966, "NSD_Thyroid": 0.116285739183835, "T1_Thyroid": 0.7805882358445309, "DSC_Breast_all": 0.8379327497122867, "NSD_Breast_all": 0.18371702933473757, "T1_Breast_all": 0.8271078601887746 }


| 指标名称               | 前结果(ours) | 后结果(top1) | 绝对差（前-后） | 相对变化 | 优劣方向       |
|------------------------|--------|--------|------------------|----------|----------------|
| **分类任务指标**        |        |        |                  |          |                |
| Accuracy_Appendix      | 0.653  | 0.587  | +0.066           | +10.1%   | 前优           |
| AUC_Appendix           | 0.751  | 0.718  | +0.033           | +4.4%    | 前优           |
| **T2_Appendix**        | 0.702  | 0.652  | +0.050           | +7.1%    | 前优           |
| Accuracy_Breast        | 0.778  | 0.591  | +0.187           | +24.0%   | **前显著优**   |
| AUC_Breast             | 0.861  | 0.602  | +0.259           | +30.1%   | **前显著优**   |
| **T2_Breast**          | 0.820  | 0.597  | +0.223           | +27.2%   | **前显著优**   |
| Accuracy_Breast_luminal| 0.562  | 0.117  | +0.445           | +79.2%   | **前显著优**   |
| AUC_Breast_luminal     | 0.476  | 0.454  | +0.022           | +4.6%    | 前略优         |
| **T2_Breast_luminal**  | 0.519  | 0.286  | +0.233           | +44.9%   | **前显著优**   |
| Accuracy_Liver         | 0.653  | 0.702  | -0.049           | -7.5%    | 后优           |
| AUC_Liver              | 0.847  | 0.732  | +0.115           | +13.6%   | 前优           |
| **T2_Liver**           | 0.750  | 0.717  | +0.033           | +4.4%    | 前优           |
| **分割任务指标**        |        |        |                  |          |                |
| DSC_Breast             | 0.646  | 0.842  | -0.196           | -30.3%   | **后显著优**   |
| NSD_Breast             | 0.056  | 0.242  | -0.186           | -332.1%  | 前优（↓NSD）   |
| **T1_Breast**          | 0.795  | 0.800  | -0.005           | -0.6%    | 基本持平       |
| DSC_Breast_luminal     | 0.478  | 0.835  | -0.357           | -74.7%   | **后显著优**   |
| NSD_Breast_luminal    | 0.024  | 0.144  | -0.120           | -500.0%  | 前优（↓NSD）   |
| **T1_Breast_luminal** | 0.727  | 0.846  | -0.119           | -16.4%   | 后优           |
| DSC_Cardiac            | 0.635  | 0.839  | -0.204           | -32.1%   | **后显著优**   |
| NSD_Cardiac           | 0.022  | 0.065  | -0.043           | -195.5%  | 前优（↓NSD）   |
| **T1_Cardiac**        | 0.806  | 0.887  | -0.081           | -10.0%   | 后优           |
| DSC_Fetal_Head         | 0.698  | 0.925  | -0.227           | -32.5%   | **后显著优**   |
| NSD_Fetal_Head        | 0.030  | 0.111  | -0.081           | -270.0%  | 前优（↓NSD）   |
| **T1_Fetal_Head**     | 0.834  | 0.907  | -0.073           | -8.8%    | 后优           |
| DSC_Kidney             | 0.665  | 0.869  | -0.204           | -30.7%   | **后显著优**   |
| NSD_Kidney            | 0.035  | 0.113  | -0.078           | -222.9%  | 前优（↓NSD）   |
| **T1_Kidney**         | 0.815  | 0.878  | -0.063           | -7.7%    | 后优           |
| DSC_Thyroid            | 0.384  | 0.677  | -0.293           | -76.3%   | **后显著优**   |
| NSD_Thyroid           | 0.026  | 0.116  | -0.090           | -346.2%  | 前优（↓NSD）   |
| **T1_Thyroid**        | 0.679  | 0.781  | -0.102           | -15.0%   | 后优           |
| DSC_Breast_all         | 0.547  | 0.838  | -0.291           | -53.2%   | **后显著优**   |
| NSD_Breast_all        | 0.037  | 0.184  | -0.147           | -397.3%  | 前优（↓NSD）   |
| **T1_Breast_all**     | 0.755  | 0.827  | -0.072           | -9.5%    | 后优           |

- trail_debug_13:再12的基础上，单独训练分割200个epoch

PG_589_0.7173

| 数据集名称            | seg(DSC)     | 备注                     |
|-----------------------|---------|--------------------------|
| BUS-BRA               | 0.618   | 乳腺超声分割             |
| BUSIS                 | 0.791   | 乳腺肿瘤分割             |
| BUSI                  | 0.498   | 乳腺超声图像分割         |
| CAMUS                 | 0.930   | **心脏超声分割（最佳）** |
| DDTI                  | 0.713   | 甲状腺结节分割           |
| Fetal_HC              | 0.863   | 胎儿头围测量             |
| KidneyUS              | 0.757   | 肾脏超声分割             |
| private_Thyroid       | 0.447   | 私有甲状腺数据集         |
| private_Kidney        | 0.794   | 私有肾脏分割             |
| private_Fetal_Head    | 0.810   | 私有胎儿头围分割         |
| private_Cardiac       | 0.779   | 私有心脏分割             |
| private_Breast_luminal        | 0.688  | 私有乳腺腔面型分割             |
| private_Breast        | 0.816   | 私有乳腺分割             |

| 数据集名称             | cls(Acc)  | 备注                     |
|------------------------|---------|--------------------------|
| Appendix               | 0.748   | 阑尾炎分类               |
| BUS-BRA                | 0.702   | 乳腺超声分类             |
| BUSI                   | 0.774   | 乳腺图像分类             |
| Fatty-Liver            | 0.806   | **脂肪肝分类（最佳）**   |
| private_Liver          | 0.677   | 私有肝脏分类             |
| private_Breast_luminal | 0.690   | 乳腺腔面型分类           |
| private_Breast         | 0.622   | 私有乳腺分类             |
| private_Appendix       | 0.714   | 私有阑尾炎分类           |

实验结果对比表（PG-589_0.7173 vs PG-395-0.7075）

| 数据集名称            | PG-589 (实验组) | PG-395 (对照组) | 差值 (实验-对照) | 相对变化  | 显著性判定       |
|-----------------------|-----------------|-----------------|------------------|-----------|------------------|
| BUS-BRA               | 0.618           | 0.477           | +0.141           | +29.6%    | **显著提升↑↑**   |
| BUSIS                 | 0.791           | 0.617           | +0.174           | +28.2%    | **显著提升↑↑**   |
| BUSI                  | 0.498           | 0.413           | +0.085           | +20.6%    | **显著提升↑↑**   |
| CAMUS                 | 0.930           | 0.907           | +0.023           | +2.5%     | 轻微提升↑        |
| DDTI                  | 0.713           | 0.610           | +0.103           | +16.9%    | **显著提升↑↑**   |
| Fetal_HC              | 0.863           | 0.806           | +0.057           | +7.1%     | 明显提升↑        |
| KidneyUS              | 0.757           | 0.646           | +0.111           | +17.2%    | **显著提升↑↑**   |
| private_Thyroid       | 0.447           | 0.414           | +0.033           | +8.0%     | 明显提升↑        |
| private_Kidney        | 0.794           | 0.747           | +0.047           | +6.3%     | 明显提升↑        |
| private_Fetal_Head    | 0.810           | 0.730           | +0.080           | +11.0%    | **显著提升↑↑**   |
| private_Cardiac       | 0.779           | 0.691           | +0.088           | +12.7%    | **显著提升↑↑**   |
| private_Breast_luminal| 0.688           | 0.559           | +0.129           | +23.1%    | **显著提升↑↑**   |
| private_Breast        | 0.816           | 0.743           | +0.073           | +9.8%     | 明显提升↑        |

| 数据集名称             | PG-589 (实验组) | PG-395 (对照组) | 差值 (实验-对照) | 相对变化  | 显著性判定       |
|------------------------|-----------------|-----------------|------------------|-----------|------------------|
| Appendix               | 0.748           | 0.768           | -0.020           | -2.6%     | 轻微下降↓        |
| BUS-BRA                | 0.702           | 0.736           | -0.034           | -4.6%     | 明显下降↓↓       |
| BUSI                   | 0.774           | 0.705           | +0.069           | +9.8%     | **显著提升↑↑**   |
| Fatty-Liver            | 0.806           | 0.700           | +0.106           | +15.1%    | **显著提升↑↑**   |
| private_Liver          | 0.677           | 0.700           | -0.023           | -3.3%     | 轻微下降↓        |
| private_Breast_luminal | 0.690           | 0.638           | +0.052           | +8.1%     | 明显提升↑        |
| private_Breast         | 0.622           | 0.600           | +0.022           | +3.7%     | 轻微提升↑        |
| private_Appendix       | 0.714           | 0.769           | -0.055           | -7.2%     | 明显下降↓↓       |​

trail_debug_14:再13的基础上，单独训练分割100个epoch

**best_model_699_0.745**

| 数据集                | DSC       |
|-----------------------|-----------|
| BUS-BRA               | 0.7273    |
| BUSIS                 | 0.8165    |
| BUSI                  | 0.5590    |
| CAMUS                 | 0.9373    |
| DDTI                  | 0.7562    |
| Fetal_HC              | 0.9015    |
| KidneyUS              | 0.8156    |
| private_Thyroid       | 0.5252    |
| private_Kidney        | 0.8427    |
| private_Fetal_Head    | 0.8686    |
| private_Cardiac       | 0.8105    |
| private_Breast_luminal| 0.7191    |
| private_Breast        | 0.8544    |


| 数据集                | Acc       |
|-----------------------|-----------|
| Appendix              | 0.7316    |
| BUS-BRA               | 0.6963    |
| BUSI                  | 0.7897    |
| Fatty-Liver           | 0.7939    |
| private_Liver         | 0.6774    |
| private_Breast_luminal| 0.6620    |
| private_Breast        | 0.6444    |
| private_Appendix      | 0.6667    |

实验结果对比分析 (best_model_699_0.745 vs PG_589_0.7173)

| 数据集                | 实验组(DSC/Acc) | 对照组(DSC/Acc) | 差值(Δ)   | 变化趋势 | 显著性分析 |
|-----------------------|----------------|----------------|-----------|---------|------------|
| **分割任务(DSC)**     |                |                |           |         |            |
| BUS-BRA               | 0.7273         | 0.618          | +0.1093   | ↑↑      | 显著提升   |
| BUSIS                 | 0.8165         | 0.791          | +0.0255   | ↑       | 小幅提升   |
| BUSI                  | 0.5590         | 0.498          | +0.0610   | ↑       | 中等提升   |
| CAMUS                 | 0.9373         | 0.930          | +0.0073   | →       | 基本持平   |
| DDTI                  | 0.7562         | 0.713          | +0.0432   | ↑       | 小幅提升   |
| Fetal_HC              | 0.9015         | 0.863          | +0.0385   | ↑       | 小幅提升   |
| KidneyUS              | 0.8156         | 0.757          | +0.0586   | ↑       | 中等提升   |
| private_Thyroid       | 0.5252         | 0.447          | +0.0782   | ↑↑      | 显著提升   |
| private_Kidney        | 0.8427         | 0.794          | +0.0487   | ↑       | 小幅提升   |
| private_Fetal_Head    | 0.8686         | 0.810          | +0.0586   | ↑       | 中等提升   |
| private_Cardiac       | 0.8105         | 0.779          | +0.0315   | ↑       | 小幅提升   |
| private_Breast_luminal| 0.7191         | 0.688          | +0.0311   | ↑       | 小幅提升   |
| private_Breast        | 0.8544         | 0.816          | +0.0384   | ↑       | 小幅提升   |
| **分类任务(Acc)**     |                |                |           |         |            |
| Appendix              | 0.7316         | 0.748          | -0.0164   | ↓       | 小幅下降   |
| BUS-BRA               | 0.6963         | 0.702          | -0.0057   | →       | 基本持平   |
| BUSI                  | 0.7897         | 0.774          | +0.0157   | ↑       | 小幅提升   |
| Fatty-Liver           | 0.7939         | 0.806          | -0.0121   | ↓       | 小幅下降   |
| private_Liver         | 0.6774         | 0.677          | +0.0004   | →       | 完全持平   |
| private_Breast_luminal| 0.6620         | 0.690          | -0.0280   | ↓       | 中等下降   |
| private_Breast        | 0.6444         | 0.622          | +0.0224   | ↑       | 小幅提升   |
| private_Appendix      | 0.6667         | 0.714          | -0.0473   | ↓↓      | 显著下降   |


- trail_debug_16：各个数据集使用一个单独的decoder，做特异性优化 **不 work**

- trail_debug_17：测试用的，各个数据集使用一个单独的decoder，做特异性优化 

- trail_debug_18：调整学习率lr=0.001，各个数据集使用一个单独的decoder，做特异性优化

- trail_debug_19：调整学习率lr=0.00001，各个数据集使用一个单独的decoder，做特异性优化。导入之前的encoder的权重并冻结encoder，只训练decoder和task Heads。

- trail_debug_20：**不work**
    - lr：Warmup + Cosine Annealing
    - 各个数据集使用一个单独的decoder，做特异性优化。
    - 导入他给的预训练权重，将encoder的权重冻结，只训练decoder和task Heads。- 对分割的两个损失函数的权重进行动态调整
    - 添加早停

- trail_debug_21：在20的基础上
    - 使用[官方](https://github.com/Zehui-Lin/PerceptGuide/releases/tag/v1.0.0)的权重进行初始化，并冻结encoder
    - 不使用早停
    - 扩充数据，同种任务和器官的数据用同一个decoder

- trail_debug_22：在20的基础上
    - 使用[官方](https://github.com/Zehui-Lin/PerceptGuide/releases/tag/v1.0.0)的权重进行初始化，并冻结encoder
    - 不使用早停
    - 扩充数据，同种任务和器官的数据用同一个decoder
    - lr 1e-5