# trail_debug_36

| 实验ID  | 平均分割分数 (Dice) | 平均分类分数 (Accuracy) | Thyroid (seg) | Kidney (seg) | Fetal_Head (seg) | Cardiac (seg) | Breast_luminal (seg) | Breast (seg) | Liver (cls) | Breast_luminal (cls) | Breast (cls) | Appendix (cls) | 测试时间               |
|---------|---------------------|-------------------------|---------------|--------------|-------------------|---------------|-----------------------|--------------|-------------|-----------------------|--------------|----------------|-------------------------|
| 0.8171  | 0.8058               | 0.6008                  | 0.5767        | 0.8589       | 0.9138            | 0.7929        | 0.7830                | 0.9093       | 0.6364      | 0.6667                | 0.6000       | 0.5000         | 2025-08-12 15:42:15~24  |
| 0.8202  | 0.8133               | 0.6360                  | 0.5296        | 0.8730       | 0.9232            | 0.8718        | 0.7713                | 0.9111       | 0.7273      | 0.5833                | 0.7333       | 0.5000         | 2025-08-13 01:05:23~29  |
| 0.8276  | 0.8227               | 0.7443                  | 0.5619        | 0.8929       | 0.9112            | 0.8753        | 0.7886                | 0.9064       | 0.7273      | 0.7083                | 0.6667       | 0.8750         | 2025-08-13 01:04:51~58  |
| 0.8348  | 0.8274               | 0.7172                 | 0.5550        | 0.8929       | 0.9262            | 0.8805        | 0.7780                | 0.9320       | 0.7273      | 0.6667                | 0.6000       | 0.8750         | 2025-08-13 00:48:40~46  |




## 原始tta得分:
    Private segmentation performance:
    private_Thyroid: 0.6747
    private_Kidney: 0.9204
    private_Fetal_Head: 0.9220
    private_Cardiac: 0.8646
    private_Breast_luminal: 0.8617
    private_Breast: 0.8693
    Mean Dice: 0.8521
    Private classification performance:
    private_Liver: 0.8182
    private_Breast_luminal: 0.7083
    private_Breast: 0.7333
    private_Appendix: 1.0000
    Mean Acc: 0.8150

## trail_debug_45-1:对logit做平均,之后在转成概率图:分割提升0.0007
    Private segmentation performance:
    private_Thyroid: 0.6797
    private_Kidney: 0.9204
    private_Fetal_Head: 0.9220
    private_Cardiac: 0.8646
    private_Breast_luminal: 0.8603
    private_Breast: 0.8696
    Mean Dice: 0.8528
    Private classification performance:
    private_Liver: 0.8182
    private_Breast_luminal: 0.7083
    private_Breast: 0.7333
    private_Appendix: 1.0000
    Mean Acc: 0.8150

## trail_debug_45-1-1:取消tumor的限制: 不是tumor类型的用了tumor的增强之后效果**变差了**
    Private segmentation performance:
    private_Thyroid: 0.6797
    private_Kidney: 0.8962
    private_Fetal_Head: 0.9157
    private_Cardiac: 0.7697
    private_Breast_luminal: 0.8603
    private_Breast: 0.8696
    Mean Dice: 0.8319
    Private classification performance:
    private_Liver: 0.8182
    private_Breast_luminal: 0.7083
    private_Breast: 0.7333
    private_Appendix: 1.0000
    Mean Acc: 0.8150


## trail_debug_45-1-2:加了8种方向变换,分割性能**变差**
    Private segmentation performance:
    private_Thyroid: 0.5226
    private_Kidney: 0.9204
    private_Fetal_Head: 0.9220
    private_Cardiac: 0.8646
    private_Breast_luminal: 0.7340
    private_Breast: 0.8561
    Mean Dice: 0.8033
    Private classification performance:
    private_Liver: 0.8182
    private_Breast_luminal: 0.7083
    private_Breast: 0.6667
    private_Appendix: 1.0000
    Mean Acc: 0.7983

## trail_debug_45-1-3: 调大对比度,几乎没变化

## trail_debug_45-1-4: 
