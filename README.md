# 计算NPV/PPV/敏感性/特异性

## 计算方法
详见CSDN POST：
[https://blog.csdn.net/iling5/article/details/130526176?spm=1001.2014.3001.5501](https://blog.csdn.net/iling5/article/details/130526176?spm=1001.2014.3001.5501)



## Demo

```python
python demo.py

```

**输出**
```bash
  Thr    ACC    PPV    NPV Sens(Rec/TPR)   Spec YoudenIdx     F1 TrueBen TrueMal PredBen PredMal      TP      FP      TN      FN
0.000  0.540  0.540  0.000         1.000  0.000     0.000  0.701  27.000  23.000  50.000   0.000  27.000  23.000   0.000   0.000
0.050  0.540  0.543  0.500         0.926  0.087     0.013  0.685  27.000  23.000  46.000   4.000  25.000  21.000   2.000   2.000
0.100  0.580  0.571  0.625         0.889  0.217     0.106  0.696  27.000  23.000  42.000   8.000  24.000  18.000   5.000   3.000
```
