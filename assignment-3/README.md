# Assignment 3: Language Generative Model Based on LSTM

- Gradient calculation and backpropagation using numpy
- Language model trained on Tang poems dataset
- Generative Tang poems model

Here we refer to the excellent implementation by [@changchaokun](<https://github.com/changchaokun>). [[Link]](<https://github.com/ichn-hu/PRML-Spring19-Fudan/tree/master/assignment-3/16307130138>)



## Appendix: LSTM + CPF for NER

- 马尔可夫随机场

  - X(t)是随时间变化的随机变量，若其后验概率只和前一时步的状态相关则该随机变量满足马尔可夫性
  - 隐马尔可夫模型：在马尔可夫性质的假设下做序列标注，只需要根据语料库计算状态转移矩阵

  ![img](https://pic1.zhimg.com/v2-8a122040c6adbd35d846d7c4a424e278_r.jpg)

- 条件随机场CRF

  - 推断：根据已知变量的分布去推断未知变量的分布，其中：

    - 生成式：联合分布P(x, y)
    - 判别式：后验分布P(y | x)

  - 条件随机场

    - 当前状态至于其有连边的状态相关，即为条件随机场

    - 常用的是线性条件随机场，和HMM类似，但是隐变量从有向图变成了无向图，即当前状态与其前后时刻的状态均相关

    ![img](https://pic1.zhimg.com/80/v2-117f06c1212ad6992165119450152940_hd.jpg)

- LSTM + CRF：显示控制一致性

![img](https://pic3.zhimg.com/v2-d2f81f90e43dc8e12b802b302a150bba_r.jpg)