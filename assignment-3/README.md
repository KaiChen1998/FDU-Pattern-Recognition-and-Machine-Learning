# Assignment 3: Language Generative Model Based on LSTM

- Gradient calculation and backpropagation using numpy
- Language model trained on Tang poems dataset
- Generative Tang poems model

Here we refer to the excellent implementation by [@changchaokun](<https://github.com/changchaokun>). [[Link]](<https://github.com/ichn-hu/PRML-Spring19-Fudan/tree/master/assignment-3/16307130138>)



## Appendix: BiLSTM + CPF for NER

- 马尔可夫随机场

  - X(t)是随时间变化的随机变量，若其后验概率只和前一时步的状态相关则该随机变量满足马尔可夫性
  - 马尔科夫性另一种解释：若给定X(t - 1)则X(t)与{X(i), i = 1... t - 2}均独立
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

- BiLSTM + CRF for NER：显示控制一致性

  ![img](https://pic3.zhimg.com/v2-d2f81f90e43dc8e12b802b302a150bba_r.jpg)

  - embedding：
    - word embedding：预训练
    - character embedding：随机初始化
    - 两者同时使用并在训练过程当中更新

  - BIO encoding for NER：
    - B-class：某种待测name entity的开始
    - I-class：某种待测name entity的中间，并不意味着结束
    - O：非待测name entity

  - BiLSTM：照常输出name entity class set上的后验分布
  - CRF：
    - 输入时BiLSTM在每一时步得到的概率分布
    - 问题：若直接使用LSTM的结果，则可能带来强不一致性（e.g. 有I无B）
    - Idea：
      - emission score：BiLSTM得到的概率分布
      - transition score：实际上就是CRF的状态转移矩阵
      - 综合考虑两者来保证一致性
  - Training：
    - 单条路径的score：根据emission score和transition score，将name entity type和transition都视作path的一部分，加和他们的score值
    - Loss：CE，但是概率分布定义在所有路线的集合上，即定义了一颗|C|叉树。
    - 问题：计算所有可能路径的score值
    - 解法：动态分布
      - 维护两个|C|维向量：obs和previous。
      - obs代表当前的|C|个emission score
      - previous中任意$C_i$代表在t-1时步，所有到达class i的路径score之和（对于非GT的路径我们只关心其score和，因此空间复杂度和输入大小解耦）
      - previous和obs两两组合，并加上transition matrix

  - Inference：
    - 目标：找到树中score值最大的路径，即类最短路问题
    - DP：两个list
      - 一个list记录每一时步到|C|个class的最大路径的当前score值（其实不用存list只要存上一时步的就好）
      - 另一个list存索引，以在DP结束的时候回溯

- 参考：[[link]](<https://mp.weixin.qq.com/s?__biz=MzI4MDYzNzg4Mw==&mid=2247492786&idx=3&sn=c5867d33477097df7ae78a0168bb33f2&chksm=ebb7dc66dcc05570771bfa22c1bdcf52774ed132ca56218bc74604a3f5d4c268fece603e8261&mpshare=1&scene=1&srcid=032884sdUVHAc0uCRSZJOUoH&sharer_sharetime=1585326282836&sharer_shareid=c47b85daf6bf0d13114eb8b891f7f7ce&key=c31cce5eab85a94ad2f52ad57241f71744fe5365d3ac8c23e303ff0ae6327229ad0c68f52714215f534a09f06b0f48cadbbc8363c6efc1fbcbc08f96633a7ac6d67500b644d977f82d6818b5e4f9c82e&ascene=1&uin=MTkxMTI3NDE2MA%3D%3D&devicetype=Windows+10&version=62080079&lang=zh_CN&exportkey=AQonyWhKIoye%2F7aFGaDp7qA%3D&pass_ticket=UDVLcuq492T%2BlGVwXtV5Gv2aajOS6Ab2pFAlsh1KJn20dExoLLVTbAo4k4LhCipM>) & [[zhihu1]](<https://zhuanlan.zhihu.com/p/69890528>) & [[zhihu2]](<https://zhuanlan.zhihu.com/p/70777941>)