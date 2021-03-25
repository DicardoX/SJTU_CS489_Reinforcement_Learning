#  Project 1: Dynamic Programing

[强化学习的最简单实现----矩阵找路问题](https://blog.csdn.net/qq_42511414/article/details/109962364)

-----------

- Policy Evaluation & Policy Iteration
- Value Iteration

------

## 1 强化学习的主要组成部分

- **策略 *(policy)* **：
  - *Agent* 的行为函数
  - 从状态到行为的映射
  - 类别：
    - 确定性：$a = \pi(s)$
    - 随机性：$\pi(a|S) = P[A=a | S=s]$
- **值函数 (*Value function*)**：
  - 对未来奖励的预测
  - 用来评估状态的好坏
  - $v_{\pi}(s) = E_{\pi}[R_t + \gamma R_{t+1} + \gamma^2R_{t+2} + ... | S_t = s]$
- **模型 (*Model*)**：
  - 预测环境接下来会怎么做
  - *Transitions Model*：预测下一个状态，动态特性，$P_{SS'}^a = P[S' = s' | S = s, A = a]$
  - *Rewards Model*：预测下一个奖励，$R_S^a = E[R | S=s, A=a]$

---



## 2 马尔可夫奖励过程 *Markov Reward Processes*

### 2.1 定义：

<img src="./cut/截屏2021-03-25 上午11.38.44.png" alt="avatar" style="zoom:50%;" />

- 注意，*R* 可以被理解为离开状态 $S_t$ 后，得到的奖励

----



### 2.2 *MRP* 的贝尔曼方程

 										<img src="./cut/截屏2021-03-25 上午11.42.12.png" alt="avatar" style="zoom:40%;" />

<img src="./cut/截屏2021-03-25 上午11.43.23.png" alt="avatar" style="zoom:50%;" />

-----



## 3 马尔可夫决策过程 *Markov Decision Processes*

### 3.1 定义

​	注意：红色部分为相较于 *MRP* 过程添加的部分。

<img src="./cut/截屏2021-03-25 下午12.43.04.png" alt="avatar" style="zoom:50%;" />

- 策略 *policy* 的定义：

<img src="./cut/截屏2021-03-25 下午12.47.19.png" alt="avatar " style="zoom:50%;" />

-------



### 3.2 状态函数 / 行为函数

- 状态函数的定义：
  - 从状态 *s* 开始，按照策略 $\pi$ 的期望返回值

<img src="./cut/截屏2021-03-25 下午12.52.42.png" alt="avatar" style="zoom:50%;" />

- 行为函数的定义：
  - 从状态 *s* 开始，采取行为 *a* ，然后按照策略 $\pi$ 的期望返回值

<img src="./cut/截屏2021-03-25 下午12.54.44.png" alt="avatar" style="zoom:50%;" />

-----



### 3.3 贝尔曼期望方程

<img src="./cut/截屏2021-03-25 下午12.56.58.png" alt="avatar" style="zoom:50%;" />

- 两种值函数之间的联系：

<img src="./cut/截屏2021-03-25 下午12.57.47.png" alt="avatar" style="zoom:50%;" />

<img src="./cut/截屏2021-03-25 下午12.57.59.png" alt="avatar" style="zoom:50%;" />

- 基于上述关系求出的两种值函数的迭代关系：

<img src="./cut/截屏2021-03-25 下午12.59.32.png" alt="avatar" style="zoom:50%;" />

<img src="./cut/截屏2021-03-25 下午12.59.43.png" alt="avatar" style="zoom:50%;" />

------



### 3.4 最优化值函数*Optimal Value Function*

- 定义：

<img src="./cut/截屏2021-03-25 下午1.07.37.png" alt="avatar" style="zoom:50%;" />

- 最优化值函数的迭代关系：

<img src="./cut/截屏2021-03-25 下午1.10.58.png" alt="avatar" style="zoom:50%;" />

<img src="./cut/截屏2021-03-25 下午1.11.16.png" alt="avatar" style="zoom:50%;" />

- 最优化策略的寻找：
  - 通过最大化 $q_*(s, a)$ 来获取

<img src="./cut/截屏2021-03-25 下午1.13.15.png" alt="avatar" style="zoom:50%;" />

----------



## Topic

<img src="./cut/截屏2021-03-22 上午11.52.55.png" alt="avatar" style="zoom:80%;" />


