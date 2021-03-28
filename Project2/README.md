# Monte-Carlo(MC) Learning and Temporal-Difference(TD) Learning

-----

## 实验目的：

- 蒙特卡罗学习 *Monte-Carlo Learning*:
  - First-visit
  - Every-visit
- 时序差分学习 *Temporal-Difference Learning*

-------

## 运行本项目的方法：

- 添加必要依赖包：
  - numpy
  - secrets
- 添加main.py的执行路径，直接运行即可

------

## 实验原理：

### 1 蒙特卡罗学习 *Moten-Carlo Learning*

- 可以直接从随机 *episodes* 中学习
- *Model-free*，不需要知道 *MDP* 的传递 / 奖励情况，*MDP* 中选取的 *episode* 必须有终点
- 基本思想：**某状态的值函数等于平均采样返回值**

<img src="./cut/截屏2021-03-28 下午12.19.25.png" alt="avatar" style="zoom:50%;" />

------------



### 1.1 First-visit

- 算法流程：

<img src="./cut/截屏2021-03-28 下午12.20.26.png" alt="avatar" style="zoom:50%;" />

- 伪代码：

<img src="./cut/截屏2021-03-28 下午12.21.29.png" alt="avatar" style="zoom:50%;" />

-----



### 1.2 Every-visit

- 算法流程：

<img src="/Users/dicardo/Downloads/SJTU_CS489_Reinforcement_Learning/Project2/cut/截屏2021-03-28 下午12.22.51.png" alt="avatar" style="zoom:50%;" />

------



### 2 时序差分学习 *Temporal-Difference Learning* 

- 时序差分算法是一种无模型的强化学习算法。它继承了动态规划(Dynamic Programming)和蒙特卡罗方法(Monte Carlo Methods)的优点，从而对状态值(state value)和策略(optimal policy)进行预测。从本质上来说，时序差分算法和动态规划一样，是一种bootstrapping的算法。同时，也和蒙特卡罗方法一样，是一种无模型的强化学习算法，其原理也是基于了试验。虽然，时序差分算法拥有动态规划和蒙特卡罗方法的一部分特点，但它们也有不同之处。以下是它们各自的backup图：

<img src="./cut/截屏2021-03-28 下午3.12.50.png" alt="avatar " style="zoom:50%;" />

​	根据它们的backup图可以知道，动态规划的backup操作是基于当前状态和下一个状态的reward，蒙特卡罗方法的backup是基于一个完整的episode的reward，而时序差分算法的backup是基于当前状态和下一个状态的reward。其中，最基础的时序差分算法被称为TD(0)。它也有许多拓展，如n-step TD算法和TD(lambda)算法。

- ***TD0* 算法**
  - 也需要随机生成 *episode*

<img src="./cut/截屏2021-03-28 下午3.15.09.png" alt="avatar" style="zoom:50%;" />

<img src="./cut/截屏2021-03-28 下午3.15.34.png" alt="avatar" style="zoom:50%;" />

--------

## 实验代码分析：

- 由于本次实验对报告不作要求，这里不详细进行项目代码的分析，可以参见`./code`目录下各`.py`文件中完备的代码注释。

-------

## 实验结果：

- *First-visit in Monte-Carlo Learning*

<img src="./cut/截屏2021-03-28 下午5.34.15.png" alt="avatar" style="zoom:50%;" />

- *Every-visit in Monte-Carlo Learning*

<img src="./cut/截屏2021-03-28 下午5.34.58.png" alt="avatar" style="zoom:50%;" />

- *Temporal Difference Learning*

<img src="./cut/截屏2021-03-28 下午5.35.13.png" alt="avatar" style="zoom:50%;" />

----

## 结果分析 & 实验结论

- 经验证，上述三种算法得到的策略均具有最优性，实验成功。
- 相较于 *Every-visit* 方案，*First-visit* 的收敛速度更快，但稳定性较差，因此需要设置一个最少迭代次数来保证算法的稳定性，对另外两种算法也设置了这样的 *baseline*
- 对于时序差分算法，由于并没有像前两种算法那样基于大数定律，其最终结果是无法达到在阈值 $\theta$ 很小情况下的收敛的，因此将其 $\theta$ 设置为一个较大的，较易满足的值，例如0.2，而主要控制其最少迭代次数
- 为了进一步保证稳定性，我们同时设置前两种方案需要累计达到两次阈值，才能退出
- 对于时序差分算法，我们对其中的 $\alpha$ （最终取值为0.01，想法是在迭代次数较大时尽量减少V的波动）、$\theta$ 和最少迭代次数进行了一定程度的调参，以保证算法的稳定性

