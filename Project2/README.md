# Monte-Carlo(MC) Learning and Temporal-Difference(TD) Learning

-----

- Monte-Carlo Learning:
  - First-visit
  - Every-visit
- Temporal-Difference Learning

-------



## 1 Moten-Carlo Learning

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

----



