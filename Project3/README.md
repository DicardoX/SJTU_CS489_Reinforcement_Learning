# Sarsa in On-policy and Q-Learning in Off-policy



- *Sarsa* 和 *Q-Learning* 的对比：

  - *Sarsa* 算法随机初始化一个 *state*，在当前 *state* 下利用 $\epsilon$ *- greedy* 选择一个当前即将采用的 *action*，然后开始迭代：
    - **说到做到**地采取该 *action*，观察下一个 *state'*，再利用 **$\epsilon$ *- greedy* **寻找下一个 *action'*，且**说到做到**地在下一次迭代中采取该 *action'*
    - 更新*Q(state, action)*：基于当前的 *Q(state, action)* 和即将要采用的 *Q(state', action')* 
    - 移动：*state = state'*，*action = action'*
    - 重复直到到达 *terminal state*
  - *Q-Learning* 算法随机初始化一个 *state*，然后开始迭代：
    - 在当前 *state* 下利用 $\epsilon$ *- greedy* 选择一个当前即将采用的 *action*
    - **说到做到**地采取 *action*，观察下一个 *state'*，再利用**普通的基于 *Q* 的 *greedy* **寻找下一个 *action'*，仅**假设**采取该 *action'*，实际采取哪个 *action'* 需要在每次迭代的第一步做决定
    - 更新*Q(state, action)*：基于当前的 *Q(state, action)* 和即将要采用的 *Q(state', action')* 
    - 移动：*state = state'*
    - 重复直到到达 *terminal state*

  <img src="./cut/截屏2021-04-09 下午7.17.11.png" alt="avatar" style="zoom:50%;" />

  - Sarsa 是说到做到型, 所以我们也叫他 on-policy, 在线学习, 学着自己在做的事情. 而 Q learning 是说到但并不一定做到, 所以它也叫作 Off-policy, 离线学习. 而因为有了 maxQ, Q-learning 也是一个特别勇敢的算法
  - 为什么说他勇敢呢, 因为 Q learning 机器人 永远都会选择最近的一条通往成功的道路, 不管这条路会有多危险. 而 Sarsa 则是相当保守, 他会选择离危险远远的, 拿到宝藏是次要的, 保住自己的小命才是王道. 这就是使用 Sarsa 方法的不同之处.

