# *Actor-critic (AC) Algorithm & Asynchronous Advantage Actor-critic (A3C) Algorithm*

----------



## 1 *Actor-critic (AC) Algorithm*

[强化学习(十四) Actor-Critic](https://www.cnblogs.com/pinard/p/10272023.html)

### 1.1 *AC* 算法简介

- *Actor-Critic* 从名字上看包括两部分，演员 (*Actor*) 和评价者 (*Critic*)。其中 ***Actor* 使用策略函数，负责生成动作 (*Action*) 并和环境交互**。而 ***Critic* 使用价值函数，负责评估 *Actor* 的表现，并指导Actor下一阶段的动作**。

- 在 *Actor-Critic* 算法中，我们需要做两组近似，这也已经在 *Policy-based Reinforcement Learning* 中提到：

- 第一组是策略函数的近似：

    ​																				$\pi_{\theta}(s, a) \ = \ P(a|s, \theta) \ \approx \ \pi(a|s)$

- 第二组时价值函数的近似，对于状态价值和动作价值函数分别是：

    ​																				$\hat v(s, w) \ \approx \ v_\pi(s)$

    ​																				$\hat q(s, a, w) \ \approx \ q_\pi(s, a)$

- 在之前学过的 *Monte-Carlo Policy Gradient (REINFORCE)* 算法中，我们通过**随机梯度上升 *Stochastic Gradient Ascent*** 的方法来更新网络参数，参考策略梯度定理，并使用返回值 $v_t$ 来作为 $Q^{\pi_\theta}(s_t, a_t)$ 的 *unbiased sample*，得出策略的参数更新公式为：

    ​																				$\Delta \theta \ = \ \alpha \nabla_\theta \log \pi_\theta(s_t, a_t)v_t$

- 在 *AC* 系统中的 *Actor* 里，需要使用上述策略更新的公式，来不断更新参数，生成最优动作。但与 *Monte-Carlo Policy Gradient (REINFORCE)* 不同的是，公式中的 $v_t$ 不再来自于蒙特卡罗采样，而应该从 *AC* 系统中的另一个组成部分 —— *Critic* 那里得到。

- 对于 *Critic*，我们参考 *DQN* 的做法，使用一个 *Q* 网络来进行表示，其输入可以为状态，输出则为最优动作的价值或全部动作各自的价值。

- 因此，***AC* 算法的主要框架**为：

    - *Critic* 通过 *Q* 网络计算当前状态的最优价值 $v_t$，返回给 *Actor*；从 *Actor* 得到反馈和新的状态后，更新 *Q* 网络
    - *Actor* 利用 $v_t$ 迭代更新策略函数的参数 $\theta$，并以此选择动作，得到反馈和新的状态，返回给 *Critic*

-----



## 2 *Asynchronous Advantage Actor-critic (A3C) Algorithm*

[强化学习(十五) A3C](https://www.cnblogs.com/pinard/p/10334127.html)

### 2.1 *AC* 算法的优缺点

- 优点：可以进行单步更新, 相较于传统 *Policy Gradient* 算法的回合更新要快
- 缺点：算法难以收敛。Actor的行为取决于 Critic 的 Value，但是由于 Critic本身就很难收敛，actor一起更新的话就更难收敛了。

----

### 2.2 *A3C* 算法简介

- 可以使用经验回放来解决难以收敛的问题，但经验回放本身也有自己的问题：回放池经验数据相关性太强，用于训练的时候效果很可能不佳。
- *A3C* 算法则是基于经验回放的思想，使用多线程同时与环境交互学习的方法，解决了经验回放数据相关性的问题。在多线程中，每个线程都将自身与环境交互学习到的成功汇总起来，整理并保存在一个公共的地方，并定期把大家齐心学习的成果拿出来，指导自己和环境后面的学习交互，以达到**异步并行**的目的。

------------

### 2.3 *A3C* 相较于 *AC* 算法的改进

- **异步训练框架**：

    - 图中上面的 ***Global Network*** 就是上一节说的共享的公共部分，主要是一个公共的神经网络模型，这个神经网络**包括 *Actor* 网络和*Critic* 网络两部分的功能**。下面有 ***n* 个 *worker* 线程**，**每个线程里有和公共的神经网络一样的网络结构**，每个线程会独立的和环境进行交互得到经验数据，这些线程之间**互不干扰，独立运行**。
    - 每个线程和环境交互到一定量的数据后，就计算在自己线程里的神经网络损失函数的梯度，但是这些梯度却并不更新自己线程里的神经网络，而是去更新公共的神经网络。也就是 ***n* 个线程会独立的使用累积的梯度分别更新公共部分的神经网络模型参数**。**每隔一段时间，线程会将自己的神经网络的参数更新为公共神经网络的参数，进而指导后面的环境交互**。
    - 可见，公共部分的网络模型就是我们要学习的模型，而线程里的网络模型主要是用于和环境交互使用的，这些线程里的模型可以帮助线程更好的和环境交互，拿到高质量的数据帮助模型更快收敛。

- **网络结构优化**：

    - 与 *AC* 算法不同的是，这里我们将 *Actor* 和 *Critic* 网络在逻辑上合并，即输入状态，输出状态价值 $v_t$ 和对应的策略 $\pi$，当然，二值在物理上仍是两个独立的网络，分别处理：

        <img src="./cut/截屏2021-05-14 下午10.18.47.png" alt="avatar" style="zoom:50%;" />

- ***Critic* 评估点的优化**：

    - 在 *AC* 算法中，我们讨论了使用单步采样来近似估计 $Q(S, A)$，即：$Q(S, A) = R + \gamma V(S')$，其中 $V(S)$ 的值需要通过 *Critic* 网络学习得到。因此优势函数可以表达为：

        ​																		$A(S, t) \ = \ R + \gamma V(S') - V(S)$

    - 在 *A3C* 算法中，**使用 *N* 步采样以加快收敛**，因此优势函数可以表示为：

        ​													$A(S, t) \ = \ R_t + \gamma R_{t+1} + ... + \gamma^{n-1}R_{t+n-1} + \gamma^n V(S') - V(S)$

        对于 *Actor* 和 *Critic* 的损失函数部分，和 *AC* 基本相同。有一个小的优化点就是在  *Actor-Critic* 策略函数的损失函数中，加入了策略 $\pi$ 的熵项，系数为 *c*，即策略参数的梯度更新变成如下形式：

        ​															$\Delta \theta \ = \ \alpha \nabla_\theta \log \pi_\theta(s_t, a_t)A(S, t) + c \nabla_\theta H(\pi(S_t, \theta))$

-----









