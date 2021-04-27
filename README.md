# SJTU_CS489_Reinforcement_Learning

[Project Ref 1: karroyan/CS489-Reinforcement-Learning-Project](https://github.com/karroyan/CS489-Reinforcement-Learning-Project)

----

## 1. `Pycharm conda`环境的配置

- 在`pycharm`中新建`conda`环境，`conda excutable`选择`/opt/anaconda3/bin/conda`即可。

----



## 2. Mac Pycharm gym工具安装

- pycharm terminal 运行`pip install gym`
- 运行后报错：

```
... ...
ImportError: Can't find framework /System/Library/Frameworks/OpenGL.framework.
... ...
ImportError: 
    Error occurred while running `from pyglet.gl import *`
    HINT: make sure you have OpenGL install. On Ubuntu, you can run 'apt-get install python-opengl'.
    If you're running on a server, you may need a virtual frame buffer; something like this should work:
    'xvfb-run -s "-screen 0 1400x900x24" python <your_script.py>'
```

- 之后安装`opengl`：运行：`sudo pip install pyglet==1.5.11`
- 示例程序即可正常运行：

```
import gym

env = gym.make('CartPole-v0')
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample())  # take a random action
```

----





