# ZJU-CSE-AI-ML-DQN-Robot
浙江大学控制科学与工程学院《人工智能与机器学习》课程“机器人自动走迷宫”实验 DFS+QN+DQN

- 资料仅供参考，请勿直接使用或者抄袭！
- 如果有用，请为我点一颗星，谢谢！

<center><strong><font size="8">机器人自动走迷宫



### 一 题目背景

#### 1.1 实验题目

在本实验中，要求分别使用基础搜索算法和 ***Deep QLearning*** 算法，完成机器人自动走迷宫。

<img src="https://github.com/Antom2000/ZJU-CSE-AI-ML-DQN-Robot/blob/main/PHOTO/dqn_size10.gif" />

<strong>图1 地图（size10）</strong>



​	如上图所示，左上角的红色椭圆既是起点也是机器人的初始位置，右下角的绿色方块是出口。
游戏规则为：从起点开始，通过错综复杂的迷宫，到达目标点(出口)。

+ 在任一位置可执行动作包括：向上走 `'u'`、向右走 `'r'`、向下走 `'d'`、向左走 `'l'`。

+ 执行不同的动作后，根据不同的情况会获得不同的奖励，具体而言，有以下几种情况。
  - 撞墙
  - 走到出口
  - 其余情况

+ 需要您分别实现**基于基础搜索算法**和 **Deep QLearning 算法**的机器人，使机器人自动走到迷宫的出口。

#### 1.2 实验要求

+ 使用 ***Python*** 语言。
+ 使用基础搜索算法完成机器人走迷宫。
+ 使用 ***Deep QLearning*** 算法完成机器人走迷宫。
+ 算法部分需要自己实现，不能使用现成的包、工具或者接口。

#### 1.3 实验使用重要python包

```python
import os
import random
import numpy as np
import torch
```

### 二 迷宫介绍

###### 通过迷宫类 ***Maze*** 可以随机创建一个迷宫。

1. 使用  ***Maze(maze_size=size)***  来随机生成一个 size * size 大小的迷宫。
2. 使用 ***print()*** 函数可以输出迷宫的 ***size*** 以及画出迷宫图
3. 红色的圆是机器人初始位置
4. 绿色的方块是迷宫的出口位置

<img src="https://github.com/Antom2000/ZJU-CSE-AI-ML-DQN-Robot/blob/main/PHOTO/dqn_size10-1638271871100.gif" />

<strong>图2 gif地图（size10）</strong>



###### Maze 类中重要的成员方法如下：

1. sense_robot() ：获取机器人在迷宫中目前的位置。

> return：机器人在迷宫中目前的位置。

2. move_robot(direction) ：根据输入方向移动默认机器人，若方向不合法则返回错误信息。

> direction：移动方向, 如:"u", 合法值为： ['u', 'r', 'd', 'l']

> return：执行动作的奖励值

3. can_move_actions(position)：获取当前机器人可以移动的方向

> position：迷宫中任一处的坐标点 

> return：该点可执行的动作，如：['u','r','d']

4. is_hit_wall(self, location, direction)：判断该移动方向是否撞墙

> location, direction：当前位置和要移动的方向，如(0,0) , "u"

> return：True(撞墙) / False(不撞墙)

5. draw_maze()：画出当前的迷宫

### 三 算法介绍

#### 3.1 深度优先算法

###### 算法具体步骤：

- 选取图中某一顶点$V_i$为出发点，访问并标记该顶点；

- 以Vi为当前顶点，依次搜索$V_i$的每个邻接点$V_j$，若$V_j$未被访问过，则访问和标记邻接点$V_j$，若$V_j$已被访问过，则搜索$V_i$的下一个邻接点；

- 以$V_j$为当前顶点，重复上一步骤），直到图中和$V_i$有路径相通的顶点都被访问为止；

- 若图中尚有顶点未被访问过（非连通的情况下），则可任取图中的一个未被访问的顶点作为出发点，重复上述过程，直至图中所有顶点都被访问。

###### 时间复杂度：

​	查找每个顶点的邻接点所需时间为$O(n^2)$，n为顶点数，算法的时间复杂度为$O(n^2)$

#### 3.2 强化学习QLearning算法

​	***Q-Learning*** 是一个值迭代（Value Iteration）算法。与策略迭代（Policy Iteration）算法不同，值迭代算法会计算每个”状态“或是”状态-动作“的值（Value）或是效用（Utility），然后在执行动作的时候，会设法最大化这个值。 因此，对每个状态值的准确估计，是值迭代算法的核心。通常会考虑**最大化动作的长期奖励**，即不仅考虑当前动作带来的奖励，还会考虑动作长远的奖励。

##### 3.2.1 Q值计算与迭代

​	***Q-learning*** 算法将状态（state）和动作（action）构建成一张 Q_table 表来存储 Q 值，Q 表的行代表状态（state），列代表动作（action）：

<img src="https://imgbed.momodel.cn/20200914161241.png" width=400px style="zoom: 67%;" />

​	在 ***Q-Learning*** 算法中，将这个长期奖励记为 Q 值，其中会考虑每个 ”状态-动作“ 的 Q 值，具体而言，它的计算公式为：

$$
Q(s_{t},a) = R_{t+1} + \gamma \times\max_a Q(a,s_{t+1})
$$

​	也就是对于当前的“状态-动作” $(s_{t},a)$，考虑执行动作 $a$ 后环境奖励 $R_{t+1}$，以及执行动作 $a$ 到达 $s_{t+1}$后，执行任意动作能够获得的最大的Q值 $\max_a Q(a,s_{t+1})$，$\gamma$ 为折扣因子。

​	计算得到新的 Q 值之后，一般会使用更为保守地更新 Q 表的方法，即引入松弛变量 $alpha$ ，按如下的公式进行更新，使得 Q 表的迭代变化更为平缓。

$$
Q(s_{t},a) = (1-\alpha) \times Q(s_{t},a) + \alpha \times(R_{t+1} + \gamma \times\max_a Q(a,s_{t+1}))
$$

##### 3.2.2 机器人动作的选择

在强化学习中，**探索-利用** 问题是非常重要的问题。 具体来说，根据上面的定义，会尽可能地让机器人在每次选择最优的决策，来最大化长期奖励。但是这样做有如下的弊端：    

1. 在初步的学习中，Q 值是不准确的，如果在这个时候都按照 Q 值来选择，那么会造成错误。
2. 学习一段时间后，机器人的路线会相对固定，则机器人无法对环境进行有效的探索。

因此需要一种办法，来解决如上的问题，增加机器人的探索。通常会使用 **epsilon-greedy** 算法：

1. 在机器人选择动作的时候，以一部分的概率随机选择动作，以一部分的概率按照最优的 Q 值选择动作。
2. 同时，这个选择随机动作的概率应当随着训练的过程逐步减小。

<center><img src="http://imgbed.momodel.cn/20200602153554.png" width=400 style="zoom: 60%;" ><img src="http://imgbed.momodel.cn/20200601144827.png" width=400 style="zoom: 60%;" >



##### 3.2.3  Q-Learning 算法的学习过程

<img src="http://imgbed.momodel.cn/20200601170657.png" width=900 style="zoom: 80%;" >

#####  3.2.4 Robot 类

​	在本作业中提供了 ***QRobot*** 类，其中实现了 Q 表迭代和机器人动作的选择策略，可通过 `from QRobot import QRobot` 导入使用。

**QRobot 类的核心成员方法**

1. sense_state()：获取当前机器人所处位置

> return：机器人所处的位置坐标，如： (0, 0)

2. current_state_valid_actions()：获取当前机器人可以合法移动的动作

> return：由当前合法动作组成的列表，如： ['u','r']

3. train_update()：以**训练状态**，根据 QLearning 算法策略执行动作

> return：当前选择的动作，以及执行当前动作获得的回报, 如： 'u', -1

4. test_update()：以**测试状态**，根据 QLearning 算法策略执行动作

> return：当前选择的动作，以及执行当前动作获得的回报, 如：'u', -1

5. reset()

> return：重置机器人在迷宫中的位置

##### 3.2.5 Runner 类

​	***QRobot*** 类实现了 ***QLearning*** 算法的 Q 值迭代和动作选择策略。在机器人自动走迷宫的训练过程中，需要不断的使用 ***QLearning*** 算法来迭代更新 Q 值表，以达到一个“最优”的状态，因此封装好了一个类 ***Runner*** 用于机器人的训练和可视化。可通过 `from Runner import Runner` 导入使用。

**Runner 类的核心成员方法：**

1. run_training(training_epoch, training_per_epoch=150): 训练机器人，不断更新 Q 表，并讲训练结果保存在成员变量 train_robot_record 中

> training_epoch, training_per_epoch: 总共的训练次数、每次训练机器人最多移动的步数

2. run_testing()：测试机器人能否走出迷宫

3. generate_gif(filename)：将训练结果输出到指定的 gif 图片中

> filename：合法的文件路径,文件名需以 `.gif` 为后缀

4. plot_results()：以图表展示训练过程中的指标：Success Times、Accumulated Rewards、Runing Times per Epoch

#### 3.3 DQN

​	DQN 算法使用神经网络来近似值函数，算法框图如下。

<img src="https://imgbed.momodel.cn/20200918101137.png" alt="Image" style="zoom:50%;" />

​	在本次实验中，使用提供的神经网络来预计四个动作的评估分数，同时输出评估分数。

**ReplayDataSet 类的核心成员方法**

+ add(self, state, action_index, reward, next_state, is_terminal) 添加一条训练数据

> state: 当前机器人位置

> action_index: 选择执行动作的索引

> reward： 执行动作获得的回报

> next_state：执行动作后机器人的位置

> is_terminal：机器人是否到达了终止节点（到达终点或者撞墙）

+ random_sample(self, batch_size)：从数据集中随机抽取固定batch_size的数据

> batch_size: 整数，不允许超过数据集中数据的个数

+ **build_full_view(self, maze)：开启金手指，获取全图视野**

> maze: 以 Maze 类实例化的对象



### 四 求解结果

#### 4.1 深度优先

​	编写深度优先搜索算法，并进行测试，通过使用堆栈的方式，来进行一层一层的迭代，最终搜索出路径。主要过程为，入口节点作为根节点，之后查看此节点是否被探索过且是否存在子节点，若满足条件则拓展该节点，将该节点的子节点按照先后顺序入栈。若探索到一个节点时，此节点不是终点且没有可以拓展的子节点，则将此点出栈操作，循环操作直到找到终点。

###### 测试结果如下：

- 若`maze_size=5`，运行基础搜索算法，最终成果如下：

```python
搜索出的路径： ['r', 'd', 'r', 'd', 'd', 'r', 'r', 'd']
恭喜你，到达了目标点
Maze of size (5, 5)
```

![img](data:image/png;base64,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)

<strong>图3 基础搜索地图（size5）</strong>



- 若`maze_size=10`，运行基础搜索算法，最终成果如下：

```python
搜索出的路径： ['r', 'r', 'r', 'r', 'r', 'r', 'r', 'd', 'r', 'd', 'd', 'd', 'r', 'd', 'd', 'd', 'l', 'd', 'd', 'r']
恭喜你，到达了目标点
Maze of size (10, 10)
```

![img](data:image/png;base64,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)

<strong>图4 基础搜索地图（size10）</strong>



- 若`maze_size=20`，运行基础搜索算法，最终成果如下：

```python
搜索出的路径： ['d', 'r', 'u', 'r', 'r', 'r', 'r', 'd', 'r', 'd', 'r', 'r', 'r', 'r', 'd', 'd', 'r', 'd', 'd', 'd', 'd', 'r', 'r', 'r', 'r', 'r', 'd', 'r', 'r', 'd', 'r', 'd', 'd', 'l', 'l', 'd', 'd', 'd', 'd', 'd', 'r', 'd', 'd', 'r']
恭喜你，到达了目标点
Maze of size (20, 20)
```

![img](data:image/png;base64,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)

<strong>图5 基础搜索地图（size20）</strong>



###### 部分代码如下：

```python
def myDFS(maze):
        """
        对迷宫进行深度优先搜索
        :param maze: 待搜索的maze对象
        """
        start = maze.sense_robot()
        root = SearchTree(loc=start)
        queue = [root]  # 节点堆栈，用于层次遍历
        h, w, _ = maze.maze_data.shape
        is_visit_m = np.zeros((h, w), dtype=np.int)  # 标记迷宫的各个位置是否被访问过
        path = []  # 记录路径
        peek = 0
        while True:
            current_node = queue[peek]  # 栈顶元素作为当前节点
            #is_visit_m[current_node.loc] = 1  # 标记当前节点位置已访问
            if current_node.loc == maze.destination:  # 到达目标点
                path = back_propagation(current_node)
                break
            if current_node.is_leaf() and is_visit_m[current_node.loc] == 0:  # 如果该点存在叶子节点且未拓展
                is_visit_m[current_node.loc] = 1  # 标记该点已拓展
                child_number = expand(maze, is_visit_m, current_node)
                peek+=child_number  # 开展一些列入栈操作
                for child in current_node.children:
                    queue.append(child)  # 叶子节点入栈
            else:
                queue.pop(peek)  # 如果无路可走则出栈
                peek-=1
        return path
```



#### 4.2 QLearning

​	在算法训练过程中，首先读取机器人当前位置，之后将当前状态加入Q值表中，如果表中已经存在当前状态则不需重复添加。之后，生成机器人的需要执行动作，并返回地图奖励值、查找机器人现阶段位置。接着再次检查并更新Q值表，衰减随机选取动作的可能性。

​	***QLearning***算法实现过程中，主要是对Q值表的计算更新进行了修改和调整，调整后的Q值表在运行时性能优秀，计算速度快且准确性、稳定性高。之后调节了随机选择动作可能性的衰减率。因为在测试过程中发现，如果衰减太慢的话会导致随机性太强，间接的减弱了奖励的作用，故最终通过调整，发现衰减率取0.5是一个较为优秀的且稳定的值。

###### 	部分代码如下：

```python
    def train_update(self):
        """
        以训练状态选择动作，并更新相关参数
        :return :action, reward 如："u", -1
        """
        self.state = self.maze.sense_robot()  # 获取机器人当初所处迷宫位置

        # 检索Q表，如果当前状态不存在则添加进入Q表
        if self.state not in self.q_table:
            self.q_table[self.state] = {a: 0.0 for a in self.valid_action}

        action = random.choice(self.valid_action) if random.random() < self.epsilon else max(self.q_table[self.state], key=self.q_table[self.state].get)  # action为机器人选择的动作
        reward = self.maze.move_robot(action)  # 以给定的方向移动机器人,reward为迷宫返回的奖励值
        next_state = self.maze.sense_robot()  # 获取机器人执行指令后所处的位置

        # 检索Q表，如果当前的next_state不存在则添加进入Q表
        if next_state not in self.q_table:
            self.q_table[next_state] = {a: 0.0 for a in self.valid_action}

        # 更新 Q 值表
        current_r = self.q_table[self.state][action]
        update_r = reward + self.gamma * float(max(self.q_table[next_state].values()))
        self.q_table[self.state][action] = self.alpha * self.q_table[self.state][action] +(1 - self.alpha) * (update_r - current_r)

        self.epsilon *= 0.5  # 衰减随机选择动作的可能性

        return action, reward
```

###### 测试结果如下：

- 若`maze_size=3`，运行强化学习搜索算法，最终成果如下：

<img src="https://github.com/Antom2000/ZJU-CSE-AI-ML-DQN-Robot/blob/main/PHOTO/image-20211209193708357.png" />

<strong>图6 强化学习搜索gif地图（size3）</strong>


<img src="https://github.com/Antom2000/ZJU-CSE-AI-ML-DQN-Robot/blob/main/PHOTO/dqn_size3.gif" />

<center><strong>图7 训练结果</strong>
</center>


- 若`maze_size=5`，运行强化学习搜索算法，最终成果如下：

<img src="https://github.com/Antom2000/ZJU-CSE-AI-ML-DQN-Robot/blob/main/PHOTO/dqn_size5.gif" />

<strong>图8 强化学习搜索gif地图（size5）</strong>

<img src="https://github.com/Antom2000/ZJU-CSE-AI-ML-DQN-Robot/blob/main/PHOTO/image-20211209194029128.png" />

<strong>图9 训练结果</strong>



- 若`maze_size=10`，运行强化学习搜索算法，最终成果如下：

<img src="https://github.com/Antom2000/ZJU-CSE-AI-ML-DQN-Robot/blob/main/PHOTO/dqn_size10-1638273841747.gif" />

<strong>图10 强化学习搜索gif地图（size10）</strong>


<img src="https://github.com/Antom2000/ZJU-CSE-AI-ML-DQN-Robot/blob/main/PHOTO/image-20211209194050869.png" />

<strong>图11 训练结果</strong>



- 若`maze_size=11`，运行强化学习搜索算法，最终成果如下：

<img src="https://github.com/Antom2000/ZJU-CSE-AI-ML-DQN-Robot/blob/main/PHOTO/dqn_size11.gif" />

<strong>图12 强化学习搜索gif地图（size11）</strong>


<img src="https://github.com/Antom2000/ZJU-CSE-AI-ML-DQN-Robot/blob/main/PHOTO/image-20211209194108924.png" />
<img src="https://github.com/Antom2000/ZJU-CSE-AI-ML-DQN-Robot/blob/main/PHOTO/image-20211209194651397.png" />

<strong>图13 训练结果</strong>


​	经过测试，强化学习搜索算法可以快速给出走出迷宫的路径并且随着训练轮次增加，成功率也逐渐上升。当训练轮次足够时，最终后期准确率可以达到100%。

#### 4.3 DQN

​	在***Q-Learning*** 的基础上，使用神经网络来估计评估分数，用于决策之后的动作。只需在***Q-Learning***相应部分替换为神经网络的输出即可。

###### 测试结果如下：

- 若`maze_size=3`，运行***DQN***算法，最终成果如下：

  <img src="https://github.com/Antom2000/ZJU-CSE-AI-ML-DQN-Robot/blob/main/PHOTO/image-20220110225441520.png" />

  <strong>图14 训练结果</strong>
  

- 若`maze_size=5`，运行***DQN***算法，最终成果如下：

  <img src="https://github.com/Antom2000/ZJU-CSE-AI-ML-DQN-Robot/blob/main/PHOTO/image-20220110225441520.png" />

  <strong>图15 训练结果</strong>
  

- 若`maze_size=10`，运行***DQN***算法，最终成果如下：

  <img src="https://github.com/Antom2000/ZJU-CSE-AI-ML-DQN-Robot/blob/main/PHOTO/image-20220110225441520.png" />

  <strong>图16 训练结果</strong>
  

#### 4.4 提交结果测试

##### 4.4.1 基础搜索算法测试

<img src="https://github.com/Antom2000/ZJU-CSE-AI-ML-DQN-Robot/blob/main/PHOTO/image-20211208160808475.png" />

<strong>图17 基础搜索算法路径</strong>




​	用时0秒

##### 4.4.2 强化学习算法（初级）

<img src="https://github.com/Antom2000/ZJU-CSE-AI-ML-DQN-Robot/blob/main/PHOTO/image-20211130200955430.png" />

<strong>图18 强化学习算法（初级）</strong>



​	用时0秒

##### 4.4.3 强化学习算法（中级）

<img src="https://github.com/Antom2000/ZJU-CSE-AI-ML-DQN-Robot/blob/main/PHOTO/image-20211130200945686.png" />

<strong>图19 强化学习算法（中级）</strong>



​	用时0秒

##### 4.4.4 强化学习算法（高级）

<img src="https://github.com/Antom2000/ZJU-CSE-AI-ML-DQN-Robot/blob/main/PHOTO/image-20211130201010799.png" />

<strong>图20 强化学习算法（高级）</strong>



​	用时0秒

##### 4.4.5 DQN算法（初级）

<img src="https://github.com/Antom2000/ZJU-CSE-AI-ML-DQN-Robot/blob/main/PHOTO/image-20220110231101445.png" />

<strong>图21 DQN算法（初级）</strong>



​	用时2秒

##### 4.4.6 DQN算法（中级）

<img src="https://github.com/Antom2000/ZJU-CSE-AI-ML-DQN-Robot/blob/main/PHOTO/image-20220110231113620.png" />

<strong>图22 DQN算法（中级）</strong>



​	用时3秒

##### 4.4.7 DQN算法（高级）

<img src="https://github.com/Antom2000/ZJU-CSE-AI-ML-DQN-Robot/blob/main/PHOTO/image-20220110225954118.png" />

<strong>图23 DQN算法（高级）</strong>



​	用时105秒

### 五 比较分析

​	比较基础搜索算法、强化学习搜索算法和***DQN***算法可以发现，在训练轮次提升到较大数字的时候，三算法速度均很快且有较高的准确率。***DQN***算法耗费时间长，但是性能稳定。

|        算法         | 用时 |   状态   |
| :-----------------: | :--: | :------: |
| **深度优先+size20** |  0   | 到达终点 |
| **强化学习+size3**  |  0   | 到达终点 |
| **强化学习+size5**  |  0   | 到达终点 |
| **强化学习+size11** |  0   | 到达终点 |
|    **DQN+size3**    |  2   | 到达终点 |
|    **DQN+size5**    |  3   | 到达终点 |
|   **DQN+size11**    | 105  | 到达终点 |



### 六 心得与感想

​	通过本次实验，学习到了多种基础搜索算法的具体工作方式，并且在实践中实现了这些基础搜索算法；同时学习了强化学习搜索算法的工作原理和应用方式，让我对这些方法有了更加深刻的认识，也对各个环节有了更加深层次的理解，受益匪浅。



***程序清单：***

```python
# 导入相关包 
import os
import random
import numpy as np
import torch
from QRobot import QRobot
from ReplayDataSet import ReplayDataSet
from torch_py.MinDQNRobot import MinDQNRobot as TorchRobot # PyTorch版本
import matplotlib.pyplot as plt
# ------基础搜索算法-------
def my_search(maze):
    # 机器人移动方向
    move_map = {
        'u': (-1, 0), # up
        'r': (0, +1), # right
        'd': (+1, 0), # down
        'l': (0, -1), # left
    }
    # 迷宫路径搜索树
    class SearchTree(object):


        def __init__(self, loc=(), action='', parent=None):
            """
            初始化搜索树节点对象
            :param loc: 新节点的机器人所处位置
            :param action: 新节点的对应的移动方向
            :param parent: 新节点的父辈节点
            """

            self.loc = loc  # 当前节点位置
            self.to_this_action = action  # 到达当前节点的动作
            self.parent = parent  # 当前节点的父节点
            self.children = []  # 当前节点的子节点

        def add_child(self, child):
            """
            添加子节点
            :param child:待添加的子节点
            """
            self.children.append(child)

        def is_leaf(self):
            """
            判断当前节点是否是叶子节点
            """
            return len(self.children) == 0
    def expand(maze, is_visit_m, node):
        """
        拓展叶子节点，即为当前的叶子节点添加执行合法动作后到达的子节点
        :param maze: 迷宫对象
        :param is_visit_m: 记录迷宫每个位置是否访问的矩阵
        :param node: 待拓展的叶子节点
        """
        child_number = 0  # 记录叶子节点个数
        can_move = maze.can_move_actions(node.loc)
        for a in can_move:
            new_loc = tuple(node.loc[i] + move_map[a][i] for i in range(2))
            if not is_visit_m[new_loc]:
                child = SearchTree(loc=new_loc, action=a, parent=node)
                node.add_child(child)
                child_number+=1
        return child_number  # 返回叶子节点个数
                
    def back_propagation(node):
        """
        回溯并记录节点路径
        :param node: 待回溯节点
        :return: 回溯路径
        """
        path = []
        while node.parent is not None:
            path.insert(0, node.to_this_action)
            node = node.parent
        return path

    def myDFS(maze):
        """
        对迷宫进行深度
        :param maze: 待搜索的maze对象
        """
        start = maze.sense_robot()
        root = SearchTree(loc=start)
        queue = [root]  # 节点堆栈，用于层次遍历
        h, w, _ = maze.maze_data.shape
        is_visit_m = np.zeros((h, w), dtype=np.int)  # 标记迷宫的各个位置是否被访问过
        path = []  # 记录路径
        peek = 0
        while True:
            current_node = queue[peek]  # 栈顶元素作为当前节点
            #is_visit_m[current_node.loc] = 1  # 标记当前节点位置已访问

            if current_node.loc == maze.destination:  # 到达目标点
                path = back_propagation(current_node)
                break

            if current_node.is_leaf() and is_visit_m[current_node.loc] == 0:  # 如果该点存在叶子节点且未拓展
                is_visit_m[current_node.loc] = 1  # 标记该点已拓展
                child_number = expand(maze, is_visit_m, current_node)
                peek+=child_number  # 开展一些列入栈操作
                for child in current_node.children:
                    queue.append(child)  # 叶子节点入栈
            else:
                queue.pop(peek)  # 如果无路可走则出栈
                peek-=1
            # 出队
            #queue.pop(0)

        return path
    path = myDFS(maze)
    return path
# --------强化学习算法---------
# 导入相关包
import random
from Maze import Maze
from Runner import Runner
from QRobot import QRobot

class Robot(QRobot):

    valid_action = ['u', 'r', 'd', 'l']

    def __init__(self, maze, alpha=0.5, gamma=0.9, epsilon=0.5):
        """
        初始化 Robot 类
        :param maze:迷宫对象
        """
        self.maze = maze
        self.state = None
        self.action = None
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon  # 动作随机选择概率
        self.q_table = {}

        self.maze.reset_robot()  # 重置机器人状态
        self.state = self.maze.sense_robot()  # state为机器人当前状态

        if self.state not in self.q_table:  # 如果当前状态不存在，则为 Q 表添加新列
            self.q_table[self.state] = {a: 0.0 for a in self.valid_action}

    def train_update(self):
        """
        以训练状态选择动作，并更新相关参数
        :return :action, reward 如："u", -1
        """
        self.state = self.maze.sense_robot()  # 获取机器人当初所处迷宫位置

        # 检索Q表，如果当前状态不存在则添加进入Q表
        if self.state not in self.q_table:
            self.q_table[self.state] = {a: 0.0 for a in self.valid_action}

        action = random.choice(self.valid_action) if random.random() < self.epsilon else max(self.q_table[self.state], key=self.q_table[self.state].get)  # action为机器人选择的动作
        reward = self.maze.move_robot(action)  # 以给定的方向移动机器人,reward为迷宫返回的奖励值
        next_state = self.maze.sense_robot()  # 获取机器人执行指令后所处的位置

        # 检索Q表，如果当前的next_state不存在则添加进入Q表
        if next_state not in self.q_table:
            self.q_table[next_state] = {a: 0.0 for a in self.valid_action}

        # 更新 Q 值表
        current_r = self.q_table[self.state][action]
        update_r = reward + self.gamma * float(max(self.q_table[next_state].values()))
        self.q_table[self.state][action] = self.alpha * self.q_table[self.state][action] +(1 - self.alpha) * (update_r - current_r)

        self.epsilon *= 0.5  # 衰减随机选择动作的可能性

        return action, reward

    def test_update(self):
        """
        以测试状态选择动作，并更新相关参数
        :return :action, reward 如："u", -1
        """
        self.state = self.maze.sense_robot()  # 获取机器人现在所处迷宫位置

        # 检索Q表，如果当前状态不存在则添加进入Q表
        if self.state not in self.q_table:
            self.q_table[self.state] = {a: 0.0 for a in self.valid_action}
        
        action = max(self.q_table[self.state],key=self.q_table[self.state].get)  # 选择动作
        reward = self.maze.move_robot(action)  # 以给定的方向移动机器人

        return action, reward

import random
import numpy as np
import torch
from QRobot import QRobot
from ReplayDataSet import ReplayDataSet
from torch_py.MinDQNRobot import MinDQNRobot as TorchRobot # PyTorch版本
import matplotlib.pyplot as plt
from Maze import Maze
import time
from Runner import Runner
class Robot(TorchRobot):

    def __init__(self, maze):
        """
        初始化 Robot 类
        :param maze:迷宫对象
        """
        super(Robot, self).__init__(maze)
        maze.set_reward(reward={
            "hit_wall": 5.0,
            "destination": -maze.maze_size ** 2.0,
            "default": 1.0,
        })
        self.maze = maze
        self.epsilon = 0
        """开启金手指，获取全图视野"""
        self.memory.build_full_view(maze=maze)
        self.loss_list = self.train()

    def train(self):
        loss_list = []
        batch_size = len(self.memory)

        while True:
            loss = self._learn(batch=batch_size)
            loss_list.append(loss)
            success = False
            self.reset()
            for _ in range(self.maze.maze_size ** 2 - 1):
                a, r = self.test_update()
                if r == self.maze.reward["destination"]:
                    return loss_list

    def train_update(self):
        def state_train():
            state=self.sense_state()
            return state
        def action_train(state):
            action=self._choose_action(state)
            return action
        def reward_train(action):
            reward=self.maze.move_robot(action)
            return reward
        state = state_train()
        action = action_train(state)
        reward = reward_train(action)
        return action, reward

    def test_update(self):
        def state_test():
            state = torch.from_numpy(np.array(self.sense_state(), dtype=np.int16)).float().to(self.device)
            return state
        state = state_test()
        self.eval_model.eval()
        with torch.no_grad():
            q_value = self.eval_model(state).cpu().data.numpy()
        def action_test(q_value):
            action=self.valid_action[np.argmin(q_value).item()]
            return action
        def reward_test(action):
            reward=self.maze.move_robot(action)
            return reward
        action = action_test(q_value)
        reward = reward_test(action)
        return action, reward

```

