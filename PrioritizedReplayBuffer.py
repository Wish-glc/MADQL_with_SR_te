from collections import deque
import numpy as np
import random

class PrioritizedReplayBuffer():
    def __init__(self, maxlen,n_actions):
        # 初始化两个双端队列：buffer存储经验数据，priorities存储对应优先级，都设置最大长度限制
        # 保存动作空间维度n_actions
        # 初始化内存计数器mem_cntr为0，用于跟踪当前存储的经验数量
        self.buffer = deque(maxlen=maxlen)
        self.priorities = deque(maxlen=maxlen)
        
        self.n_actions = n_actions
        self.mem_cntr = 0

    def add(self, experience):  # 将经验数据 experience 添加到 buffer 中
        
        #1. 创建动作的One-Hot编码：将离散动作转换为向量形式，只有对应动作位置为1，其余为0
        actions = np.zeros(self.n_actions)  # 创建一个长度为n_actions的向量，所有元素初始化为0
        actions[experience[1]] = 1.0    # 将对应动作位置的元素设置为1.0

        #2. 替换经验中的动作数据（元组→列表→元组）
        experience = list(experience)   # 将元组转换为列表
        experience[1] = actions  # 将动作向量替换为动作向量
        experience = tuple(experience)  # 将列表转换为元组

        #3. 添加到缓冲区
        self.buffer.append(experience)  # 将更新后的元组添加到缓冲区
        self.priorities.append(max(self.priorities, default=1))     # 添加优先级，新经验的优先级设为当前最大值（首次添加时默认为1）

        #4. 内存计数更新
        self.mem_cntr += 1  # 内存计数更新，记录添加的经验总数

        #print(self.priorities)

    def get_probabilities(self, priority_scale):
        # 计算概率
        scaled_priorities = np.array(self.priorities) ** priority_scale
        sample_probabilities = scaled_priorities / sum(scaled_priorities)
        return sample_probabilities

    def get_importance(self, probabilities):
        importance = 1/len(self.buffer) * 1/probabilities
        importance_normalized = importance / max(importance)
        return importance_normalized

    def sample(self, batch_size, priority_scale=1.0):
        sample_size = min(len(self.buffer), batch_size)
        sample_probs = self.get_probabilities(priority_scale)
        sample_indices = random.choices(range(len(self.buffer)), k=sample_size, weights=sample_probs)
        #samples = np.array(self.buffer)[sample_indices]
        samples = np.array(self.buffer, dtype=object)[sample_indices]

        importance = self.get_importance(sample_probs[sample_indices])


        return map(list, zip(*samples)), importance, sample_indices

    def set_priorities(self, indices, errors, offset=0.01):
        for i,e in zip(indices, errors):

            self.priorities[i] = abs(e) + offset

