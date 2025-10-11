import gymnasium as gym 
import torch
import torch.nn as nn
from torch.distributions import Categorical
import ale_py
import torch.optim as optim
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage
from tensordict import TensorDict
import copy
import numpy as np
import wandb



class QFunction(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.output = n_actions 

        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU()
        )

        x = torch.zeros([1, 210, 160])
        output = self.net(x.unsqueeze(0)).flatten().shape[0]
        input_features = output
        
        self.linear = nn.Sequential(
            nn.Linear(input_features, 128),
            nn.ReLU(),
            nn.Linear(128, self.output)
        )


    def forward(self, x):
        y = self.net(x)
        y = self.linear(y.view(y.size(0), -1))
        return y 

    def qvalue(self, x, a):
        if a.ndim == 1:
            a = a.unsqueeze(1)
        if x.ndim == 3:
            y = self.forward(x.unsqueeze(0))  #defines a batch
        else :
            y = self.forward(x)
        qvalue = torch.gather(y, 1, a)
        return qvalue

    def optimal_action(self, x):
        if x.ndim == 3:
            y = self.forward(x.unsqueeze(0)) #defines batch
            indices = torch.argmax(y, dim=1).unsqueeze(1)
        else :
            y = self.forward(x)
            indices = torch.argmax(y, dim=1).unsqueeze(1)
        return indices, torch.gather(y, 1, indices)


class DQN:
    def __init__(self):
        self.env = gym.make("ALE/Pong-v5", obs_type='grayscale', frameskip=4)
        self.nactions = self.env.action_space.n

        self.rb = TensorDictReplayBuffer(storage = LazyTensorStorage(50000), batch_size=32)

        self.cqf = QFunction(n_actions = self.nactions)
        self.cqf.cuda()
        self.optimizer = optim.Adam(self.cqf.parameters(), lr=0.001)
        self.tqf = copy.deepcopy(self.cqf)
        self.tqf.cuda()



    def train(self, n_episodes):

        wandb.init(
            project = 'pong'
        )

        epsilon = 0.6
        gamma = 0.99
        ep_reward = 0
        eps_reward = []
        steps = 0
    
        for i in range(0, n_episodes):
            done = False
            epl = 0 
            ep_reward = 0
            obs, _ = self.env.reset()

            while not done:
                steps+=1
                if(np.random.rand() < epsilon):
                    action = self.env.action_space.sample()
                    next_obs, reward, done, _, _ = self.env.step(action)
                
                else :
                    action, _ = self.cqf.optimal_action(torch.tensor(obs, dtype=torch.float32, device='cuda').unsqueeze(0))
                    next_obs, reward, done, _, _ = self.env.step(action)

                action = torch.reshape(torch.tensor(action, device='cpu'), [1,])
                td = TensorDict({
                    "observation" : torch.tensor(obs, dtype=torch.float32, device='cpu').unsqueeze(0), 
                    "action" : action,
                    "next_observation" : torch.tensor(next_obs, dtype=torch.float32, device='cpu').unsqueeze(0), 
                    "reward" : torch.tensor(reward, dtype=torch.float32, device='cpu'),
                    "done" : torch.tensor(done, dtype=torch.float32, device='cpu').to(torch.int)
                }, 
                batch_size=())

                self.rb.add(td)

                obs = next_obs
                epl+=1
                ep_reward = ep_reward+reward

            
            if len(self.rb.storage) > 50000 and steps%4==0:
                batch_data = self.rb.sample()
                obs = batch_data['observation'].to('cuda')
                action = batch_data['action'].to('cuda')
                next_obs = batch_data['next_observation'].to('cuda')
                reward = batch_data['reward'].to('cuda')
                done = batch_data['done'].to('cuda')

                current_qvalue = self.cqf.qvalue(obs, action).squeeze(1)

                _, target_values = self.tqf.optimal_action(next_obs)
                target_values = target_values.squeeze(1)
                target_qvalue = reward + gamma*(1-done)*(target_values)

                loss = nn.MSELoss()
                output = loss(current_qvalue, target_qvalue)
                self.optimizer.zero_grad()
                output.backward()
                self.optimizer.step()

                if steps%10000 == 0:
                    for target_param, current_param in zip(self.tqf.parameters(), self.cqf.parameters()):
                        target_param.data.copy_(0.01 * current_param.data + (1-0.01)*target_param.data)

            eps_reward.append(ep_reward)
            wandb.log({
                "Episode Reward " : ep_reward,
                "Reward" : np.mean(eps_reward),
                "Episode Length" : epl
            })




dqn_pong = DQN()
dqn_pong.train(5000)





