import gymnasium as gym 
import torch 
import torch.nn as nn
import copy
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage
from tensordict import TensorDict
import wandb
import ale_py
import numpy as np



class Critic(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.output = n_actions 

        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size = 8, stride=4),
            nn.ReLU(), 
            nn.Conv2d(32, 64, kernel_size = 6, stride=2),
            nn.ReLU(), 
            nn.Conv2d(64, 128, kernel_size = 3, stride=1),
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
            y = self.forward(x.unsqueeze(0))
        else :
            y = self.forward(x)
        
        qvalue = torch.gather(y, 1, a)
        return qvalue

    def optimal_action(self, x):
        if x.ndim == 3:
            y = self.forward(x.unsqueeze(0))
            indices = torch.argmax(y, dim=1).unsqueeze(1)
        else :
            y = self.forward(x)
            indices = torch.argmax(y, dim=1).unsqueeze(1)
        
        return indices, torch.gather(y, 1, indices)

class Actor(nn.Module):
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

    def sample(self, x):  #batch
        if x.ndim == 3:
            logits = self.forward(x.unsqueeze(0))
        else :
             logits = self.forward(x) 
        # dist = torch.distributions.categorical.Categorical(logits = logits)
        # action = dist.sample()
        # log_prob = dist.log_prob(torch.tensor(action))

        actions, log_probs = self.gumbel_softmax_sample(logits = logits, temperature = 50.0)
        return actions, log_probs

    def sample_gumbel(self, logits, eps=1e-20):
        U = torch.rand_like(logits)
        sample = -torch.log(-torch.log(U + eps) + eps)
        return sample

    def gumbel_softmax_sample(self, logits, temperature):
        y = logits + self.sample_gumbel(logits)
        y = y/temperature
        dist = torch.nn.Softmax()
        probs = dist(y)
        action = torch.multinomial(probs, num_samples=1)
        log_probs = torch.log(probs.gather(1, action))
        return action, log_probs

    def gumbel_softmax(self, logits, temperature, hard=False):
        action, log_probs = self.gumbel_softmax_sample(logits, temperature)
        return action, log_probs



class SAC:
    def __init__(self):
        self.env = gym.make("ALE/Pong-v5", obs_type='grayscale', frameskip=4)
        self.nactions = self.env.action_space.n 

        self.rb = TensorDictReplayBuffer(storage=LazyTensorStorage(50000), batch_size=32)

        self.criticA = Critic(n_actions = self.nactions)
        self.criticAtarget = copy.deepcopy(self.criticA)
        self.optimizerA = torch.optim.Adam(self.criticA.parameters(), lr=0.0003)

        self.criticB = Critic(n_actions = self.nactions)
        self.criticBtarget = copy.deepcopy(self.criticB)
        self.optimizerB = torch.optim.Adam(self.criticB.parameters(), lr=0.0003)

        self.actor = Actor(n_actions = self.nactions)
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=0.0003)


        self.criticA.cuda()
        self.criticAtarget.cuda()
        self.criticB.cuda()
        self.criticBtarget.cuda()
        self.actor.cuda()


    def train(self, n_episodes):

        wandb.init(
            project = 'pong'
        )

        gamma = 0.99 
        ep_reward = 0
        eps_reward = []
        learning_steps = 5000
        alpha = 0.5
        steps = 0 
        for i in range(0, n_episodes):
            done = False 
            epl = 0 
            ep_reward = 0 
            obs, _ = self.env.reset()

            while not done:
                steps+=1
                if steps < learning_steps:
                    action = self.env.action_space.sample()
                    next_obs, reward, done, _, _ = self.env.step(action)

                else :
                    action, _ = self.actor.sample(torch.tensor(obs, dtype=torch.float32, device='cuda').unsqueeze(0))
                    next_obs, reward, done, _, _ = self.env.step(action)

                action = torch.reshape(torch.tensor(action, device='cpu'), [1,])
                td = TensorDict({
                    "observation" : torch.tensor(obs, dtype=torch.float32, device='cpu').unsqueeze(0),
                    "action" : action, 
                    "next_observation" : torch.tensor(next_obs, dtype=torch.float32, device='cpu').unsqueeze(0),
                    "reward" : torch.tensor(reward, dtype=torch.float32, device='cpu'),
                    "done" : torch.tensor(done, dtype=torch.float32, device='cpu').to(torch.int)
                }, batch_size=())

                self.rb.add(td)

                obs = next_obs
                epl+=1 
                ep_reward = ep_reward + reward 

            if len(self.rb.storage) > learning_steps and steps%4==0:
                batch_data = self.rb.sample()
                obs = batch_data['observation'].to('cuda')
                action = batch_data['action'].to('cuda')
                next_obs = batch_data['next_observation'].to('cuda')
                reward = batch_data['reward'].to('cuda')
                done = batch_data['done'].to('cuda')

                qvalue_a = self.criticA.qvalue(obs, action).squeeze(1)
                qvalue_b = self.criticB.qvalue(obs, action).squeeze(1)

                next_action, log_prob = self.actor.sample(next_obs)

                target_a = self.criticAtarget.qvalue(next_obs, next_action)
                target_b = self.criticBtarget.qvalue(next_obs, next_action)

                target = (torch.min(target_a, target_b) - alpha*log_prob).squeeze(1)
                target = reward + gamma*(1-done)*target 

                loss = nn.MSELoss()
                critic_loss_a = loss(qvalue_a, target)
                self.optimizerA.zero_grad()
                critic_loss_a.backward(retain_graph=True)
                self.optimizerA.step()

                critic_loss_b = loss(qvalue_b, target)
                self.optimizerB.zero_grad()
                critic_loss_b.backward()
                self.optimizerB.step()

                action_predicted, log_prob_predicted = self.actor.sample(obs)
                qvalue_a = self.criticA.qvalue(obs, action_predicted).squeeze(1)
                qvalue_b = self.criticB.qvalue(obs, action_predicted).squeeze(1)

                qvalue = torch.min(qvalue_a, qvalue_b)
                actor_loss = -qvalue + alpha*log_prob_predicted 
                actor_loss = torch.mean(actor_loss)
                actor_loss = torch.tensor(actor_loss, requires_grad = True)
                self.optimizer.zero_grad()
                actor_loss.backward()
                self.optimizer.step()

                tau = 0.005
                if steps%10000 == 0:
                    for target_param, current_param in zip(self.criticAtarget.parameters(), self.criticA.parameters()):
                        target_param.data.copy_(tau * current_param.data + (1-tau)*target_param.data)
                    for target_param, current_param in zip(self.criticBtarget.parameters(), self.criticB.parameters()):
                        target_param.data.copy_(tau * current_param.data + (1-tau)*target_param.data)

            eps_reward.append(ep_reward)
            wandb.log({
                "Episode Reward " : ep_reward,
                "Reward" : np.mean(eps_reward),
                "Episode Length" : epl
            })

                



dqn_pong = SAC()
dqn_pong.train(100)









        




        





               
