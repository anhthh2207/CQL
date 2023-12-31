import torch
import torch.nn as nn
from networks import DDQN, DDQN_Online
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import numpy as np
import random

class CQLAgent():
    def __init__(self, state_size, action_size, hidden_size=256, device="cpu"):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.tau = 1e-3
        self.gamma = 0.99
        
        self.network = DDQN(state_size=self.state_size,
                            action_size=self.action_size,
                            layer_size=hidden_size
                            ).to(self.device)

        self.target_net = DDQN(state_size=self.state_size,
                            action_size=self.action_size,
                            layer_size=hidden_size
                            ).to(self.device)
        
        self.optimizer = optim.Adam(params=self.network.parameters(), lr=1e-3)
        
    
    def get_action(self, state, epsilon):
        if random.random() > epsilon:
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            self.network.eval()
            with torch.no_grad():
                action_values = self.network(state)
            self.network.train()
            action = np.argmax(action_values.cpu().data.numpy(), axis=1)
        else:
            action = random.choices(np.arange(self.action_size), k=1)
        return action
        
    def learn(self, experiences):
        self.optimizer.zero_grad()
        states, actions, rewards, next_states, dones = experiences
        states, actions, rewards, next_states, dones = states.to(self.device).float(), actions.to(self.device).to(torch.int64).reshape(-1, 1), rewards.to(self.device).float().reshape(-1, 1), next_states.to(self.device).float(), dones.to(self.device).float().reshape(-1, 1)
        # print("states: ", states.shape)
        # print("dtype of state: ", states.dtype)
        # print("actions: ", actions.shape)
        # print("actions dtype: ", actions.dtype)
        # print("rewards: ", rewards.shape)
        # print("rewards dtype: ", rewards.dtype)
        # print("next_states: ", next_states.shape)
        # print("next_states dtype: ", next_states.dtype)
        # print("dones: ", dones.shape)
        # print("dones dtype: ", dones.dtype)
        with torch.no_grad():
            Q_targets_next = self.target_net(next_states).detach().max(1)[0].unsqueeze(1)
            Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        Q_a_s = self.network(states)
        Q_expected = Q_a_s.gather(1, actions)
        
        cql1_loss = torch.logsumexp(Q_a_s, dim=1).mean() - Q_a_s.mean()

        bellmann_error = F.mse_loss(Q_expected, Q_targets)
        
        q1_loss = cql1_loss + 0.5 * bellmann_error
        
        q1_loss.backward()
        clip_grad_norm_(self.network.parameters(), 1)
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.network, self.target_net)
        return q1_loss.detach().item(), cql1_loss.detach().item(), bellmann_error.detach().item()
        
        
    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)




class CQLAgent_Conv():
    def __init__(self, state_size, action_size, device="cpu"):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.tau = 1e-3
        self.gamma = 0.99
        
        self.network = DDQN_Online(state_size=self.state_size,
                            action_size=self.action_size).to(self.device)

        self.target_net = DDQN_Online(state_size=self.state_size,
                            action_size=self.action_size).to(self.device)
        
        self.optimizer = optim.Adam(params=self.network.parameters(), lr=1e-3)
        
    
    def get_action(self, state, epsilon):
        if random.random() > epsilon:
            # state = torch.from_numpy(state).float().unsqueeze(0).to(self.device).permute(0, -1, 1, 2)
            state = torch.from_numpy(state).float().unsqueeze(0).unsqueeze(0).to(self.device)
            self.network.eval()
            with torch.no_grad():
                action_values = self.network(state)
            self.network.train()
            action = np.argmax(action_values.cpu().data.numpy(), axis=1)
        else:
            action = random.choices(np.arange(self.action_size), k=1)
        return action
        
    def learn(self, experiences):
        self.optimizer.zero_grad()
        states, actions, rewards, next_states, dones = experiences
        # states, actions, rewards, next_states, dones = states.to(self.device).permute(0, -1, 1, 2).float(), actions.to(self.device).to(torch.int64).reshape(-1, 1), rewards.to(self.device).float().reshape(-1, 1), next_states.permute(0, -1, 1, 2).to(self.device).float(), dones.to(self.device).float().reshape(-1, 1)     
        states, actions, rewards, next_states, dones = states.to(self.device).unsqueeze(1).float(), actions.to(self.device).to(torch.int64).reshape(-1, 1), rewards.to(self.device).float().reshape(-1, 1), next_states.to(self.device).unsqueeze(1).float(), dones.to(self.device).float().reshape(-1, 1)     
        # print("states: ", states.shape)
        # print("dtype of state: ", states.dtype)
        # print("actions: ", actions.shape)
        # print("actions dtype: ", actions.dtype)
        # print("rewards: ", rewards.shape)
        # print("rewards dtype: ", rewards.dtype)
        # print("next_states: ", next_states.shape)
        # print("next_states dtype: ", next_states.dtype)
        # print("dones: ", dones.shape)
        # print("dones dtype: ", dones.dtype)
        with torch.no_grad():
            Q_targets_next = self.target_net(next_states).detach().max(1)[0].unsqueeze(1)
            Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        Q_a_s = self.network(states)
        Q_expected = Q_a_s.gather(1, actions)
        
        cql1_loss = torch.logsumexp(Q_a_s, dim=1).mean() - Q_a_s.mean()

        bellmann_error = F.mse_loss(Q_expected, Q_targets)
        
        q1_loss = cql1_loss + 0.5 * bellmann_error
        
        q1_loss.backward()
        clip_grad_norm_(self.network.parameters(), 1)
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.network, self.target_net)
        return q1_loss.detach().item(), cql1_loss.detach().item(), bellmann_error.detach().item()
        
        
    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)