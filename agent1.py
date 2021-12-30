import numpy as np
from Env import CustomEnv

class LogisticPolicy:
	def __init__(self, θ, α, γ):
		# Initialize paramters θ, learning rate α and discount factor γ
		self.θ = np.reshape(θ,(500,1))
		self.α = α
		self.γ = γ
	def logistic(self, y):
		# definition of logistic function
		return 1/(1 + np.exp(-y))
	def probs(self, x):
		# returns probabilities of two actions
		x=np.reshape(x, (1, 500))
		y = x @ self.θ
		prob0 = self.logistic(y)
		return np.array([prob0, 1-prob0])
	def act(self, x):
		# sample an action in proportion to probabilities
		probs = self.probs(x)
		probs=np.reshape(probs,(1,2))
		probs=probs[0]
		action = np.random.choice([0, 1], p=probs)
		return action, probs[action]
	def grad_log_p(self, x):
		# calculate grad-log-probs
		x=np.reshape(x, (1, 500))
		y = x @ self.θ
		grad_log_p0 = x - x*self.logistic(y)
		grad_log_p1 = - x*self.logistic(y)
		return grad_log_p0, grad_log_p1
	def grad_log_p_dot_rewards(self, grad_log_p, actions, discounted_rewards):
		# dot grads with future rewards for each action in episode
		return grad_log_p.T @ discounted_rewards

	def discount_rewards(self, rewards):
		# calculate temporally adjusted, discounted rewards
		discounted_rewards = np.zeros(len(rewards))
		cumulative_rewards = 0
		for i in reversed(range(0, len(rewards))):
			cumulative_rewards = cumulative_rewards * self.γ + rewards[i]
			discounted_rewards[i] = cumulative_rewards
		return discounted_rewards

	def update(self, rewards, obs, actions):
		# calculate gradients for each action over all observations
		grad_log_p = np.array([self.grad_log_p(ob)[action] for ob,action in zip(obs,actions)])
		#assert grad_log_p.shape == (len(obs), 500)
		# calculate temporaly adjusted, discounted rewards
		discounted_rewards = self.discount_rewards(rewards)
		# gradients times rewards
		dot = self.grad_log_p_dot_rewards(grad_log_p, actions, discounted_rewards)
		# gradient ascent on parameters
		self.θ += self.α*dot

import statistics

def scale(observation):
	templist=[]
	for j in range(50):
		templist.append(observation[j][0])
	mean=statistics.mean(templist)
	stdev=statistics.stdev(templist)
	for k in range(4):
		for j in range(50):
			observation[j][k]=(observation[j][k]-mean/stdev)
	templist=[]
	for j in range(50):
		templist.append(observation[j][4])
	mean=statistics.mean(templist)
	stdev=statistics.stdev(templist)
	for j in range(50):
		observation[j][4]=(observation[j][4]-mean/stdev)
	return observation
	

def run_episode(env, policy, render=True):
	observation = env.reset()
	totalreward = 0
	observations = []
	actions = []
	rewards = []
	probs = []
	done = False
	for i in range(len(env.df)-100):
		if render:
			env.render()
		if done:
			print("done")
			break;
		observation=scale(observation)
		observations.append(observation)
		action, prob = policy.act(observation)
		observation, reward, done= env.step(action+1)
		totalreward += reward
		if i%10000==0:
			print("i",i," totalreward",totalreward)
		rewards.append(reward)
		actions.append(action)
		probs.append(prob)
	return totalreward, np.array(rewards), np.array(observations), np.array(actions), np.array(probs)
	
def train(θ, α, γ, Policy, learn_env,valid_env, MAX_EPISODES=1000, evaluate=False):
	# initialize environment and policy
	episode_rewards = []
	policy = Policy(θ,α,γ)
	# train until MAX_EPISODES
	for i in range(MAX_EPISODES):
		# run a single episode
		total_reward,rewards,observations,actions,probs=run_episode(learn_env,policy)
		episode_rewards.append(total_reward)
		policy.update(rewards, observations, actions)
		print("learning_EP: " + str(i) + " Score: " + str(total_reward))
	if evaluate:
		for i in range(MAX_EPISODES):
			# run a single episode
			total_reward,rewards,observations,actions,probs=run_episode(valid_env,policy)
			episode_rewards.append(total_reward)
			print("validing_EP: " + str(i) + " Score: " + str(total_reward))
	return episode_rewards, policy