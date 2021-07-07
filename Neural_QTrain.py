import gym
import tensorflow as tf
import numpy as np
import random

# General Parameters
# -- DO NOT MODIFY --
ENV_NAME = 'CartPole-v0'
EPISODE = 200000  # Episode limitation
STEP = 200  # Step limitation in an episode
TEST = 10  # The number of tests to run every TEST_FREQUENCY episodes
TEST_FREQUENCY = 100  # Num episodes to run before visualizing test accuracy

# TODO: HyperParameters
GAMMA =  0.9# discount factor
INITIAL_EPSILON = 0.5 # starting value of epsilon
FINAL_EPSILON =  0.01# final value of epsilon
EPSILON_DECAY_STEPS =  100# decay period
REPLAY_SIZE = 2000
BATCH_SIZE = 32


# Create environment
# -- DO NOT MODIFY --
env = gym.make(ENV_NAME)
epsilon = INITIAL_EPSILON
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.n

# Placeholders
# -- DO NOT MODIFY --
state_in = tf.placeholder("float", [None, STATE_DIM])
action_in = tf.placeholder("float", [None, ACTION_DIM])
target_in = tf.placeholder("float", [None])



# -- DO NOT MODIFY ---
def explore(state, epsilon):
    """
    Exploration function: given a state and an epsilon value,
    and assuming the network has already been defined, decide which action to
    take using e-greedy exploration based on the current q-value estimates.
    """
    Q_estimates = q_values.eval(feed_dict={
        state_in: [state]
    })
    if random.random() <= epsilon:
        action = random.randint(0, ACTION_DIM - 1)
    else:
        action = np.argmax(Q_estimates)
    one_hot_action = np.zeros(ACTION_DIM)
    one_hot_action[action] = 1
    return one_hot_action






# TODO: Define Network Graph

hidden_unit = 128

w1 = tf.Variable(tf.random_normal(shape=[STATE_DIM, hidden_unit], mean=0,stddev=0.3))
b1 = tf.Variable(tf.zeros([hidden_unit]))

w2 = tf.Variable(tf.random_normal(shape=[hidden_unit, hidden_unit], mean=0,stddev=0.3))
b2 = tf.Variable(tf.zeros([hidden_unit]))

w3 = tf.Variable(tf.random_normal(shape=[hidden_unit, ACTION_DIM], mean=0, stddev=0.3))
b3 = tf.Variable(tf.zeros([ACTION_DIM]))

layer1 = tf.nn.relu(tf.matmul(state_in, w1) + b1)

layer2 = tf.nn.relu(tf.matmul(layer1, w2) + b2)

q_values = tf.matmul(layer2, w3) + b3
q_action = tf.reduce_sum(tf.multiply(q_values, action_in), reduction_indices=1)




loss = tf.reduce_sum(tf.square(target_in - q_action))
optimizer = tf.train.AdamOptimizer().minimize(loss)






# Start session - Tensorflow housekeeping
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())

replay_buffer=[]


# Main learning loop
for episode in range(EPISODE):

    # initialize task
	state = env.reset()
    # Update epsilon once per episode
	epsilon -= epsilon / EPSILON_DECAY_STEPS


    # Move through env according to e-greedy policy
	for step in range(STEP):

		action = explore(state, epsilon)

		next_state, reward, done, _ = env.step(np.argmax(action))

		if reward == 1:
			replay_buffer.append([state, action, reward, next_state, done])
			if len(replay_buffer) > REPLAY_SIZE:
				replay_buffer.pop(REPLAY_SIZE//2) 
		else:
			replay_buffer.insert(0,[state, action, reward, next_state, done])
			if len(replay_buffer) > REPLAY_SIZE:
				replay_buffer.pop(0)


		if len(replay_buffer) > BATCH_SIZE:
			batch = random.sample(replay_buffer, BATCH_SIZE)
			state_batch = [data[0] for data in batch]
			action_batch = [data[1] for data in batch]
			reward_batch = [data[2] for data in batch]
			next_state_batch = [data[3] for data in batch]

			target = []
			nextstate_q_values = q_values.eval(feed_dict={
				state_in: next_state_batch
			})
			for i in range(0, BATCH_SIZE):
				done_batch = batch[i][4]
				if done_batch:
					target.append(reward_batch[i])
				else:
					target_val = reward_batch[i] + GAMMA * np.max(nextstate_q_values[i])
					target.append(target_val)
			session.run([optimizer], feed_dict={
				target_in: target,
				action_in: action_batch,
				state_in: state_batch
			})

        # Update
		state = next_state
		if done:
			break

    # Test and view sample runs - can disable render to save time
    # -- DO NOT MODIFY --
	if (episode % TEST_FREQUENCY == 0 and episode != 0):
		total_reward = 0
		for i in range(TEST):
			state = env.reset()
			for j in range(STEP):
				env.render()
				action = np.argmax(q_values.eval(feed_dict={
				    state_in: [state]
				}))
				state, reward, done, _ = env.step(action)
				total_reward += reward
				if done:
					break
		ave_reward = total_reward / TEST
		print('episode:', episode, 'epsilon:', epsilon, 'Evaluation '
														'Average Reward:', ave_reward)

env.close()
