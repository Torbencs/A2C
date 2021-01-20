import gym
import tensorflow as tf
from networks import ActorCriticNetwork
from actor_critic import Agent
import tensorflow_probability as tfp


env = gym.make('CartPole-v0')

model = tf.saved_model.load('export')
done = False

def choose_action(observation):
    state = tf.convert_to_tensor([observation], dtype=tf.float32)
    _, probs = model(state)

    action_probabilities = tfp.distributions.Categorical(probs=probs)
    action = action_probabilities.sample()
    
    return action.numpy()[0]



for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        action = choose_action(observation)
        observation, reward, done, info = env.step(action) 
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()
