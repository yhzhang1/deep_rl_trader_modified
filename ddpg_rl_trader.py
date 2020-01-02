import numpy as np

# import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, CuDNNLSTM, Input, Concatenate
from keras.optimizers import Adam

# keras-rl agent
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory

# trader environment
from TraderEnv import OhlcvEnv
from TraderEnv import Oca1a1vb1b1vEnv
from TraderEnv import OcNewActionSpaceEnv
from TraderEnv import DDPGEnv
# custom normalizer
from util import NormalizerProcessor
from util import DDPGProcessor

import pandas as pd
import keras
import os
#--------------------customer import
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
import tensorflow as tf
import keras
import datetime

def set_gpu_option():
    os.environ["CUDA_VISIBLE_DEVICES"]="1" 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    keras.backend.tensorflow_backend.set_session(sess)
    
def create_model(shape, nb_actions):
    model = Sequential()
    model.add(CuDNNLSTM(64, input_shape=shape, return_sequences=True))
    model.add(CuDNNLSTM(64))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions, activation='linear'))
    
    return model

def create_actor(input_shape, nb_actions):
    
    actor = Sequential()
    actor.add(CuDNNLSTM(64, input_shape=input_shape, return_sequences=True))
    actor.add(CuDNNLSTM(32, return_sequences=False))
    actor.add(Dense(16, activation='relu'))
    actor.add(Dense(nb_actions, activation='tanh'))
    actor.summary()
    
    return actor

def create_critic(action_input, observation_input):
    '''
    flattened_observation = Flatten()(observation_input)
    x = Concatenate()([action_input, flattened_observation])
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x = Dense(1)(x)
    x = Activation('linear')(x)
    critic = Model(inputs=[action_input, observation_input], outputs=x)
    print(critic.summary())    
    '''
    
    x = CuDNNLSTM(64, return_sequences=True)(observation_input)
    x = CuDNNLSTM(32, return_sequences=False)(x)
    x = Concatenate()([action_input, x])
    x = Dense(16, activation='relu')(x)
    x = Dense(1, activation='linear')(x)
    critic = Model(inputs=[action_input, observation_input], outputs=x)
    critic.summary()
    
    return critic


def main():
    set_gpu_option()
    # OPTIONS
    ENV_NAME = 'DDPGEnv-v0'
    TIME_STEP = 30

    # Get the environment and extract the number of actions.

    PATH_TRAIN = '/home/data/training_x_150.h5'
    PATH_TEST = '/home/data/test_x_150.h5'
    """
    env = OhlcvEnv(TIME_STEP, path=PATH_TRAIN)
    env_test = OhlcvEnv(TIME_STEP, path=PATH_TEST)
    """
    store = pd.HDFStore(PATH_TRAIN, mode='r')
    varieties_list = store.keys()
    print('varieties_list: ', varieties_list)
    print('num varieties: ', len(varieties_list))
    
    variety = 'RB'
    print('variety: ', variety)
    
    # get selected features
    SELECTED_FACTOR_PATH = '~/feature_selection/根据互信息选出的特征，根据重要性排序.csv'
    selected_factor_df = pd.read_csv(SELECTED_FACTOR_PATH, index_col=0)
    selected_factor_list = selected_factor_df[variety].to_list()
    
    env = DDPGEnv(TIME_STEP, variety=variety, path=PATH_TRAIN, selected_factor_list=selected_factor_list)
    #env_test = DDPGEnv(TIME_STEP, variety=variety, path=PATH_TEST,  selected_factor_list=selected_factor_list)

    # random seed
    np.random.seed(123)
    env.seed(123)

    nb_actions = env.action_space.shape[0]
    print('nb_actions: ', nb_actions)

    print('env.observation_space.shape: ', env.observation_space.shape)
    print('env.observation_space: ', env.observation_space)
    
    # create actor
    actor = create_actor(input_shape=env.shape, nb_actions=nb_actions)
    
    # create critic
    action_input = Input(shape=(nb_actions,), name='action_input')
    observation_input = Input(shape=env.shape, name='observation_input')
    critic = create_critic(action_input, observation_input)
    


    # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and even the metrics!
    memory = SequentialMemory(limit=50000, window_length=TIME_STEP)

    random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.3)
    ddpg = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                  memory=memory, nb_steps_warmup_critic=1000, nb_steps_warmup_actor=1000,
                  random_process=random_process, gamma=.99, target_model_update=1e-3, processor=DDPGProcessor())
    ddpg.compile(optimizer=Adam(lr=1e-3), metrics=['mae'])

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_grads=True)
    for _ in range(3):
        ddpg.fit(env, nb_steps=140000, nb_max_episode_steps=140000, visualize=False, verbose=2)

    """
        try:
            # validate
            info = dqn.test(env_test, nb_episodes=1, visualize=False)
            n_long, n_short, total_reward, portfolio = info['n_trades']['long'], info['n_trades']['short'], info[
                'total_reward'], int(info['portfolio'])
            np.array([info]).dump(
                './info/duel_dqn_{0}_weights_{1}LS_{2}_{3}_{4}.info'.format(ENV_NAME, portfolio, n_long, n_short,
                                                                     total_reward))
            print('info saved')
            dqn.save_weights(
                './model/duel_dqn_{0}_weights_{1}LS_{2}_{3}_{4}.h5f'.format(ENV_NAME, portfolio, n_long, n_short, total_reward),
                overwrite=True)
            print('weight saved')
        except KeyboardInterrupt:
            continue
        """

if __name__ == '__main__':
    main()