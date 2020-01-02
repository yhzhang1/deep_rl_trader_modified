import numpy as np

# import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, CuDNNLSTM, Reshape, Conv2D, Dropout
from keras.optimizers import Adam

# keras-rl agent
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory

# trader environment
from TraderEnv import OhlcvEnv
from TraderEnv import Oca1a1vb1b1vEnv
from TraderEnv import OcNewActionSpaceEnv
# custom normalizer
from util import NormalizerProcessor

import pandas as pd
import keras
import tensorflow as tf
import os
from keras.callbacks import TensorBoard

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

def create_model2(shape, nb_actions):
    model = Sequential()
    model.add(Reshape((1,-1), input_shape=shape))
    model.add(CuDNNLSTM(64, return_sequences=True))
    model.add(CuDNNLSTM(64))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions, activation='linear'))
    
    return model

def create_cnn_model(shape, nb_actions):
    model = Sequential()
    model.add(Reshape(shape+(1,), input_shape=shape))
    model.add(Conv2D(64, kernel_size=(20,int(shape[1]/2)), padding='same', activation='relu', strides=(2,1)))
    model.add(Conv2D(32, kernel_size=(20,int(shape[1]/2)), padding='same', activation='relu', strides=(2,1)))
    model.add(Flatten())
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions, activation='linear'))
    
    return model

def create_dense_model(shape, nb_actions):
    model = Sequential()
    model.add(Reshape((-1,), input_shape=shape))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(nb_actions, activation='linear'))

    return model

def main():
    # OPTIONS
    ENV_NAME = 'OcNewActionSpaceEnv-v0'
    TIME_STEP = 100
    set_gpu_option()
    # Get the environment and extract the number of actions.
    '''
    PATH_TRAIN = "./data/train/"
    PATH_TEST = "./data/test/"
    '''
    PATH_TRAIN = '/home/data/training_x_150.h5'
    PATH_TEST = '/home/data/test_x_150.h5'
    """
    env = OhlcvEnv(TIME_STEP, path=PATH_TRAIN)
    env_test = OhlcvEnv(TIME_STEP, path=PATH_TEST)
    """
    store = pd.HDFStore(PATH_TRAIN, mode='r')
    varieties_list = store.keys()
    variety = 'I'
    print('variety: ', variety)
    env = OcNewActionSpaceEnv(TIME_STEP, variety=variety, path=PATH_TRAIN)
    env_test = OcNewActionSpaceEnv(TIME_STEP, variety=variety, path=PATH_TEST)

    # random seed
    np.random.seed(123)
    env.seed(123)

    nb_actions = env.action_space.n
    print('nb_actions: ', nb_actions)
    print('env.shape: ', env.shape)
    model = create_model(shape=env.shape, nb_actions=nb_actions)
    print(model.summary())

    # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and even the metrics!
    memory = SequentialMemory(limit=50000, window_length=TIME_STEP)
    # policy = BoltzmannQPolicy()
    policy = EpsGreedyQPolicy()
    # enable the dueling network
    # you can specify the dueling_type to one of {'avg','max','naive'}
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=200,
                   enable_dueling_network=True, dueling_type='avg', target_model_update=1e-2, policy=policy,
                   processor=NormalizerProcessor())
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])
    
    tbCallBack = TensorBoard(histogram_freq=0, write_grads=True, write_images=True)

    while True:
        # train
        '''
        for e in range(500):
            print('epoch: {}'.format(e))
            if os.path.isfile('weights'):
                print('weight file exist')
                print('load weights')
                dqn.load_weights('weights')
            else:
                print('weight file does not exist')
        '''
        dqn.fit(env, nb_steps=70000, nb_max_episode_steps=None, visualize=False, verbose=2, callbacks=[tbCallBack])
            #dqn.save_weights('weights', overwrite=True)
        #print('fit: ', fit)

        
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
        

if __name__ == '__main__':
    main()