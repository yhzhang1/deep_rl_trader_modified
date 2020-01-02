import numpy as np
from rl.core import Processor
from rl.util import WhiteningNormalizer
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from time import sleep

ADDITIONAL_STATE = 2
class NormalizerProcessor(Processor):
    def __init__(self):
        self.scaler = StandardScaler()
        self.normalizer = None

    def process_state_batch(self, batch):
        batch_len = batch.shape[0]
        k = []
        for i in range(batch_len):
            observe = batch[i][..., :-ADDITIONAL_STATE]
            #print('observe.shape: ', observe.shape)
            #print('observe: ', observe)
            #observe = self.scaler.fit_transform(observe)
            #print('observe: ', observe)
            agent_state = batch[i][..., -ADDITIONAL_STATE:]
            #print('agent_state: ', agent_state)
            temp = np.concatenate((observe, agent_state),axis=1)
            #print('temp: ', temp)
            temp = temp.reshape((1,) + temp.shape)
            #print('temp: ', temp)
            #sleep(10)
            k.append(temp)
        batch = np.concatenate(tuple(k))
        return batch
    
class DDPGProcessor(Processor):
    def process_action(self, action):
        action = np.clip(action[0], -1, 1)
        
        return action
        
