import numpy as np
from lstm_layers import LstmParams,LstmLayers
class ToyLayers():
    @classmethod
    def loss(self,pred,label):
        return (pred[0]-label)**2
    @classmethod
    def bottom_diff(self,pred,label):
        diff = np.zeros_like(pred)
        ## implicity there exist a layer , w.T*pred, and w =[1,0,0,0], for simplicity we do it mannualy
        diff[0]= 2*(pred[0]-label)
        return diff
    
def train():
    # np.random.seed(0)
    hidden_units = 128
    x_dim = 50
    lstm_param = LstmParams(hidden_units,x_dim)
    lstm_net = LstmLayers(lstm_param)
    y_list = [-0.5,0.8,0.1,-0.5]
    input_val_arr = [np.random.random(x_dim) for _ in y_list]

    for curr_iter in range(100000):
        print('iter %2s '%(str(curr_iter)),end=':' )
        for ind in range(len(y_list)):
            lstm_net.x_list_add(input_val_arr[ind])

        print('y_pred =[ '+ ','.join(["%2.5f" %(lstm_net.lstm_node_list[ind].state.h[0]) for ind in range(len(y_list)) ]) + ']' ,end=',')

        loss = lstm_net.y_list_is(y_list,ToyLayers)
        print("loss:", "%.3e" % loss)

        lstm_net.lstm_parm.apply_diff(lr=0.001)
        lstm_net.x_list_clear()

if __name__ == '__main__':
    train()
