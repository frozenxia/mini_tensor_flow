import numpy as np

import random
import math


def sigmoid(x):
    return 1.0 /(1+np.exp(-x))

def sigmoid_derivative(x):
    return (1.0-x)*x

def tanh_derivative(x):
    return 1.0-x**2
## uniform random in [a,b] with shape *args
def random_arr(a,b,*args):
    return np.random.rand(*args)*(b-a)+a


class LstmParams(object):
    def __init__(self,hidden_units,x_dim):
        self.hidden_units = hidden_units
        self.x_dim = x_dim
        concat_dim = hidden_units + x_dim

        ## weight matrices

        self.wi = random_arr(-1,1,hidden_units,concat_dim)
        self.wf = random_arr(-1,1,hidden_units,concat_dim)
        self.wo = random_arr(-1,1,hidden_units,concat_dim)
        self.wg = random_arr(-1,1,hidden_units,concat_dim)


        ## bias terms

        self.bi = random_arr(-1,1,hidden_units)
        self.bf = random_arr(-1,1,hidden_units)
        self.bo = random_arr(-1,1,hidden_units)
        self.bg = random_arr(-1,1,hidden_units)


        ## diffs (derivatives of loss functions)

        self.wi_diffs = np.zeros((hidden_units,concat_dim))
        self.wf_diffs = np.zeros((hidden_units,concat_dim))
        self.wo_diffs = np.zeros((hidden_units,concat_dim))
        self.wg_diffs = np.zeros((hidden_units,concat_dim))

        self.bi_diffs = np.zeros(hidden_units)
        self.bf_diffs = np.zeros(hidden_units)
        self.bo_diffs = np.zeros(hidden_units)
        self.bg_diffs = np.zeros(hidden_units)


    def apply_diff(self,lr = 0.01):

        self.wi -= self.wi_diffs*lr
        self.wf -= self.wf_diffs*lr
        self.wo -= self.wo_diffs*lr
        self.wg -= self.wg_diffs*lr

        self.bi -= self.bi_diffs*lr
        self.bf -= self.bf_diffs*lr
        self.bo -= self.bo_diffs*lr
        self.bg -= self.bg_diffs*lr

        self.wi_diffs = np.zeros_like(self.wi)
        self.wf_diffs = np.zeros_like(self.wf)
        self.wo_diffs = np.zeros_like(self.wo)
        self.wg_diffs = np.zeros_like(self.wg)


        self.bi_diffs = np.zeros_like(self.bi)
        self.bf_diffs = np.zeros_like(self.bf)
        self.bo_diffs = np.zeros_like(self.bo)
        self.bg_diffs = np.zeros_like(self.bg)

class LstmState(object):
    def __init__(self,hidden_units):
        self.hidden_units = hidden_units

        self.i = np.zeros(hidden_units)
        self.f = np.zeros(hidden_units)
        self.o = np.zeros(hidden_units)
        self.g = np.zeros(hidden_units)
        self.s = np.zeros(hidden_units)
        self.h = np.zeros(hidden_units)

        self.bottom_diff_h = np.zeros_like(self.h)
        self.bottom_diff_s = np.zeros_like(self.s)


class LstmNode(object):
    def __init__(self,lstm_state,lstm_params):

        ## store params
        self.state = lstm_state
        self.params = lstm_params

        ## 
        self.xc = None

    # forward propagation 
    def bottom_data_is(self,input_x,s_prev=None,h_prev=None):

        if s_prev is None:
            s_prev = np.zeros_like(self.state.s)
        if h_prev is None:
            h_prev = np.zeros_like(self.state.h)

        self.s_prev = s_prev
        self.h_prev = h_prev

        xc = np.hstack((input_x,h_prev))

        self.state.i = sigmoid(np.dot(self.params.wi,xc) + self.params.bi)
        self.state.f = sigmoid(np.dot(self.params.wf,xc)+ self.params.bf)
        self.state.o = sigmoid(np.dot(self.params.wo,xc) + self.params.bo)
        self.state.g = np.tanh(np.dot(self.params.wg,xc)+ self.params.bg)

        self.state.s = self.state.g *self.state.i + self.s_prev * self.state.f
        self.state.h = self.state.s * self.state.o

        self.xc  = xc
    

    # back propagation
    def top_data_is(self,top_diff_h,top_diff_s):
        
        ## ds =  
        # ds = self.state.o*top_diff_h + top_diff_s
        ds = self.state.o*top_diff_h 
        do = self.state.s*top_diff_h

        di = self.state.g*ds
        dg = self.state.i*ds
        df = self.s_prev*ds
        
        di_input = sigmoid_derivative(self.state.i)*di
        df_input = sigmoid_derivative(self.state.f)*df
        do_input = sigmoid_derivative(self.state.o)*do
        dg_input = tanh_derivative(self.state.g)*dg

        ## wi_diff

        self.params.wi_diffs += np.outer(di_input,self.xc)
        self.params.wf_diffs += np.outer(df_input,self.xc)
        self.params.wo_diffs += np.outer(do_input,self.xc)
        self.params.wg_diffs += np.outer(dg_input,self.xc)


        self.params.bi_diffs += di_input
        self.params.bf_diffs += df_input
        self.params.bo_diffs += do_input
        self.params.bg_diffs += dg_input

        dxc  = np.zeros_like(self.xc)

        dxc += np.dot(self.params.wi.T,di_input)
        dxc += np.dot(self.params.wf.T,df_input)
        dxc += np.dot(self.params.wo.T,do_input)
        dxc += np.dot(self.params.wg.T,dg_input)


        self.state.bottom_diff_s = ds*self.state.f
        self.state.bottom_diff_h = dxc[self.params.x_dim:]
    
class LstmLayers():
    def __init__(self,lstm_parm):
        self.lstm_parm = lstm_parm
        self.lstm_node_list = []
        self.x_list =[]

    def x_list_clear(self):
        self.x_list =[]
    
    def x_list_add(self,x):
        self.x_list.append(x)

        if len(self.x_list) > len(self.lstm_node_list):
            lstm_state = LstmState(self.lstm_parm.hidden_units)
            self.lstm_node_list.append(LstmNode(lstm_state,self.lstm_parm))
        
        idx = len(self.x_list) -1

        if idx == 0:
            self.lstm_node_list[idx].bottom_data_is(x)
        else:
            s_prev = self.lstm_node_list[idx-1].state.s
            h_prev = self.lstm_node_list[idx-1].state.h
            self.lstm_node_list[idx].bottom_data_is(x,s_prev=s_prev,h_prev=h_prev)
    
    def y_list_is(self,y_list,loss_layer):
        assert(len(y_list) == len(self.x_list))

        idx = len(y_list) -1 

        loss = loss_layer.loss(self.lstm_node_list[idx].state.h,y_list[idx])
        diff_h = loss_layer.bottom_diff(self.lstm_node_list[idx].state.h,y_list[idx])

        diff_s = np.zeros(self.lstm_parm.hidden_units)

        self.lstm_node_list[idx].top_data_is(diff_h,diff_s)

        idx -= 1

        while idx >=0:
            loss += loss_layer.loss(self.lstm_node_list[idx].state.h,y_list[idx])
            diff_h = loss_layer.bottom_diff(self.lstm_node_list[idx].state.h,y_list[idx])
            diff_h += self.lstm_node_list[idx+1].state.bottom_diff_h
            diff_s += self.lstm_node_list[idx+1].state.bottom_diff_s

            self.lstm_node_list[idx].top_data_is(diff_h,diff_s)

            idx -=1

        return loss

        



        

    



