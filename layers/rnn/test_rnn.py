import numpy as np
import tensorflow as tf 
from rnn_layer import RnnModel

# import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

num_steps = 5
batch_size = 200
num_class = 2
state_size = 4
learning_rate=0.1

def gen_data(size=10000000):
    X= np.array(np.random.choice(2,size=(size,)))
    # print(X)
    Y= []
    for i in range(size):
        threshold = 0.5
        if X[i-3] == 1:
            threshold += 0.5
        if X[i-8] == 1:
            threshold -= 0.25
        
        if np.random.rand() > threshold:
            Y.append(0)
        else:
            Y.append(1)
    return X,Y

def gen_batch(raw_data,batch_size,num_steps):
    raw_x,raw_y = raw_data
    data_len = len(raw_x)
    # print(data_len)

    batch_partiton_length = int(data_len/batch_size)
    data_x = np.zeros(shape=(batch_size,batch_partiton_length),dtype=np.int32)
    data_y = np.zeros(shape=(batch_size,batch_partiton_length),dtype=np.int32)

    for i in range(batch_size):
        data_x[i] = raw_x[batch_partiton_length*i:batch_partiton_length*(i+1)]
        data_y[i] = raw_y[batch_partiton_length*i:batch_partiton_length*(i+1)]
    
    epoch_size = int(batch_partiton_length/num_steps)
    # print(epoch_size)
    for i in range(epoch_size):
        yield (data_x[:,num_steps*i:num_steps*(i+1)],data_y[:,num_steps*i:num_steps*(i+1)])


def gen_epoches (n,num_steps):
    for i in range(n):
        yield gen_batch(gen_data(),batch_size,num_steps)

def train_network(num_epochs,num_steps,state_size=4,verbos=True):
    rnn_model = RnnModel(batch_size,num_steps,num_class,state_size,learning_rate)
    train_losses=[]
    with  tf.Session() as session:
        session.run(tf.global_variables_initializer())
        
        for idx ,epoch in enumerate(gen_epoches(num_epochs,num_steps)):
            training_loss = 0
            training_state = np.zeros((batch_size,state_size))
            if verbos:
                print('\nEPOCH',idx)
            for step,(X,Y) in enumerate(epoch):
                # print(X.shape)
                # print(step)
                y_as_list,tr_loss,training_loss_,training_state,_=session.run([rnn_model.y_as_lsit,rnn_model.losses,rnn_model.total_loss,rnn_model.final_state,rnn_model.train_step],feed_dict={rnn_model.input_x:X,rnn_model.input_y:Y,rnn_model.init_state:training_state})
                training_loss += training_loss_
                # print(y_as_list,'\n\n')
                if step %100 == 0 and step > 0:
                    if verbos:
                        print('average loss at step ',step ,'  ' ,training_loss_ ,' for last 100 steps ',training_loss/100)
                    train_losses.append(training_loss)
                    training_loss = 0.0
                # break
    return train_losses

        # pass

# print(gen_data(10))
training_loss = train_network(1,num_steps)
plt.plot(training_loss)
plt.show()


