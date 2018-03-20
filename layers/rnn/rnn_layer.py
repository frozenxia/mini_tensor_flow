import tensorflow as tf

class RnnModel():
    def __init__(self,batch_size,num_steps,num_class,state_size,learning_rate):
        self.input_x = tf.placeholder(dtype=tf.int32,shape=(batch_size,num_steps),name='input_x')
        self.input_y = tf.placeholder(dtype=tf.int32,shape=(batch_size,num_steps),name='input_y')
        self.init_state = tf.zeros([batch_size,state_size],tf.float32)
        
        print(self.input_x,self.input_y)

        # batch,x_length,x_dim
        x_one_hot = tf.one_hot(self.input_x,num_class)

        # batch,x_length,x_dim
        rnn_inputs = tf.unstack(x_one_hot,axis=1)
        # print(rnn_inputs)
        with tf.variable_scope('rnn_cell'):
            W = tf.get_variable('W',dtype=tf.float32,shape=[num_class+state_size,state_size])
            b = tf.get_variable('b',dtype=tf.float32,shape=[state_size],initializer=tf.constant_initializer(0))

        
        # print('rnn types \n\n')
        # print(self.init_state)
        def rnn_cell(rnn_input,state):
            # print(rnn_input,'\n')
            # print(state,'\n')
            with tf.variable_scope('rnn_cell', reuse=True):
                W = tf.get_variable('W',dtype=tf.float32,shape=[num_class+state_size,state_size])
                b = tf.get_variable('b',dtype=tf.float32,shape=[state_size],initializer=tf.constant_initializer(0))
            
            return tf.tanh(tf.matmul(tf.concat([rnn_input, state], 1),W)+b)
            # return 
        
        state = self.init_state
        rnn_outputs = []
        for rnn_input in rnn_inputs:
            state = rnn_cell(rnn_input,state)
            rnn_outputs.append(state)
        self.final_state = rnn_outputs[-1]


        with tf.variable_scope('softmax'):
            W = tf.get_variable('W',dtype=tf.float32,shape=[state_size,num_class])
            b = tf.get_variable('b',dtype=tf.float32,shape=[num_class],initializer=tf.constant_initializer(0))
            logits = [tf.matmul(rnn_output,W)+b for rnn_output in rnn_outputs]
        self.predictons = [tf.nn.softmax(logit) for logit in logits]

        self.y_as_lsit = tf.unstack(self.input_y,num=num_steps,axis=1)
        print('y_as_list \n\n')
        # print(y_as_lsit)
        self.losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label,logits=logit) for label,logit in zip(self.y_as_lsit,logits)]
        self.total_loss = tf.reduce_mean(self.losses)

        self.train_step = tf.train.AdagradOptimizer(learning_rate).minimize(self.total_loss)


        



        



