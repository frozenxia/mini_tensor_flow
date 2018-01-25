class Op(object):
    def __init__(self,name='op'):
        self._name = name

    
    def __sub__(self,other):
        raise NotImplementedError()
        # return self._value- other
    @abstractmethod
    def forward(self):
        raise NotImplementedError
    @abstractmethod
    def grad(self):
        raise NotImplementedError

class ConstanceOp(Op):
    def __init__(self,input,name='constance_op'):
        super(ConstanceOp,self).__init__(name)
        self._value = input
    


class AddOp(Op):
    def __init__(self,input1,input2,name='add_op'):
        super(AddOp,self).__init__(name)
        op1 = input1
        if not isinstance(input1,Op):
            op1 = ConstanceOp(input1)
        op2 = input2
        if not isinstance(input2,Op):
            op2 = ConstanceOp(input2)
        self._op1 = op1
        self._op2 = op2
    
    

if __name__ == '__main__':
    # op1 = Op(3)
    # print(op1-2)