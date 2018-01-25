from abc import ABCMeta, abstractmethod
import sys
import numpy
## mini tensor flow toy
mtf = sys.modules['numpy']


class Op(object):
    def __init__(self,name='op'):
        self._name = name
    
    def get_name (self):
        return self._name

    def set_name (self,name):
        self._name = name

    @abstractmethod
    def forward(self):
        raise NotImplementedError

    def grad(self):
        raise NotImplementedError

    def __add__(self,other):
        return AddOp(self,other)


    # def __repr__(self):



# class SimpleOp(Op):
#     def __init__(self,name="SimpleOp"):
#         super(SimpleOp,self).__init__(name)

#     def __add__(self,other):
#         raise NotImplementedError


class ConstantOp(Op):
    """ ConstantOp contains a initial value """
    def __init__(self,value,name="constant"):
        super(ConstantOp,self).__init__(name)
        self._value = value
    
    def forward(self):
        return self._value

    def grad(self):
        return 0


class AddOp(Op):
    def __init__(self,input1,input2,name="AddOp"):
        super(AddOp,self).__init__(name='op')
        if  not isinstance(input1,Op):
            self._input1 = ConstantOp(input1,name="input1")
        else:
            self._input1 = input1
        
        if  not isinstance(input2,Op):
            self._input2 = ConstantOp(input2,name="input2")
        else:
            self._input2 = input2

    def forward(self):
        return mtf.add(self._input1.forward() , self._input2.forward())

    def grad(self):
        return self._input1.grad() + self._input2.grad()

class MulOp(Op):
    def __init__(self,input1,input2,name="AddOp"):
        super(MulOp,self).__init__(name='op')
        if  not isinstance(input1,Op):
            self._opv1 = ConstantOp(input1,name="input1")
        else:
            self._opv1 = input1
        
        if  not isinstance(input2,Op):
            self._opv2 = ConstantOp(input2,name="input2")
        else:
            self._opv2 = input2
    
    def forward(self):
        return mtf.multiply(self._opv1.forward(),self._opv2.forward())

    def grad(self):
        pass


class ReluOp(Op):
    def __init__(self,input,name="ReluOp"):
        super(ReluOp,self).__init__(name)
        # print(input)
        if not isinstance(input,Op):
            self._opv = ConstantOp(input)
        else :
            self._opv = input
    def forward(self):
        ## x = max(x,0)
        return mtf.maximum(self._opv.forward(),0)
    def grad(self):
        return mtf.sign(self._opv.grad())



    