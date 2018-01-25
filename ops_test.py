from ops import *

constant_op = ConstantOp(-2)
constant_op2 = ConstantOp(-5)



add_op = MulOp(constant_op,constant_op2)

relu_op = ReluOp(add_op)
print(relu_op.forward())


# print(type(constant_op.forward()))
# print(type(relu_op.forward()))