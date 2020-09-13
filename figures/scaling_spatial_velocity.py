from pydrake.all import (PiecewiseQuaternionSlerp, RandomGenerator,
                         UniformlyRandomQuaternion)

g = RandomGenerator()
q1 = UniformlyRandomQuaternion(g)
q2 = UniformlyRandomQuaternion(g)

alpha = 0.5
s1 = PiecewiseQuaternionSlerp([0, 1], [q1, q2])
s2 = PiecewiseQuaternionSlerp([0, 1 / alpha], [q1, q2])
print(s1.EvalDerivative(0.5))
print(s2.EvalDerivative(0.5 / alpha) / alpha)
