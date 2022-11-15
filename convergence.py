from cmath import log
# 2nd order 51: 0.013974884880006576, 101: 0.003591007183831516, 201: 0.0009070651597894106, 301:0.0005867045889719425
# 2nd order convergence rate: 1.9886422352550597, 1.9994259151414708, 1.0789674870607797
# 4th orders 51: 0.00044174374737235674, 101:3.641415337356893e-05, 201:4.874176969383955e-06, 301:3.678728587179491e-06
# 4 order convergence rate: 3.652555414768546, 2.9221910439582803, 0.6968309332009133
# 6th order 51: 0.00019848222953609774, 101:4.203421063791327e-06, 201:2.854096053899836e-06, 301:2.8405286915042516e-06
# 6th order convergence rate: 5.641489390958262, 0.5625578889846542, 0.01180019152228357

q = log(2.854096053899836e-06/2.8405286915042516e-06)/log(301/201)
print('Convergence rate: ', q)
