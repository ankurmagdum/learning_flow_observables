import numpy as np
import os

hyp = np.loadtxt('hyperparameters.txt')

total = 2*hyp[0]*hyp[1]*hyp[2]

done = len(os.listdir('./predictions/'))

print(100*done/total, '% done')
