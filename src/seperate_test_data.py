import numpy as np
import os
import logging

logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
import random

def seperate_test_data(parameters, lifts, drags, sizes, reps, test_fraction = 1/2):
    
    random.seed(1)

    N = len(parameters)
    a = int(test_fraction * N)
    indices = np.array(range(N))
    
    test  = indices[list(range(a))]
    train = indices[list(range(a,N))]

    test_p,  test_l,  test_d  = parameters[test],  lifts[test],  drags[test]
    train_p, train_l, train_d = parameters[train], lifts[train], drags[train]

    cwd = os.getcwd()
    os.chdir('./datasets')

    test_files = ['testing_parameters.txt', 'testing_lifts.txt', 'testing_drags.txt']
    test_data = [test_p, test_l, test_d]

    for i in range(3):
        
        with open(test_files[i], 'w') as f:
           
            f.truncate(0)
            np.savetxt(f, test_data[i])
            f.close()

    indices = np.array(range(N-a))
    S = []

    train_files = ['training_parameters', 'training_lifts', 'training_drags']

    for size in range(sizes):
        
        S.append((size + 1) * (N - a) // sizes)

        for rep in range(reps):
            
            for _ in range(1000):
                random.shuffle(indices)

            ls = indices[list(range((size + 1) * (N - a) // sizes))]
            
            train_data = [train_p[ls], train_l[ls], train_d[ls]]

            for i in range(3):
                
                with open(train_files[i]+'_S{:02d}R{:02d}.txt'.format(size,rep), 'w') as f:
                    
                    f.truncate(0)
                    np.savetxt(f, train_data[i])
                    f.close()

    os.chdir(cwd)
    np.savetxt('sizes.txt', S, fmt='%i')
