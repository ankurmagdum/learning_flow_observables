import numpy as np
import os

def compute_error_statistics(sizes, reps, conf, force):

    testing_values = np.loadtxt('./datasets/testing_'+force+'.txt')
    norm = 0.01622 if force=='drags' else 0.8200
    
    mean_error = np.zeros(sizes) 
    std_error = np.zeros(sizes)

    for size in range(sizes):

        error = np.zeros(reps)

        for rep in range(reps):
            
            filename = './predictions/prediction_'+force+'_S{:02d}R{:02d}C{:05d}.txt'.format(size,rep,conf.idx)
            
            if os.path.exists(filename):
                predicted_values = np.loadtxt(filename)
                error[rep] = np.sum(np.abs(testing_values - predicted_values)) / (norm * len(predicted_values))
                os.remove(filename)
                print(size, rep)
            else:
                print(size, rep, conf.idx)
                                                
        mean_error[size] = np.mean(error[:])
        std_error[size]  = np.std(error[:])

        print('error: ', error)

    with open('./errors/error_{}_C{:05d}.txt'.format(force,conf.idx), 'w') as f:
        
        f.truncate(0)
        np.savetxt(f, np.stack((mean_error, std_error)))
        f.close()
