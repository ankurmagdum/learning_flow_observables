import numpy as np
import os
import logging

logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from subprocess import run
from src import *


def get_training_command(size, rep, conf, force):
    
    model_file = 'model_'+force+'_S{:02d}R{:02d}C{:05d}.h5'.format(size,rep,conf.idx)
    loss_file = 'loss_'+force+'_S{:02d}R{:02d}C{:05d}.txt'.format(size,rep,conf.idx)
    
    training_parameters = 'training_parameters_S{:02d}R{:02d}.txt'.format(size,rep)
    training_values = 'training_'+force+'_S{:02d}R{:02d}.txt'.format(size,rep)
    
    simple_configuration_file = '--simple_configuration_file ./configs/{} '.format(conf.filename)
    output_model_file = '--output_model_file ./models/{} '.format(model_file)
    input_parameters_file = '--input_parameters_files ./datasets/{} '.format(training_parameters)
    input_values_file = '--input_values_files ./datasets/{} '.format(training_values)
    save_loss = '--save_loss_output_file ./losses/{} '.format(loss_file)

    training_command = 'python3 train.py '+ output_model_file + input_parameters_file \
                                                + input_values_file + simple_configuration_file + save_loss

    return training_command


if __name__ == "__main__":
    
    from mpi4py import MPI
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()
    
    parameters = np.loadtxt('all_data/parameters.txt')     # inputs
    lifts = np.loadtxt('all_data/lifts.txt')               # outputs
    drags = np.loadtxt('all_data/drags.txt')               # outputs  

    test_fraction = 1/2
    N_train = int((1 - test_fraction) * len(parameters))

    sizes = 1      
    reps = 2        
    set_sizes = [(s + 1) * N_train // sizes for s in range(sizes)]
    
    configs = hyperparameterConfiguration(rank)

    if rank == 0:

        seperate_test_data(parameters, lifts, drags, sizes, reps, test_fraction)
        np.savetxt('statistical_parameters.txt', [sizes, reps, configs.total], fmt='%i')

    comm.barrier()

    for conf, force in [(p,q) for p in configs for q in ['lifts','drags']]:
    
        for size, rep in [(x,y) for x in range(sizes) for y in range(reps)]:
        
            if conf.idx % nprocs == rank:
            
                command = get_training_command(size, rep, conf, force)
                run(command.split())
                predict(size, rep, conf, force)

        compute_error_statistics(sizes, reps, conf, force)
        print('finished conf {}, force {}'.format(conf.idx, force[:-1]), '\n')
                    
