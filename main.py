import numpy as np
import os
import logging

logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from subprocess import run
from src import *


def get_training_command(size, rep, conf, force):
   
    if not os.path.exists('./models/'):
        os.mkdir('./models/')

    model_file = 'model_'+force+'_S{:02d}R{:02d}C{:05d}.h5'.format(size,rep,conf.id)
    loss_file  = 'loss_'+force+'_S{:02d}R{:02d}C{:05d}.txt'.format(size,rep,conf.id)
    
    training_parameters = 'training_parameters_S{:02d}R{:02d}.txt'.format(size,rep)
    training_values     = 'training_'+force+ '_S{:02d}R{:02d}.txt'.format(size,rep)
    
    simple_configuration_file = '--simple_configuration_file ./configs/{} '.format(conf.file)
    output_model_file         = '--output_model_file ./models/{} '.format(model_file)
    input_parameters_file     = '--input_parameters_files ./datasets/{} '.format(training_parameters)
    input_values_file         = '--input_values_files ./datasets/{} '.format(training_values)
    save_loss                 = '--save_loss_output_file ./losses/{} '.format(loss_file)

    training_command = 'python3 train.py '+ output_model_file + input_parameters_file \
                                          + input_values_file + simple_configuration_file + save_loss

    return training_command

def train(size, rep, conf, force):

    command = get_training_command(size, rep, conf, force)
    run(command.split())


if __name__ == "__main__":
    
    from mpi4py import MPI
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()
    
    parameters = np.loadtxt('all_data/parameters.txt')     # inputs
    lifts = np.loadtxt('all_data/lifts.txt')               # outputs
    drags = np.loadtxt('all_data/drags.txt')               # outputs  

    test_fraction = 1/2
    sizes, reps = 24, 8

    '''
    the variable <sizes> is used to generate a sequence of training-sets 
    with increasing size in the following manner:

    train_set_sizes = [(s + 1) * all_train // sizes for s in range(sizes)]

    the variable <reps> is the multiplicity of every training-set size in 
    train_set_sizes
    '''
    
    configs = hyperparameterConfiguration(rank)

    if rank == 0:

        seperate_test_data(parameters, lifts, drags, sizes, reps, test_fraction)

    comm.barrier()

    for conf, force in [(p,q) for p in configs for q in ['lifts','drags']]:
    
        if conf.id % nprocs == rank:
        
            for size, rep in [(x,y) for x in range(sizes) for y in range(reps)]:
            
                train(size, rep, conf, force)
                predict(size, rep, conf, force)

            compute_error_statistics(sizes, reps, conf, force)
                
