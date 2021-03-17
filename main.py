import numpy as np
import subprocess
import os
import logging

logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from seperate_test_data import seperate_test_data
from predict import predict
from hyperparameter_config import hyperparameterConfiguration

def get_training_command(size, rep, conf, force):
    
    conf_id = conf[0]
    conf_file = conf[1]
    model_file = 'model_'+force+'_S{:02d}R{:02d}C{:05d}.h5'.format(size,rep,conf_id)
    loss_file = 'loss_'+force+'_S{:02d}R{:02d}C{:05d}.txt'.format(size,rep,conf_id)
    
    training_parameters = 'training_parameters_S{:02d}R{:02d}.txt'.format(size,rep)
    training_values = 'training_'+force+'_S{:02d}R{:02d}.txt'.format(size,rep)
    
    simple_configuration_file = '--simple_configuration_file ./configs/{} '.format(conf_file)
    output_model_file = '--output_model_file ./models/{} '.format(model_file)
    input_parameters_file = '--input_parameters_files ./datasets/{} '.format(training_parameters)
    input_values_file = '--input_values_files ./datasets/{} '.format(training_values)
    save_loss = '--save_loss_output_file ./losses/{} '.format(loss_file)

    training_command = 'bash run_python_script.sh '+ output_model_file + input_parameters_file \
                                                + input_values_file + simple_configuration_file + save_loss

    return training_command

if __name__ == "__main__":
    
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()
    
    parameters = np.loadtxt('all_data/parameters.txt')
    lifts = np.loadtxt('all_data/lifts.txt')
    drags = np.loadtxt('all_data/drags.txt')

    sizes = 2      # total no. of training-set sizes = [64,128,192,...,1024]
    reps = 2        # no. of training-sets of a given size
    
    configs = hyperparameterConfiguration('training_parameters.json',rank)

    if rank == 0:
        
        if not os.listdir('./datasets/'):

            seperate_test_data(parameters, lifts, drags, sizes, reps)
            np.savetxt('statistical_parameters.txt', [sizes, reps, configs.total], fmt='%i')

    comm.barrier()

    for size, rep in [(x,y) for x in range(sizes) for y in range(reps)]:
        
        if (size * reps + rep) % nprocs == rank:
            
            for conf in configs:
                
                for force in ['lifts', 'drags']:
                    
                    command = get_training_command(size, rep, conf, force)
                    subprocess.run(command.split())
                    predict(size, rep, conf, force)
                    
