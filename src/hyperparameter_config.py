import json
from itertools import product

class hyperparameterConfiguration:

    def __init__(self, rank):

        with open('hyper.json','r') as f:
            
            self.all_hyperparameters = json.load(f)
            f.close()
        
        with open('training_parameters.json','r') as f:

            self.training_parameters = json.load(f)
            f.close()

        self.hyperparameter_vals = self.all_hyperparameters.values()
        self.total = 0
        self.rank = rank
        self.all_config_files = []
        
        for conf_id, conf in enumerate(product(*self.hyperparameter_vals)):
            
            self.training_parameters['optimizer'] = conf[0]
            self.training_parameters['loss'] = conf[1]
            self.training_parameters['network_topology'] = [5] + [conf[3] for _ in range(conf[2])] + [1]
            self.training_parameters['activation'] = conf[4]
            self.training_parameters['epochs'] = conf[5]
            self.training_parameters['learning_rate'] = conf[6]
            self.training_parameters['retrainings'] = conf[7]

            config_file = 'conf_{:05d}.json'.format(conf_id)
            
            if self.rank == 0:
                
                with open('./configs/'+config_file,'w') as f:
                    
                    f.truncate(0)
                    json.dump(self.training_parameters, f)
                    f.close()

            self.total += 1
            self.all_config_files.append(config_file)

    def __iter__(self):
        
        self.n = 0
        
        return self

    def __next__(self):
        
        if self.n < self.total:

            conf_id = self.n
            conf_file = self.all_config_files[self.n]
            self.n += 1
            
            return self.iterator(conf_id, conf_file)

        else:

            raise StopIteration

    class iterator:

        def __init__(self, idx, filename):

            self.idx = idx
            self.filename = filename

