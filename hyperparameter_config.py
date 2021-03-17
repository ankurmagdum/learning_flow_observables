import json
from itertools import product

class hyperparameterConfiguration:

    def __init__(self, filename, rank):

        #self.layers = [2, 4, 8]
        #self.neurons_per_layer = [10, 20, 40]
        #self.learning_rates = [0.01, 0.001, 0.0001]
        #self.optimizers = ['adam']
        #self.retrainings = [5, 25, 50]
        #self.losses = ['mean_squared_error', 'mean_absolute_error']
        #self.epochs = [1000, 5000, 10000]
        #self.activations = ['relu', 'tanh']
        
        self.layers = [2]
        self.neurons_per_layer = [10]
        self.learning_rates = [0.01]
        self.optimizers = ['lbfgs']
        self.retrainings = [5]
        self.losses = ['mean_squared_error']
        self.epochs = [1000]
        self.activations = ['relu']
        self.total = 0
        self.rank = rank
        self.all_configurations = []

        with open(filename,'r') as f:
            self.base_conf = json.load(f)
            f.close()

        for conf_id, conf in enumerate(product(self.layers, self.neurons_per_layer, self.learning_rates, self.optimizers,\
                                                self.retrainings, self.losses, self.epochs, self.activations)):
            
            network_topology = [5]
            for _ in range(conf[0]):
                network_topology.append(conf[1])
            network_topology.append(1)

            self.base_conf['network_topology'] = network_topology
            self.base_conf['learning_rate'] = conf[2]
            self.base_conf['optimizer'] = conf[3]
            self.base_conf['retrainings'] = conf[4]
            self.base_conf['loss'] = conf[5]
            self.base_conf['epochs'] = conf[6]
            self.base_conf['activation'] = conf[7]

            config_file = './configs/conf_{:05d}.json'.format(conf_id)
            
            if self.rank == 0:
                with open(config_file,'w') as f:
                    f.truncate()
                    json.dump(self.base_conf, f)
                    f.close()
            
            self.all_configurations.append(self.base_conf)
            self.total += 1

    def __iter__(self):
        
        self.n = 0
        return self

    def __next__(self):
        
        if self.n < self.total:

            conf = self.all_configurations[self.n]
            conf_id = self.n
            conf_file = './configs/conf_{:05d}_{:03d}.json'.format(self.n, self.rank)
            
            with open(conf_file,'w') as f:
                json.dump(conf, f)
                f.close()

            self.n += 1
            
            return (conf_id, conf_file[10:])

        else:

            raise StopIteration
