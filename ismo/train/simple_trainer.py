from ismo.train import Parameters
import ismo.train.optimizers
import h5py
import tensorflow.keras.callbacks
import copy
import numpy as np
import tensorflow.keras.initializers
import tensorflow as tf
import tensorflow_probability as tfp
 
def function_factory(model, loss, train_x, train_y):
    """A factory to create a function required by tfp.optimizer.lbfgs_minimize.
    Args:
        model [in]: an instance of `tf.keras.Model` or its subclasses.
        loss [in]: a function with signature loss_value = loss(pred_y, true_y).
        train_x [in]: the input part of training data.
        train_y [in]: the output part of training data.
    Returns:
        A function that has a signature of:
            loss_value, gradients = f(model_parameters).
    """

    # obtain the shapes of all trainable parameters in the model
    shapes = tf.shape_n(model.trainable_variables)
    n_tensors = len(shapes)

    # we'll use tf.dynamic_stitch and tf.dynamic_partition later, so we need to
    # prepare required information first
    count = 0
    idx = [] # stitch indices
    part = [] # partition indices

    for i, shape in enumerate(shapes):
        n = np.product(shape)

        idx.append(tf.reshape(tf.range(count, count+n, dtype=tf.int32), shape))
        part.extend([i]*n)
        count += n


    part = tf.constant(part)

    @tf.function
    def assign_new_model_parameters(params_1d):
        """A function updating the model's parameters with a 1D tf.Tensor.
        Args:
            params_1d [in]: a 1D tf.Tensor representing the model's trainable parameters.
        """

        params = tf.dynamic_partition(params_1d, part, n_tensors)
        for i, (shape, param) in enumerate(zip(shapes, params)):
            model.trainable_variables[i].assign(tf.reshape(param, shape))

    # now create a function that will be returned by this factory
    @tf.function
    def f(params_1d):
        """A function that can be used by tfp.optimizer.lbfgs_minimize.
        This function is created by function_factory.
        Args:
           params_1d [in]: a 1D tf.Tensor.
        Returns:
            A scalar loss and the gradients w.r.t. the `params_1d`.
        """

        # use GradientTape so that we can calculate the gradient of loss w.r.t. parameters
        with tf.GradientTape() as tape:
            # update the parameters in the model
            assign_new_model_parameters(params_1d)
            # calculate the loss
            loss_value = loss(model(train_x, training=True), train_y)
        
        # calculate gradients and convert to 1D tf.Tensor
        grads = tape.gradient(loss_value, model.trainable_variables)
        grads = tf.dynamic_stitch(idx, grads)

        # print out iteration & loss
        f.iter.assign_add(1)
        #tf.print("Iter:", f.iter, "loss:", loss_value)

        # store loss value so we can retrieve later
        tf.py_function(f.history.append, inp=[loss_value], Tout=[])
    
        return loss_value, grads

    # store these information as members so we can use them outside the scope
    f.iter = tf.Variable(0)
    f.idx = idx
    f.part = part
    f.shapes = shapes
    f.assign_new_model_parameters = assign_new_model_parameters
    f.history = []

    return f

class SimpleTrainer(object):

    def __init__(self, *, training_parameters: Parameters,
                 model):

        self.retrainings = training_parameters.retrainings
        self.optimizer = training_parameters.optimizer
        self.learning_rate = training_parameters.learning_rate
        self.activation = training_parameters.activation
        self.epochs = training_parameters.epochs
        self.model = model
        self.loss = training_parameters.loss
        self.callbacks = []
        
        if training_parameters.should_use_early_stopping:
            self.callbacks.append(
                tensorflow.keras.callbacks.EarlyStopping(monitor='loss', patience=training_parameters.early_stopping_patience))

        self.writers = []

    def load_from_file(self, filename):
        self.model.load_weights(filename)

    def fit(self, parameters, values):
        best_loss = None
        
        for retraining_number in range(self.retrainings):
            
            self.__reinitialize(self.model)
            
            if self.optimizer == 'lbfgs':

                func = function_factory(self.model, tf.keras.losses.MeanSquaredError(), np.float32(parameters), np.float32(values))
                init_params = tf.dynamic_stitch(func.idx, self.model.trainable_variables)
            
                results = tfp.optimizer.lbfgs_minimize(value_and_gradients_function=func,  
                   initial_position=init_params, max_iterations=5000, tolerance=1.0 * np.finfo(float).eps,
                   x_tolerance=1.0 * np.finfo(float).eps, f_relative_tolerance=1.0 * np.finfo(float).eps)
            
                func.assign_new_model_parameters(results.position)
                loss = func.history[-1]
                tf.print(loss)

            else:
            
                hist = self.model.fit(parameters, values, batch_size=parameters.shape[0]//4,
                                  epochs=self.epochs, verbose=0, callbacks=self.callbacks)

                loss = hist.history['loss'][-1]
                print(loss)
            
            if best_loss is None or loss < best_loss:
                best_weights = copy.copy(self.model.get_weights())
                if self.optimizer == 'lbfgs':
                    best_loss_hist = copy.copy(func.history)
                else:
                    best_loss_hist = copy.copy(hist)

        self.model.set_weights(best_weights)
        self.write_best_loss_history(best_loss_hist)

    def write_best_loss_history(self, loss_history):
        for writer in self.writers:
            writer(loss_history, self.optimizer)

    def add_loss_history_writer(self, writer):
        self.writers.append(writer)

    def save_to_file(self, outputname):
        self.model.save(outputname)

    def __reinitialize(self, model):
        # See https://stackoverflow.com/a/51727616 (with some modifications, does not run out of the box)
        for layer in model.layers:
            weights = np.zeros_like(layer.get_weights()[0])
            biases = np.zeros_like(layer.get_weights()[1])

            if hasattr(layer, 'kernel_initializer'):
                if self.activation == 'tanh':
                    initializer = tf.keras.initializers.GlorotNormal()
                    weights = initializer(weights.shape)
                else:
                    weights = layer.kernel_initializer(weights.shape)

            if hasattr(layer, 'bias_initializer'):
                if self.activation == 'tanh':
                    initializer = tf.keras.initializers.GlorotNormal()
                    biases = initializer(biases.shape)
                else:
                    biases = layer.bias_initializer(biases.shape)

            layer.set_weights((weights, biases))
        
        if self.optimizer != 'lbfgs':
            model.compile(optimizer=ismo.train.optimizers.create_optimizer(self.optimizer, self.learning_rate),
                      loss=self.loss)
