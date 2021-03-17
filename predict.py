import os
import tensorflow as tf
import numpy as np

def predict(size, rep, conf, force):
    
    conf_id = conf[0]
    model_file = 'model_'+force+'_S{:02d}R{:02d}C{:05d}.h5'.format(size,rep,conf_id)
    model = tf.keras.models.load_model('./models/'+model_file)

    testing_parameters = 'testing_parameters.txt'.format(size,rep)
    prediction = model.predict(np.loadtxt('./datasets/'+testing_parameters))

    np.savetxt('./predictions/prediction_'+force+'_S{:02d}R{:02d}C{:05d}.txt'.format(size,rep,conf_id), prediction)
    os.remove('./models/'+model_file)
    #os.remove('./configs/'+conf[1])
