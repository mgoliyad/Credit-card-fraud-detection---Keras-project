import numpy as np
import tensorflow as tf
from numpy import newaxis
from keras.layers import Dense, Dropout, LSTM, Bidirectional, Conv1D
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.backend import flatten

class acc_Callback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('acc')>0.99):
    #if(logs.get('loss')<0.4):
      print("\nReached 99% accuracy so cancelling training!")
      self.model.stop_training = True

class Model(acc_Callback):
    def __init__(self):
        self.model = Sequential()
        self.acc_Callback = acc_Callback()
        
    def load_model(self, filepath):
        self.model = load_model(filepath)
        
    def build_model(self, configs):
        for layer in configs['model']['layers']:
            neurons = layer['neurons'] if 'neurons' in layer else None
            filters = layer['filters'] if 'filters' in layer else None
            kernel_size = layer['kernel_size'] if 'kernel_size' in layer else None
            dropout_rate = layer['rate'] if 'rate' in layer else None
            activation = layer['activation'] if 'activation' in layer else None
            return_seq = layer['return_seq'] if 'return_seq' in layer else None
            input_timesteps = layer['input_timesteps'] if 'input_timesteps' in layer else None
            channels = layer['channels'] if 'input_dim' in layer else None
            input_dim = layer['input_dim'] if 'input_dim' in layer else None
            #learnin_rate = configs['model']['learnin_rate']
            metrics = [configs['model']['metrics']]
            
            if layer['type'] == 'bidirectional': self.model.add(Bidirectional(LSTM(neurons, return_sequences=return_seq), input_shape=(input_timesteps, channels)))
            if layer['type'] == 'dense': self.model.add(Dense(neurons, activation=activation))
            if layer['type'] == 'lstm': self.model.add(LSTM(neurons, input_shape=(input_timesteps, input_dim), return_sequences=return_seq))
            if layer['type'] == 'dropout': self.model.add(Dropout(dropout_rate))
            if layer['type'] == 'cnn': self.model.add(Conv1D(filters, kernel_size, strides=1, padding='same', activation='sigmoid', input_shape=[None, 1]))
            if layer['type'] == 'flatten': self.model.add(flatten())
            
            self.model.compile(loss=configs['model']['loss'], optimizer=configs['model']['optimizer'], metrics=metrics)
             
             
    def train(self, x, y, x_val, y_val, epochs, batch_size):  
	
        #acc_Callback = acc_Callback()
        callbacks = [
                    #EarlyStopping(monitor='binary_accuracy', patience=5), 
                    #ReduceLROnPlateau(monitor='binary_accuracy', patience=2, cooldown=2),
                    self.acc_Callback
                    #LearningRateScheduler(lambda epoch: 1e-8 * 10**(epoch / 20))
                    ]

        x = x[:, :, newaxis]
        x_val = x_val[:, :, newaxis]
        estimator = self.model.fit(x,y, validation_data=(x_val, y_val), epochs=epochs, batch_size=batch_size,callbacks=callbacks, verbose=1)
		
        return estimator
    
    def predict(self, data):
        #Add this lines if only one feature
        data = data[:, :, newaxis]
        predicted = self.model.predict(data)
        #predicted = np.reshape(predicted, (predicted.size,))
        return np.array(predicted)
    
