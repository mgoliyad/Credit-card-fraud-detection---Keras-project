import os
import json
import numpy as np
import matplotlib.pyplot as plt
from data_reader import DataLoader
from model import Model
import datetime as dt

def plot(train_data, val_data, label1, label2):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(train_data, label=label1, color='r')
    plt.plot(val_data, label=label2, color='b')
    plt.legend()
    plt.show()

def main():
    configs = json.load(open('config.json', 'r'))
    if not os.path.exists(configs['model']['save_dir']): os.makedirs(configs['model']['save_dir'])
    batch_size = configs['training']['batch_size']
    data = DataLoader(os.path.join('data', configs['data']['filename']))
        
    model = Model()
    
    model.build_model(configs)
    
    x_train, y_train, x_val, y_val, x_test, y_test = data.get_data(configs)
    
    start_dt = dt.datetime.now()
    estimator = model.train(x_train, y_train, x_val, y_val, epochs = configs['training']['epochs'], batch_size = batch_size)
    end_dt = dt.datetime.now()
    print('Time taken to train model: %s' % (end_dt - start_dt))
    
    #print(estimator.history.keys())
    
    plot(estimator.history['acc'], estimator.history['val_acc'], label1='Train accuracy', label2='Test accuracy')
    plot(estimator.history['loss'], estimator.history['val_loss'], label1='Train loss', label2='Test loss')
    
    predicted_prob = model.predict(x_test)

    prediction = np.where(predicted_prob >=0.5, 1, 0)
    
    print('We got ', np.sum(prediction), ' misclassified cases out of ', len(y_test), ' total cases.' )
    
if __name__ == '__main__':
    main()
