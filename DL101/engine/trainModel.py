import warnings
warnings.filterwarnings("ignore")
import pandas as pd 
import numpy as np
from termcolor import colored
from IPython.core.display import display
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from engine import model_evaluation_utils as meu
from keras.callbacks import EarlyStopping
from sklearn.feature_extraction import DictVectorizer
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD


# setting display parameterwith pandas
pd.set_option('display.max_rows', 50000)
pd.set_option('display.max_columns', 50000)
pd.set_option('display.width', 100000)
pd.set_option('display.max_colwidth', -1)

def TrainModel(df):    
    X_dict = df[['Strategy', 'Profile']].to_dict(orient='records') 
    dv_X = DictVectorizer(sparse=False) 
    X_encoded = dv_X.fit_transform(X_dict)  
    encode = dv_X.vocabulary_
    print('encode Dictionary \n', encode)    
    Profile = {'bike': 0,'car': 1,'walking': 2,}
    Strategy = {'Micro': 3,'RI': 4,'SIA': 5}   
    dmap = {'bike': 0,'car': 1,'walking': 2}
    df['Profile']= df['Profile'].map(Profile)
    df['Strategy']= df['Strategy'].map(Strategy)
    df['Catchment Population']= round(df['Catchment Population']/7000)     
    
    x_train = np.array(df.drop('Duration(Days)', axis=1))
    y_train = np.array(keras.utils.to_categorical(df['Duration(Days)'], ))
    x_test = np.array(df.drop('Duration(Days)', axis=1))
    y_test = np.array(keras.utils.to_categorical(df['Duration(Days)'], ))  

    Epochs = 1000    
    INIT_LR = 1e-2    
    Batch = 32    
    MOM = 0.5
    sgd = SGD(lr=INIT_LR, momentum=MOM, decay=INIT_LR / Epochs)
    model = Sequential()
    model.add(Dense(20, activation='relu', input_shape=(x_train.shape[1],)))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(y_train.shape[1], activation='softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    print('INITIALLIZING_TRAINING.....\n')
    H = model.fit(x_train,
                        y_train,
                        epochs=Epochs,
                         verbose = 0,                     
                         batch_size=Batch, 
                         validation_data=(x_test,
                                          y_test))

    eval_result = model.evaluate(x_train,
                        y_train,)
    print("Test accuracy:", eval_result[1],
         "\nTest loss:", eval_result[0],'\n')
    #extracting predictions Neural network
    y_pred = model.predict(x_test)
    # NEURAL NETWORK MODEL ACCURACY
    #checking the prediction and print the accuracy
    print(colored('MODEL ACCURACY = ',
                  'blue',attrs=['reverse','blink']),
              np.round(accuracy_score(y_test.argmax(axis=1),
             y_pred.argmax(axis=1))*100,2),'%\n')
    model.save("model/Test_covens_model.h5")
    print('The Model is saved as "Test_covens_model.h5" saved!')