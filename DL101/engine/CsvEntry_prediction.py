## #turn of warnings 
import warnings
warnings.filterwarnings("ignore")
import pandas as pd 
import numpy as np
from keras.models import load_model
from engine import model_evaluation_utils as meu
from IPython.core.display import display



def DataFrame_Testing(df):
    Profile = {'bike': 0,'car': 1,'walking': 2,}
    Strategy = {'Micro': 3,'RI': 4,'SIA': 5}
    df['Profile']= df['Profile'].map(Profile)
    df['Strategy']= df['Strategy'].map(Strategy)
    df = df[['Strategy','Team Number','Catchment Population','Profile']]
    df['Catchment Population']= round(df['Catchment Population']/7000) 
    model = load_model('model/Test_covens_model.h5')
    prediction = (model.predict_classes(df, verbose=1))
    data = {'Duration(Days)': prediction}
    result = pd.DataFrame(data=data)
    df['Duration(Days)'] = result['Duration(Days)']
    labels = list(df['Duration(Days)'])
    display('RESULT',df)
    meu.display_model_performance_metrics(true_labels=labels, 
                                      predicted_labels=prediction, 
                                      classes=list(set(labels)))
