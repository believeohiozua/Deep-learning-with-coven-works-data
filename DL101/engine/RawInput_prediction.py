## #turn of warnings 
import warnings
warnings.filterwarnings("ignore")
import pandas as pd 
import numpy as np
from keras.models import load_model
from engine import model_evaluation_utils as meu
from termcolor import colored



def RawInput_Testing(): 
    try:
        Strategy = int(input("select Strategy {3 for 'Micro', 4 for 'RI':, 5 for 'SIA'} :") )
        Team_Number = int(input('Enter Team Number: '))
        Catchment_Population = int(input('Enter Catchment Population: '))        
        Profile = int(input("select Profile  {0 for 'bike':, 1 for 'car':, 2 for 'walking'} :"))
    except:
        Team_Number = None
        Catchment_Population = None
        Strategy = None
        Profile = None
        print("ERROR REPORT: Numeric entry Only")
    try:
        if Strategy:
            if Strategy >5 or Strategy <3:
                print("ERROR REPORT: Check your entery for Strategy |max = 5, min=3|")
            else:
                pass
        if Profile:
            if Profile >2 or Profile <0:
                print("ERROR REPORT: Check your entery for Profile |max = 2, min=0|")
            else:
                pass    
        if Strategy <=5 and Strategy >=3 and Profile <=3 and Profile >=0 and type(Strategy)==int and type(Profile)== int and  type(Catchment_Population) ==int and  type(Team_Number) ==int:
            data = {'Strategy': [Strategy],'Team Number': [Team_Number], 'Catchment Population': [Catchment_Population],
                   'Profile': [Profile],}
            df = pd.DataFrame(data=data)
            df = df[['Strategy','Team Number','Catchment Population','Profile']]
            df['Catchment Population']= round(df['Catchment Population']/7000) 
            print('PREDICTING.....')
            model= load_model('model/Test_covens_model.h5')
            prediction = (model.predict_classes(df, verbose=0))
            if prediction[0]>1:
                print(colored('duration = ','blue',attrs=['reverse','blink']) ,prediction[0],'days', '\n')
            else:
                print(colored('duration = ','blue',attrs=['reverse','blink']) ,prediction[0],'day','\n')
            labels = list(prediction)
            meu.display_model_performance_metrics(true_labels=labels, 
                                          predicted_labels=prediction, 
                                          classes=list(set(labels)))
        else:
            return 'one or more of your entery is invalid, Please check the "ERROR REPORT" and try Again'
    except:
        print('one or more of your entery is invalid, Please check the "ERROR REPORT" and try Again')
