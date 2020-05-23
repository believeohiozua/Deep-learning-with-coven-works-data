#!/usr/bin/python3
#!/author=Believe Ohiozua
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from engine.trainModel import TrainModel
from engine.CsvEntry_prediction import DataFrame_Testing
from engine.RawInput_prediction import RawInput_Testing
import os
import logging
import argparse
import sys

def main():
    #logging
    logging.basicConfig(filename='applog.log',
                        filemode='w',
                        level=logging.INFO,
                        format='%(levelname)s:%(message)s')
    parser = argparse.ArgumentParser()   
    parser.add_argument("-tr","--Train", help ='add dataFrame to train e.g. "python app.py -tr dataset/Test_covens.xlsx" ', type=str)    
    parser.add_argument("-t","--Test", help ='Enter: "python app.py -t TM" to test model with Raw Input', type=str)
    parser.add_argument("-d","--Data", help ='add dataFrame to test e.g. "python app.py -d dataset/Test_covens.xlsx" ', type=str)
    args = parser.parse_args()
    if args.Train:
        try:
            Rawdata = pd.read_excel(args.Train)
        except:
            Rawdata = pd.read_csv(args.Train)
            pass
        TrainModel(Rawdata)        
    elif args.Test == 'TM':
        RawInput_Testing()
    elif args.Data:
        try:
            df = pd.read_excel(args.Data)
        except:
            df = pd.read_csv(args.Data)
            pass
        DataFrame_Testing(df)    
    else:
        print("Your Entry is invalid!")



if __name__=='__main__':
    main()

