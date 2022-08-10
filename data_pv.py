import glob
import pandas as pd
import os

PATH = "datasets/pv data/"
for x in glob.glob(PATH+'*.xls'):
    old = x
    new = x.split('_')[1]
    
    os.rename(x, 'solar_data/'+new+'.xls')