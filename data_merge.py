import os
import pandas as pd
import glob
from tqdm import tqdm, tqdm_pandas

df=pd.DataFrame()
datalist = os.listdir('dataset_merged/')
os.chdir('dataset_merged/')
for dir in datalist:
    
    os.chdir(dir)
    
    extension = 'csv'
    all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
    
    #combine all files in the list
    tqdm.pandas()
    combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])
    #export to csv
    NAME = os.path.abspath(os.getcwd()).split('/')[-1]
    os.chdir('/mnt/giai/EMStrack')
    PATH = "data_combined/"+NAME
    os.makedirs(PATH)
    combined_csv.to_csv(PATH+"/combined.csv", index=False)
    os.chdir('dataset_merged/')
    