import os 
import pandas as pd
import math
import csv
import numpy as np

file_weather = pd.read_csv('./weather_data/all_p.csv')
for n in range(27):
    print("process:", n)

    path = './data+/'
    name1 = 'test_'+str(n)
    name2 = 'train_'+str(n)

    file1 = pd.read_csv(path+str(name1)+'.csv')
    file2 = pd.read_csv(path+str(name2)+'.csv')
    mer1 = pd.merge(file1, file_weather, on="date-time", how='inner')
    mer2 = pd.merge(file2, file_weather, on="date-time", how='inner')

    path = './data_merge/'
    name1 = 'test_'+str(n)
    name2 = 'train_'+str(n)
    file1 = open(path+str(name1)+'.csv', 'a+', encoding='utf-8', newline='')
    wr1 = csv.writer(file1)
    file2 = open(path+str(name2)+'.csv', 'a+', encoding='utf-8', newline='')
    wr2 = csv.writer(file2)

    print("\t",len(mer1), len(mer2))
    #,date-time,day,active_power,active_energy,temp,humid
    mer1 = mer1[['date-time','day','temp','humid','active_power','active_energy']]
    mer2 = mer2[['date-time','day','temp','humid','active_power','active_energy']]
    
    line=[['month','date','time','day','temp','humid','active_power','active_energy']]
    wr1.writerows(line)
    wr2.writerows(line)

    train_set = []
    test_set = []
    for x in range(len(mer1)):
        line = list(mer1.loc[x][:])
        if np.isnan(line[2]) or np.isnan(line[3]):
            continue
        else:
            new_line = [int(line[0][5:7]),int(line[0][8:10]),int(line[0][11:]),line[1],line[2],line[3],line[4],line[5]]
            test_set.append(new_line)
    wr1.writerows(test_set)
    for x in range(len(mer2)):
        line = list(mer2.loc[x][:])
        if np.isnan(line[2]) or np.isnan(line[3]):
            continue
        else:
            new_line = [int(line[0][5:7]),int(line[0][8:10]),int(line[0][11:]),line[1],line[2],line[3],line[4],line[5]]
            train_set.append(new_line)
    wr2.writerows(train_set)
    # mer1.to_csv('./data_merge/'+name1+'.csv', mode='w')
    # mer2.to_csv('./data_merge/'+name2+'.csv', mode='w')