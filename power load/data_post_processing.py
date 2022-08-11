import os 
import pandas as pd
import math
import csv

for n in range(27):
    print("process:", n)
    name = './building_data/'+str(n)+'.csv'
    df = pd.read_csv(name)
    path = './data+/'
    name1 = 'test_'+str(n)
    name2 = 'train_'+str(n)
    file1 = open(path+str(name1)+'.csv', 'a+', encoding='utf-8', newline='')
    wr1 = csv.writer(file1)
    file2 = open(path+str(name2)+'.csv', 'a+', encoding='utf-8', newline='')
    wr2 = csv.writer(file2)

    # line=[['month','date','time','day','active_power','active_energy']]
    line=[['date-time','day','active_power','active_energy']]
    # line=[['date-time','day','active_energy']]
    train_set = []
    test_set = []

    wr1.writerows(line)
    wr2.writerows(line)
    # print(df.loc[:23][:])
    for i in range(0,len(df)-24,25):
        train_set = []
        test_set = []
        if int(list(df.loc[i][:])[0][9:11])%4 == 0:
            for j in range(24):
                x = i+j
                line = list(df.loc[x][:])
                if line[2] == 'Sat' or line[2] == 'Sun':
                    day = 0
                else:
                    day = 1
                # new_line = [int(line[0][5:7]),int(line[0][8:10]),line[1],day,line[3],line[4]]
                # new_line = [int(line[0][5:7]),int(line[0][8:10]),line[1],day,line[4]]
                new_line = [line[0]+'-'+str(line[1]),day,line[3],line[4]]
                test_set.append(new_line)
            wr1.writerows(test_set)

        else:
            for j in range(24):
                x = i+j
                line = list(df.loc[x][:])
                if line[2] == 'Sat' or line[2] == 'Sun':
                    day = 0
                else:
                    day = 1
                # new_line = [int(line[0][5:7]),int(line[0][8:10]),line[1],day,line[3],line[4]]
                # new_line = [int(line[0][5:7]),int(line[0][8:10]),line[1],day,line[4]]
                new_line = [line[0]+'-'+str(line[1]),day,line[3],line[4]]
                train_set.append(new_line)
            wr2.writerows(train_set)
    file1.close()
    file2.close()

    # df1 = pd.read_csv('./data/test_0.csv')
    # df2 = pd.read_csv('./data/train_0.csv')
    # print(len(df1), len(df2))