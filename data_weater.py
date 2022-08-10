import glob
import pandas as pd

#get the number of lines of the csv file to be read


#size of rows of data to write to the csv,

#you can change the row size according to your need
rowsize = 24


for x in glob.glob('datasets/weather/*.csv'):
    df = pd.read_csv(x, encoding='euc_kr')
    number_lines = len(df)
    for s in range(0, len(df), rowsize):
        df.iloc[s:s+rowsize].to_csv(f"datasets/weather_n/{df['일시'][s]}.csv", index=False, header=True)