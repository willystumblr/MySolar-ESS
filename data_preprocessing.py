import pandas as pd
import glob
import numpy as np
import itertools as it
import os

PATH = 'solar_data'
PATH_W = 'datasets/weather_n'
df = pd.DataFrame()

for (f1, f2) in zip(sorted(glob.glob(PATH+"/*.xls")), sorted(glob.glob(PATH_W+"/*.csv"))):
    
    print(f1, f2)
    
    df_temp = pd.read_excel(f1, usecols="A:E", header=5)
    df_wthr = pd.read_csv(f2)

    # Drop unnecesary columns
    df_temp["수평면(w/㎡)"], df_temp["외기온도(℃)"], df_temp["경사면(w/㎡)"], df_temp["모듈온도(℃)"] =  df_temp["w/㎡"], df_temp["℃"], df_temp["w/㎡.1"], df_temp["℃.1"]
    df_env=df_temp.drop(axis=1, columns=["Unnamed: 0", "w/㎡", "℃", "w/㎡.1","℃.1"])
    df_wthr = df_wthr.drop(axis=1, columns=["지점", "지점명", '운형(운형약어)', "지면상태(지면상태코드)"])
    
    df_wthr['일시'] = pd.to_datetime(df_wthr['일시'])
    
    # Process NaN value in weather data
    df_wthr['강수량(mm)'] = df_wthr['강수량(mm)'].fillna(0.0)
    df_wthr['일사(MJ/m2)'] = df_wthr['일사(MJ/m2)'].fillna(0.00)
    df_wthr['일조(hr)'] = df_wthr['일조(hr)'].fillna(0.0)
    
    
    
    
    df_soc = pd.read_excel(f1, usecols="F:G", header=5) # 축구장
    df_stu = pd.read_excel(f1, usecols="H:I", header=5) # 학생회관
    df_war = pd.read_excel(f1, usecols="J:K", header=5) # 중앙창고
    df_hac = pd.read_excel(f1, usecols="L:M", header=5) # 학사과정
    df_das = pd.read_excel(f1, usecols="N:O", header=5) # 다산빌딩
    df_fac = pd.read_excel(f1, usecols="P:Q", header=5) # 시설관리동
    df_clC = pd.read_excel(f1, usecols="R:S", header=5) # 대학C동
    df_exp = pd.read_excel(f1, usecols="T:U", header=5) # 동물실험동
    df_lib = pd.read_excel(f1, usecols="V:W", header=5) # 중앙도서관
    df_lg_ = pd.read_excel(f1, usecols="X:Y", header=5) # LG도서관
    df_rne = pd.read_excel(f1, usecols="Z:AA", header=5) # 신재생에너지동
    df_sse = pd.read_excel(f1, usecols="AB:AC", header=5) # 삼성환경동
    df_gcf = pd.read_excel(f1, usecols="AD:AE", header=5) # 중앙연구기기센터
    df_shy = pd.read_excel(f1, usecols="AF:AG", header=5) # 산업협력관 
    df_drm = pd.read_excel(f1, usecols="AH:AI", header=5) # 기숙사B동'''
    
    data = {
        "축구장": df_soc,
        "학생회관": df_stu,
        "중앙창고": df_war,
        "학사과정": df_hac,
        "다산빌딩": df_das,
        "시설관리동": df_fac,
        "대학C동": df_clC,
        "동물실험동": df_exp,
        "중앙도서관": df_lib,
        "LG도서관": df_lg_,
        "신재생에너지동": df_rne,
        "삼성환경동": df_sse,
        "산업협력관": df_shy,
        "기숙사B동": df_drm
        
    }
    '''
    
    '''
    
    for k in data.keys():
        if len(data[k].columns)<2: continue
        x = data[k]
        t = list(x.columns)
        x["누적발전량(kWh)"], x["시간당발전량(kWh)"] = x[t[0]], x[t[1]]
        x = x.drop(axis=1, columns=[t[0], t[1]])
        x=x.drop(24)
        
        
        
        # Process NaN in each building's data
        x.fillna(method='ffill', inplace=True)
                     
        x = x.join(df_env, how='right')
        x = x.join(df_wthr, how='right')
        
        x['일시'] = pd.to_datetime(x['일시'])
        
        outname = x['일시'][0].strftime('%Y-%m-%d')+'.csv'

        parent_dir = 'dataset_merged/'
        
        outdir = os.path.join(parent_dir, k)
        if not os.path.exists(os.path.join(parent_dir, k)):
            
            os.makedirs(outdir)
        
        
        
        fullname = os.path.join(outdir, outname)
        
        x.to_csv(fullname, index=False, header=True)
        
    
    
    
    
    

    
    
    
    
    
    
    