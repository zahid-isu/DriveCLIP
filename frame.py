import pandas as pd
import numpy as np
import cv2
import os
import shutil
import glob
from datetime import datetime, timedelta

"""
Dataset generation from SynDD1
0.5fps= divide by 60
1fps = divide by 30
5fps= divide by 6
10fps= divide by 3

SKIP Hat/SG for now.
"""
csv_list= glob.glob("ann/*")
for i in range(len(csv_list)):
    df= pd.read_csv(csv_list[i])
    df=df.rename(columns={"File Name": "Filename"})
    df.Filename = df.Filename[df.Filename.str.strip() != '']
    df['Filename'].fillna(method='ffill', inplace=True)

    filename= glob.glob("synvid/"+ '*')
    filename = [x for x in filename if csv_list[i].split('\\')[1].split('_')[2] in x and "Dashboard" in x]
    df=df.dropna(subset=['Label']).reset_index(drop=True)
    df['Label'] = df['Label'].astype(int).astype(str)   # do only for N/A
    df = df[df["Label"].str.contains("NA") == False].reset_index(drop=True)
    for i in range (len(filename)):
        file= filename[i].split('\\')[1].split('.')[0]
        
        if file[0]=='R':
            front = '_'.join([file.split('_')[0], file.split('_')[4], file.split('_')[5]])
        else:
            front = '_'.join([file.split('_')[0], file.split('_')[3], file.split('_')[4]])
            
        print("processing..."+f"{front}")
    #     # if file[:4]=="Rear":       #only when "rearview"
    #     #     gap= file[:4]+file[5:len(file)]
    #     #     df1= df.groupby(['Filename']).get_group(gap)
    #     # else:
        
        df1= df.groupby(['Filename']).get_group(file)
        cap = cv2.VideoCapture(filename[i])
        fps = cap.get(cv2.CAP_PROP_FPS)
        success, extract_image = cap.read()
        print(fps, extract_image.shape)
        total_time=0
        
        for idx, row in df1.iterrows():
            count =0
            label= str(int(row['Label']))
            app= row['Appearance Block']
            st = datetime.strptime(row['Start Time'],"%H:%M:%S")
            start = int(timedelta(hours=st.hour, minutes=st.minute, seconds=st.second).total_seconds())
            en = datetime.strptime(row['End Time'],"%H:%M:%S")
            end = int(timedelta(hours=en.hour, minutes=en.minute, seconds=en.second).total_seconds())
            diff = end -start
            total_time+=diff
            print(total_time)
            
            while success and cap.get(cv2.CAP_PROP_POS_MSEC) < start*1000:
                success, extract_image = cap.read()

            while success and cap.get(cv2.CAP_PROP_POS_MSEC) <= end*1000:
                
                if count % (int(2*fps)) == 0: #half fps
                    size = extract_image.shape
                    resized_image = cv2.resize(extract_image, (224, 224))
                    print(app)
                    if app =="None" or app =="No":
                        d_folder= 'frame/synhalffps/'+ label+'/'+f"{front}_label_{label}_frame_no_{str(start+int(count/60))}"+'.jpg'
                        cv2.imwrite(d_folder, resized_image) 
                    else:
                        d_folder= 'frame/synhalffps/'+ label+'/'+f"{front}_label_{label}_frame_no_{str(start+int(count/60))}_SH"+'.jpg'
                        # cv2.imwrite(d_folder, resized_image) 
                        print("Hat/SH skipped")
                    
                    
                
                    print(d_folder)               
                    count +=1
                    success, extract_image = cap.read()
                else:
                    count +=1
                    success, extract_image = cap.read()
