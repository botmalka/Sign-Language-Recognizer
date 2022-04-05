import os
import cv2

def read_files():
    file_list, sub_list = [], []
    start_dir = r"C:\Users\Tori\Documents\ASL Detector\asl_dataset"                                                                                               
    subdirs = [x[0] for x in os.walk(start_dir)]                                                                      
    for subdir in subdirs:       
        print(subdir)                                                                                     
        files = os.walk(subdir).__next__()[2]                                                                             
        if (len(files) > 0):                                                                                          
            for file in files:               
                #sub_list.append(str(subdir)[-1])                                                                       
                file_list.append(os.path.join(subdir, file)) 
                sub_list.append(str(subdir)[-1]) 
    zipped = zip(file_list, sub_list)
    return zipped

files_subs = read_files()
for files, sub in files_subs:
    print(files + " from: " + sub)

