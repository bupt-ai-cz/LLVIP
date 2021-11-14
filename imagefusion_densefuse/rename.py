# -*- coding: utf-8 -*-
import os

path1 = "./LLVIP/visible/train/"
# path1 = "./LLVIP/infrared/train/"
path2 = "./datasets/"
filelist = os.listdir(path1) 

count=1
# for file in filelist:
    # print(file)
for file in filelist:   
    Olddir=os.path.join(path1,file)   
    if os.path.isdir(Olddir):   
        continue
    filename=os.path.splitext(file)[0]   
    filetype=os.path.splitext(file)[1]   
    newname="_vi"
    # newname="_ir"
    Newdir=os.path.join(path2,filename+newname+filetype) 
    os.rename(Olddir,Newdir)
    count+=1


