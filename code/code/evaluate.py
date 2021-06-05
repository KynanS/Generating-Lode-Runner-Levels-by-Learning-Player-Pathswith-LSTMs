#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import numpy as np
import matplotlib.pyplot as plt
levels = []#list of dictionaries, each dictionary a level

#Load generated Levels
for levelFile in glob.glob("./generated_levels/annotated/*.txt"):
	print ("Processing: "+levelFile)
	with open(levelFile) as fp:
		level = {}
		y = 0
		for line in fp:
			level[y] = line
			y+=1
		levels.append(level)
size=[] 
strlist=''       
samples=len(levels)
num_space=np.zeros(samples)
num_roi=np.zeros(samples)
p_space=np.zeros(samples)
p_roi=np.zeros(samples)
# compute the proportion of empty space and interesting tiles 
for i in range(samples):
    level=levels[i]
    height=len(level)
    width=len(level[0])-1
    size.append(np.array([height,width]))
    for j in range(height):
        strlist=strlist+level[j].replace('\n','')
        for k in level[j]:
            if k=='.':
                num_space[i]=num_space[i]+1
            if k=='-' or k=='G' or k=='E' or k=="#" or k=='M':
                num_roi[i]=num_roi[i]+1
    p_space[i]=num_space[i]/(height*width)
    p_roi[i]=num_roi[i]/(height*width)                      
strs=list(set(strlist))  

levels = []#list of dictionaries, each dictionary a level

#Load the original Levels
for levelFile in glob.glob("./TheVGLC-master/Lode Runner/Processed/*.txt"):
	print ("Processing: "+levelFile)
	with open(levelFile) as fp:
		level = {}
		y = 0
		for line in fp:
			level[y] = line
			y+=1
		levels.append(level)
      
samples=len(levels)
num_space=np.zeros(samples)
num_roi=np.zeros(samples)
p_space1=np.zeros(samples)
p_roi1=np.zeros(samples)
# compute the proportion of empty space and interesting tiles 
for i in range(samples):
    level=levels[i]
    height=len(level)
    width=len(level[0])-1
    for j in range(height):
        for k in level[j]:
            if k=='.':
                num_space[i]=num_space[i]+1
            if k=='-' or k=='G' or k=='E' or k=="#" or k=='M':
                num_roi[i]=num_roi[i]+1
    p_space1[i]=num_space[i]/(height*width)
    p_roi1[i]=num_roi[i]/(height*width)

plt.figure()
plt.hist(p_space1)
plt.hist(p_space)
plt.legend(('Original levels', 'Generated levels'))
plt.xlabel('Proportion')
plt.ylabel('Counts')
plt.show()