#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 16:15:14 2023

@author: kaloso
"""


import numpy as np
#import Data_All as dl

data = np.load('dwt1.npy')

# # #data_all = [np.load(f'dwt{i}.npy') for i in range(1, 18)]

# # reshaped_data = np.transpose(data, (2, 0, 1)).reshape(153601, -1)

reshaped_data = data.reshape(data.shape[0], -1)

if __name__ == "__main__":
    
    np.save('data1.npy', reshaped_data)
    
# data1 = np.load('data1.npy')

# data2 = np.load('data2.npy')

# data3 = np.load('data3.npy')

# data4 = np.load('data4.npy')

# data5 = np.load('data5.npy')

# data6 = np.load('data6.npy')

# data7 = np.load('data7.npy')

# data8 = np.load('data8.npy')

# data9 = np.load('data9.npy')

# data10 = np.load('data10.npy')

# data11 = np.load('data11.npy')

# data12 = np.load('data12.npy')

# data13 = np.load('data13.npy')

# data14 = np.load('data14.npy')

# data15 = np.load('data15.npy')

# data16 = np.load('data16.npy')

# data17 = np.load('data17.npy')   

# data_all = np.concatenate((data1,data2,data3,data4,data5,data6,
#                            data7, data8,data9,data10,data11,data12,
#                            data13,data14,data15,data16,data17),axis=1) 
    
  
# if __name__ == "__main__":
    
#     np.save('data_all.npy', data_all)



# %%





