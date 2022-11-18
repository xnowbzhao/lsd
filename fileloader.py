import os
import numpy as np
def read_filename(data_path):
    data_filename = []
    for root, dirs, files in os.walk(data_path):
        data_filename = files

        data_filename = [root + filename for filename in data_filename]
    data_filename = sorted(data_filename)
    # print(len(data_filename))

    return data_filename


class Loader():
    def __init__(self, dataset_folder, batchsize):
        namelist = read_filename(dataset_folder)
        self.train_num = int(len(namelist)//2)
        self.train_filename=namelist[0:self.train_num]
        self.label_filename=namelist[self.train_num :self.train_num*2]
        self.batchsize=batchsize
        
    def length(self):
        return self.train_num
    def generate_batch(self, idx):
        train_data = np.fromfile(self.train_filename[idx], dtype='float32').reshape((-1,80,80,3))
        train_label = np.fromfile(self.label_filename[idx], dtype='float32').reshape((-1,3))
        case_number=train_data.shape[0]
        batch_number=case_number//self.batchsize
        train_data=train_data[0:batch_number*self.batchsize].reshape((batch_number,self.batchsize,80,80,3))
        train_label=train_label[0:batch_number*self.batchsize].reshape((batch_number,self.batchsize,3))

        return train_data, train_label
    