"""
Purpose: 
    Calculate distance and disruption index
Input:
    - -e numpy files containing embedding vectors based on citation network
    - -n citation network with sparse network datatype
    - -c configuration file
    - -v device name

Ouput:
    - Disruption_{embedding_file_namees}.npy : .npy file that contains disruption index of each paper
    - Distance.npy : .npy file that contains distance index of each paper

Author: Munjung Kim
"""  


import utils
import scipy
import node2vecs

import torch
import numpy as np
import pickle
import os
import argparse
import configparser
import sys
import logging
import tqdm

if __name__ == "__main__":
    
    MEASURE = sys.argv[1]
    EMBEDDING_IN = sys.argv[2]
    EMBEDDING_OUT = sys.argv[3]
    NETWORK = sys.argv[4]
    DEVICE = sys.argv[5]
    
    
    if MEASURE == 'disruption':
        net = scipy.sparse.load_npz(NETWORK)
        di = utils.calc_disruption_index(net, batch_size=None)
    
        
        NET_FOLDER = os.path.abspath(os.path.join(NETWORK, os.pardir))
        SAVE_DIR = os.path.join(NET_FOLDER,'disruption.npy')
        np.save(SAVE_DIR,di)
        
        

    elif MEASURE =='distance':

        logging.basicConfig(filename = 'Distance.log',level=logging.INFO, format='%(asctime)s %(message)s')

    
        # net = scipy.sparse.load_npz(NETWORK)
    
        in_vec = np.load(EMBEDDING_IN)
        out_vec = np.load(EMBEDDING_OUT)
        
        EMBEDDING_FOLDER = os.path.abspath(os.path.join(EMBEDDING_IN, os.pardir))
        
        in_vec_torch = torch.from_numpy(in_vec).to(DEVICE)
        out_vec_torch = torch.from_numpy(out_vec).to(DEVICE)

        n = len(out_vec_torch)

        distance= []
        
        batch_size = int(n/100) + 1
        
        logging.info('Starting calculating the distances')

        for i in tqdm.tqdm(range(100)):
            X = in_vec_torch[i*batch_size: (i+1)*batch_size]
            Y = out_vec_torch[i*batch_size: (i+1)*batch_size]
            numerator = torch.diag(torch.matmul(X,torch.transpose(Y,0,1)))
            norms_X = torch.sqrt((X * X).sum(axis=1))
            norms_Y = torch.sqrt((Y * Y).sum(axis=1))

            denominator = norms_X*norms_Y


            cs = 1 - torch.divide(numerator, denominator)
            distance.append(cs.tolist())
        
        distance_lst =  np.array([dis for  sublist in distance for dis in sublist])
        
        
        logging.info('Saving the files.')
        
        SAVE_DIR = os.path.join(EMBEDDING_FOLDER,'distance.npy')
        np.save(SAVE_DIR, distance_lst)
    
    
   
    
        
    
        
        
    
        
        
    
    
    