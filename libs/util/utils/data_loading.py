import pandas as pd
import os
import numpy as np


def load_papers(filename):
    papers = []

    for i, chunk in enumerate(pd.read_csv(filename, chunksize=10000)):
        papers.append(chunk)

    papers_df = pd.concat(papers)   
    return papers_df

def load_disruption_distance(pandas_papers, NETWORKS_NAME, MEASURE ):
    MEASURE_FILE = MEASURE +'.npy'
    # MEASURE_FILENAME = os.path.join('/data/sg/munjkim/APS/',NETWORKS_NAME,MEASURE_FILE)
    MEASURE_FILENAME = os.path.join('../data/',NETWORKS_NAME,MEASURE_FILE)
    try:
        measure = np.load(MEASURE_FILENAME)
        MEASURE = MEASURE.replace('/','_')
        pandas_papers[NETWORKS_NAME+'_'+MEASURE] = pandas_papers['paper_id'].apply(lambda x: measure[x])
    
        pandas_papers['PCNT_RANK_'+NETWORKS_NAME+'_'+MEASURE]=pandas_papers[NETWORKS_NAME+'_'+MEASURE].rank(pct=True)
    

 

  
        return pandas_papers
    except:
        print("no such directory:", MEASURE_FILENAME)
        return pandas_papers

    
def load_distance_restricted(pandas_papers, NETWORKS_NAME, MEASURE ):
    MEASURE_FILE = MEASURE +'.npy'
    MEASURE_FILENAME = os.path.join('../data',NETWORKS_NAME,MEASURE_FILE)
    NODE_NAME_FILE = os.path.join('../data',NETWORKS_NAME,'node_name.npy')
    
    try:
        measure = np.load(MEASURE_FILENAME)
        MEASURE = MEASURE.replace('/','_')
        node_name = np.load(NODE_NAME_FILE)
        node_name_set = set(node_name)
        
        
        pandas_papers[NETWORKS_NAME+'_'+MEASURE] = pandas_papers['paper_id'].apply(lambda x: measure[np.where(node_name  == x)[0][0]] if x in node_name_set  else np.nan)
    
        pandas_papers['PCNT_RANK_'+NETWORKS_NAME+'_'+MEASURE]=pandas_papers[NETWORKS_NAME+'_'+MEASURE].rank(pct=True)
    

 

  
        return pandas_papers
    except:
        print("no such directory:", MEASURE_FILENAME)
        return pandas_papers
