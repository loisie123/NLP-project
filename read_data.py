import csv
#import numpy as np
#import pandas as pd
#from tqdm import tqdm
#import time
from collections import defaultdict, Counter
import os


#def read_embedding(path):
#    word_embedding = pd.DataFrame(pd.read_csv(path, sep=" ", header=None, quotechar = "'"))
#    return word_embedding


# dit kan getest worden met de evalution algoritme
def create_CONNL_U(path_in, path_out):
    """
    Create a CONNL-U file with usefull data.
    Input:
        :param path_in: path to file with data from ... 
        :param path_out: path to file for useful data
        
    """
    
    with open(path_in, 'r', newline='\n') as file_in:
        
        # remove file_out if it does exist
        if os.path.isfile(path_out) :
            os.remove(path_out)
        
        # create file_out
        with open(path_out, 'w') as file_out:            
            
            # read data
            reader_in = csv.reader(file_in, delimiter='\t', quoting=csv.QUOTE_ALL)
            
            
            # create word counter
            word_count = Counter()
            for row in reader:
                if len(row)>1:
                     word_count[row[1]] += 1
            
            # write data in CONNL-U file
            last_row = ''
            for row in reader:
                if len(row) != 0:
                    if row[0].split(' ')[0] != '#' and len(row[0].split('.')) == 1 and len(row[0].split('-')) == 1:
                        # write whole sentence
                        if last_row[0].split(' ')[0] == '#' and last_row[0].split(' ')[1] == 'text':
                            file_out.write(last_row + '\n')
                        word = row[1]
                        if word_count[word] == 1:
                            word = '<unk>'
                        file_out.write('{}\t{}\t{}\t{}\t{}\n'.format(row[0], word, row[3], row[6], row[7]))
                last_row = row


def create_dictionaries(path):
    """
    Create dictionaries with words from the CONNL-U file with useful information.
        :param path: path to file with useful information
    
    """
    
    # create dictionaries
    w2i = defaultdict(lambda: len(w2i))
    i2w = dict()
    t2i = defaultdict(lambda: len(w2i))
    i2t = dict()
    l2i = defaultdict(lambda: len(w2i))
    i2l = dict()
    
    with open(path, 'r', newline='\n') as file:
        reader = csv.reader(file, delimiter='\t', quoting=csv.QUOTE_ALL)
        for row in reader:
            if row[0].split(" ")[0] != "#":
                # fill in all information to dictionaries
                i2w[w2i[row[1]]] = row[1]
                i2t[t2i[row[2]]] = row[2]
                i2l[l2i[row[4]]] = row[4]
             
    # stop defaultdict behavior 
    w2i = dict(w2i)
    t2i = dict(t2i)
    l2i = dict(l2i)

    return w2i, i2w, t2i, i2t, l2i, i2l







