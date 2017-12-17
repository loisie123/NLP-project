

# Neural Graph-based Dependency parsing:
## A LSTM model to resolve ambiguity in natural language

By:
* **Koen Derks** (10518215)
* **Mirthe van Diepen** (10327428)
* **Lois van Vliet**(10438033) 


### Introduction
This research implements a neural graph-based dependency parser and its evaluation on two different languages. The projects build on the model proposed in Kiperwasser and Goldberg, where we use a Long-Short Term Memory (LSTM) neural network. 

### Packages

The packages that are used in this research are:
* PyTorch
* Networkx
 

### Data

We used the data from the Universal Dependency [project](http://universaldependencies.org). The training sets for both English and Dutch consisted of 1000 sentences. 
The Test dataset consistend of 600 sentences.  

The read_data.py reads in the CONLLU files and makes dictionaries of the existing words. 

### Training

The code for training the LSTM neural network can be found in train.py. To start training your own neural network you should run the train_data.ipytb

### Evaluate

To evaluate the correctness of our parser we used the UAS and the LAS scores. These are calculated in parser.ipytb 
To run this evaluation for English you should run in the notebook:
`test_accuracy("path-to-test-file", "path-to-network")`



### parsing 

If you want to parse a sentences you should run in the notebook

(THIS IS NOG NIET MOGELIJK)

