

# Neural Graph-based Dependency parsing:
## A LSTM model to resolve ambiguity in natural language

By:
* **Koen Derks** (10518215)
* **Mirthe van Diepen** (10327428)
* **Lois van Vliet**(10438033) 


### Introduction
This research implements a neural graph-based dependency parser and its evaluation on two different languages. The project is guided by the model proposed in Kiperwasser and Goldberg, where they use a Long-Short Term Memory (LSTM) neural network to predict the graph-based dependancy arcs and labels.  

### Packages

The packages that were used in this project are:
* PyTorch
* Networkx
 

### Data

We used the data from the Universal Dependency [project](http://universaldependencies.org). The training sets for both English and Dutch consisted of 1000 sentences. 
The Test dataset consistend of 600 sentences.  

The read_data.py reads the CONLL-U files and makes dictionaries of the existing words. 

### Training

The code for training the LSTM neural network can be found in train.py. When you want to train your own LSTM neural network, you should follow the steps in the parser.ipynb

### Evaluate

To evaluate the correctness of our parser we used the UAS and the LAS scores. When you want to test out LSTM neural networks for both languages follow the instructions in parser.ipynb. 


This was a project for the course Natural Language processing at the University of Amsterdam.  
