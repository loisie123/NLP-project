import torch
from torch.autograd import Variable
import torch.nn as nn

from read_data import *
dtype = torch.FloatTensor


class BiLSTM(nn.Module):

    def __init__(self, w2i, i2w, t2i, i2t, l2i, i2l):
        """
        Initialize bidirectional LSTM RNN.
            :param w2i: dictionary word to index
            :param i2w: dictionary index to word
            :param t2i: dictionary POS-tag to index
            :param i2t: dictionary index to POS-tag
            :param l2i: dictionary label to index
            :param i2l: dictionary index to label
        
        """
        super(BiLSTM, self).__init__()
        
        self.w2i = w2i
        self.i2w = i2w
        self.t2i = t2i
        self.i2t = i2t
        self.l2i = l2i
        self.i2l = i2l
        
        voca_size = len(self.w2i)
        tag_size = len(self.t2i)

        word_emb_dim = 100
        tag_emb_dim = 25
        
        label_emb_dim = 20

        self.word_emb = nn.Embedding(voca_size, word_emb_dim)
        self.tag_emb =nn.Embedding(tag_size, tag_emb_dim)
        # deze moet nog worden toegevoegd
        # pretrained_weight is a numpy matrix of shape (num_embeddings, embedding_dim)
        # embed.weight.data.copy_(torch.from_numpy(pretrained_weight))

        self.input_dim = word_emb_dim + tag_emb_dim
        self.output_dim = label_emb_dim + 1
        self.hidden_dim = 100 # number of nodes in hidden layers
        self.num_layer = 2 # number of hidden layers
        
        # HOE MOET BIDIRECTIONAL????
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layer)#, bidirectional=True)
        self.hidden2output = nn.Linear(self.hidden_dim, self.output_dim)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        """
        Initialize hiddden layers of dimension (num_layers, minibatch_size, hidden_dim).
        """
        return (Variable(torch.zeros(self.num_layer, 1, self.hidden_dim)),
                Variable(torch.zeros(self.num_layer, 1, self.hidden_dim)))

    def forward(self, sentence):
        """
        Forward step:
            :param sentence: a list of lists containing index of word and index of POS-tag
        """
        softmax = nn.Softmax()
        emb = self.concatenate_emb(sentence)
        hidden_output, self.hidden = self.lstm(emb.view(len(sentence), 1, -1), self.hidden)
        output = self.hidden2output(hidden_output.view(len(sentence), -1))
        normalized_output = softmax(output)
        return normalized_output

    def concatenate_emb(self, sentence):
        """
        Concatenate the word and POS-tag embedding.
            :param sentence: a list of lists containing index of word and index of POS-tag
        """
        for i, word in enumerate(sentence):
            word_emb = self.word_emb(word[0])
            tag_emb = self.tag_emb(word[1])
            if i == 0:
                emb = torch.cat([word_emb.view(-1), tag_emb.view(-1)])
            else:
                emb = torch.cat([emb.view(-1), torch.cat([word_emb.view(-1), tag_emb.view(-1)])])
        return emb


def train(path, lr, epochs):
    """
    Create and train a bidirectional LSTM RNN.
        :param path: path to files
        :param lr: learning rate
        :param epochs: number of epochs
    """
    
    w2i, i2w, t2i, i2t, l2i, i2l = create_dictionaries(path)
    
    # create model
    model = BiLSTM(w2i, i2w, t2i, i2t, l2i, i2l)
    
    # make data ready for use
    train_input = []
    train_target = []
    data_dim = 0
    with open(path, 'r', newline='\n') as file:
        reader = csv.reader(file, delimiter='\t', quotechar=None)
        # create list of sentences
        sentence_in = []
        sentence_out = []
        for row in reader:
            if data_dim > 10: break
            if row[0].split(" ")[0] != "#":
                # each sencente is list of words
                # each word is represented by values:
                # index word, index POS-tag (sentence in)
                # ARC-in, index label (sentence out)
                sentence_in.append([w2i[row[1]],t2i[row[2]]])
                sentence_out.append([int(row[3]),l2i[row[4]]])
            elif data_dim > 0:
                # new sentence start
                # put previous sentence is list of sentences
                train_input.append(Variable(torch.LongTensor(sentence_in)))
                train_target.append(Variable(torch.LongTensor(sentence_out)))
                sentence_in = []
                sentence_out = []
                data_dim += 1
            else:
                # start of first sentence
                data_dim = 1
        train_input.append(Variable(torch.LongTensor(sentence_in)))
        train_target.append(Variable(torch.LongTensor(sentence_out)))

    
    # define loss function: cross entropy loss
    cross_entropy_loss = nn.CrossEntropyLoss()

    # define optimize method: stochastic gradient descent 
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for i in range(epochs):
        
        # sentence and target are matrix
        for j in range(data_dim):
            
            # zero gradient buffers
            optimizer.zero_grad()

            # clear hidden
            model.hidden = model.init_hidden()
            
            # find output of network
            output = model(train_input[j])

            # error of output and target
            loss = cross_entropy_loss(output, train_target[j])

            # backpropagate the error
            loss.backward()

            # update the weights
            optimizer.step()

    return model