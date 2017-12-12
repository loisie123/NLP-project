import torch
from torch.autograd import Variable
import torch.nn as nn

from read_data import *

dtype = torch.FloatTensor



# probeer LSTMTAGger te snappen zodat je de embedding en lstm kan combineren
# zorg dat data in goede format is
# updates met embeddings moeten gefixed worden
# read_data moet getest worden
# edmonds moet getest worden



# dit heb ik gekopieerd.. 
class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)#, 2)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        
        # willen wij dit anders? Maakt hij niet zelf al die hidden states aan...
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(
            embeds.view(len(sentence), 1, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space)
        return tag_scores



#>>> rnn = nn.LSTM(10, 20, 2)
#>>> input = Variable(torch.randn(5, 3, 10))
#>>> h0 = Variable(torch.randn(2, 3, 20))
#>>> c0 = Variable(torch.randn(2, 3, 20))
#>>> output, hn = rnn(input, (h0, c0))


def concentrateData():
    



def train(path, lr, iterations):
    """
    Create neural netwerk and train it.
    Input:
        :param train_data: nxm np.array
        :param lr: 
        :param iterations:
    Output:
        neural netwerk
    """
    
    w2i, i2w, t2i, i2t, l2i, i2l = create_dictionaries(path)
    
    voca_size = len(w2i)
    tag_size = len(t2i)
    
    word_emb_dim = 100
    tag_emb_dim = 25
    label_emb_dim = 20
    
    word_emb = nn.Embedding(voca_size, word_emb_dim)
    tag_emb =nn.Embedding(tag_size, tag_emb_dim) 
    
    input_size = word_emb_size + tag_emb_size
    output_size = label_emb_size + 1
    hidden_size = 100 # number of nodes in hidden layers
    num_layer = 2 # number of hidden layers
    lstm_net = nn.LSTM(input_size, hidden_size, num_layer)

    # GEBRUIK DICTIONARY OM MATRIX TE MAKEN
    
    # make data ready for use
    data_in = []
    data_out = []
    with open(path, 'r', newline='\n') as file:
        reader = csv.reader(file, delimiter='\t', quoting=csv.QUOTE_ALL)
        sentence = []
        for row in reader:
            if row[0].split(" ")[0] != "#":
                # hier moeten nog tensors van gemaakt worden..
                sentence_in.append(torch.cat([w2i[row[1]],t2i[row[2]]]))
                sentence_out.append(torch.cat([row[3],l2i[row[4]]]))
            else:
                data_in.append(sentence_in)
                data_out.append(sentence_out)
                sentence_in = []
                sentence_out = []
                

    # moet dit nog in het juiste format gezet worden?
    train_input = Variable(torch.FloatTensor(data_in).type(dtype), requires_grad=False)
    train_target = Variable(torch.FloatTensor(data_out).type(dtype), requires_grad=False)
    
    """
    # input data is matrix met als rijen de vector word embedding en postag embedding onder elkaar
    # output data is matrix met als rijen number incoming arc en vector label embedding onder elkaar geplakt.
    
    input_size = word_emb_size + tag_emb_size
    output_size = label_emb_size + 1
    # in this case len(train_data[i]) = 125+21
    hidden_size = 100 # number of nodes in hidden layers
    num_layer = 2 # number of hidden layers
    
    #lstm_net = LSTMTagger(input_size, hidden_size, voca_size, tag_size)
    lstm_net = nn.LSTM(input_size, hidden_size, num_layer)
    
    # make data ready for use
    in_data = []
    out_data = []
    for matrix in train_data:
        for row in matrix:
            # afhankelijk van hoe Koen de data in de matrix zet...
            in_data.append(row[:125])
            out_data.append(row[125:])
    # ze gebruiken ergens LongTensor ipv FloatTensor.. is dat beter?
    train_input = Variable(torch.FloatTensor(in_data).type(dtype), requires_grad=False)
    train_target = Variable(torch.FloatTensor(out_data).type(dtype), requires_grad=False)
    #train_input = Variable(torch.FloatTensor(train_data[:,:3]).type(dtype), requires_grad=False)
    #train_target = Variable(torch.FloatTensor(train_data[:,-1]).type(dtype), requires_grad=False)
    """
    # define loss function: mean squared error
    cross_entropy_loss = nn.CrossEntropyLoss()

    # define optimize method: stochastic gradient descent
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)

    for i in range(iterations):
        # sentence and target are matrix
        for train_sentence, train_target in train_input, train_output:
            
            # zero gradient buffers
            optimizer.zero_grad()

            # clear hidden
            model.hidden = model.init_hidden()
            
            # find output of network
            train_output = lstm_net(train_sentence)

            # error of output and target
            loss = cross_entropy_loss(train_output, train_target)

            # backpropagate the error
            loss.backward()

            # update the weights
            optimizer.step()

    return lstm_net



