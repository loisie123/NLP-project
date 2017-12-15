import torch
from read_data import *

def dependancy_parser(sentence):
    """
    This function parsed a sentence

    sentence: [[index_word_0, index_postag_0], ... , [index_word_n, index_postag_n]]
    """


    #print(sentence)

    #inladen neural network

    # sentence in neural network

    # output verwerkenen om in edmunds te gooien

    # output parser


    return True



def test_accuracy(datafile):
    """
    This function test the accuracy of our dependacy parser.

    """


    w2i, i2w, t2i, i2t, l2i, i2l = create_dictionaries("/Volumes/Transcend/studienu/2017:2018/NLP/project/NLP-project/en-ud-train_extract.conllu")

    #open de dataset.
    with open(datafile, 'r', newline='\n') as file_in:
        reader = csv.reader(file_in, delimiter="\t", quotechar=None)
        sentence_in = []
        tree = []
        goed_voorspelde_zinnen = 0
        goed_voorspelde_arcs = 0
        goed_voorspelde_labels = 0
        sentence = []
        word_count = 0

        #voor elke rij in de reader.
        for row in reader:
            #print(row)
            if len(row ) > 1: # als de rij groter is dan dit
                word = row[1]

                sentence.append(word)
                if word in w2i:
                    index_word = w2i[word]
                else:
                    word = "<unk>"
                    index_word = w2i[word]
                postag = row[3]
                index_postag = t2i[postag]
                sentence_in.append([index_word, index_postag])


                # sla het woord, postag, arc-in en label op
                #tree.append(word, postag, row[6], row[7])


            if len(row) == 0: # then the end of a line is found.
                output  = parser(sentence_in)
                #print(sentence)

                word_count += len(output)
                # vergelijk de uitkomst van het netwerk met de daadwerkelijke outkomts
                # if len(ouput) == len(tree):
                #     good, amount_good_arcs, boolean = compare(output, tree)
                #     goed_voorspelde_zinne += good
                #     goed_voorspelde_arcs += amount_good_arcs
                #     goed_voorspelde_labels += amount_good_labels
                # else:
                #     print("something went wrong")
                #
                #

                #nieuwe lijst voor nieuwe zin moet klaargezet worden
                sentence_in = []
                tree = []
                sentence = []

            #if len(row) == 0: #einde van de zin



    # remove file_out if it does exist

def compare(output, golden_tree):
    """
    this parser compares the predicted tree to the golden tree

    output: predicted output
    golden_tree:  the known tree. 
    """

    amount_good_arcs = 0
    amount_good_labels = 0
    total = 0
    amount_good = 0
    amount_wrong = 0
    amount_good_sentences


    for i in range(len(output)):
        total += 1
        out = output[i]
        tree_word = [i]
        if output[i]== tree[i]:
            amount_good += 1
        #arc is right
        elif out[2] == tree_word[2]:
            amount_good_arcs += 1
        # label is corect
        elif out[3] == tree_word[3]:
            amount_good_labels +=1
        else:
            amount_wrong += 0

    if amount_wrong+amount_good_labels+amount_good_arcs+amount_good != total:
        print("te weinig woorden ")

    if amount_good == total:
        good = 1
    else:
        good = 0

    return good, amount_good_labels, amount_good_arcs



test_accuracy('/Volumes/Transcend/studienu/2017:2018/NLP/project/NLP-project/en-ud-test.conllu')
