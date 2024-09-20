from DenseLayer import Dense
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

learning_rate = 0.1

class WordEmbedding:
    def __init__(self, neurons):
        self.neurons = neurons

    def prep_text(self,inputfile):
        inpfile  = open(inputfile,'r')
        text = inpfile.readlines()
        newtext = []
        for chunk in text:
            chunk = chunk.lower()
            chunk = chunk.replace('\n','')
            newtext+=chunk.split('.')
        newtext = [sentence for sentence in newtext if sentence != '']
        self.text = []
        for lines in newtext:
            lines += ' <END>'
            self.text.append(lines.split(' '))
        self.text = [[item for item in sublist if item != ''] for sublist in self.text]
        # print(self.text)

        self.vocab = {}
        self.index = {}
        index = 0
        for line in self.text:
            for word in line:
                # word = word.lower()
                if word not in self.vocab:
                    self.vocab[word] = index
                    self.index[index] = word
                    index+=1
        self.vocab_size = index
        
    def one_hot_encode(self):
        self.one_hot = {}
        for word in self.vocab.keys():
            one_hot = np.zeros((self.vocab_size,1))
            one_hot[self.vocab[word]] = 1
            self.one_hot[word] = one_hot
    
    def bigram(self):
        bigram = []
        for line in self.text:
            for i in range(1,len(line)):
                # bigram.append([line[i],line[i-1]])
                bigram.append([line[i-1],line[i]])
        self.bigram = bigram

    def train(self):
        self.layers = [None,None,None]
        for pair in self.bigram:
            inp = self.one_hot.get(pair[0])
            out = self.one_hot.get(pair[1])
            for i in range(3):
                if self.layers[i] is None:
                    if i==2:
                        self.layers[i] = Dense(inp,self.vocab_size,True)
                    else:
                        self.layers[i] = Dense(inp,self.neurons,False)
                else:
                    self.layers[i].inp = inp
                inp = self.layers[i].forward()
            
            # Derivative of Cross-entropy loss w.r.t. input of softmax
            gradient = inp - out
            for i in reversed(range(3)):
                gradient = self.layers[i].backward(gradient, learning_rate)

    def extract_embeddings(self):
        # Assuming that the first layer's weights are used as word embeddings
        return self.layers[0].weights.T  # Shape (vocab_size, embedding_dim)

    def plot_embeddings(self):
        # Extract embeddings
        embeddings = self.extract_embeddings()

        # Reduce dimensionality to 2D using PCA
        pca = PCA(n_components=2)
        reduced_embeddings = pca.fit_transform(embeddings)

        # Plot
        plt.figure(figsize=(10, 10))
        for word, index in self.vocab.items():
            plt.scatter(reduced_embeddings[index, 0], reduced_embeddings[index, 1])
            plt.annotate(word, (reduced_embeddings[index, 0], reduced_embeddings[index, 1]), fontsize=9)

        plt.title('Word Embeddings Visualization')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.grid(True)
        plt.show()

# Usage:
# x = WordEmbedding(10)
# x.prep_text('training_data.txt')
# x.one_hot_encode()
# x.bigram()
# x.train()
# x.plot_embeddings()



# x = WordEmbedding()
# x.prep_text('training_data.txt')
# x.one_hot_encode()
# x.bigram()
# x.train()