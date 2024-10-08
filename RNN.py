import numpy as np
from RNNCell import RNNCell
from DenseLayer import Dense
from WordEmbedding import WordEmbedding
import time
import os
import winsound
np.random.seed(0)

class RNN:
    def __init__(self, units, denseLayers, neurons, learning_rate, epochs, temperature=0):
        self.units = units
        self.denseLayers = denseLayers
        self.neurons = neurons
        self.learning_rate = learning_rate
        self.temperature = temperature
        self.epochs = epochs

    def word_embeddings(self, inputfile):
        we = WordEmbedding(self.neurons)
        we.prep_text(inputfile)
        we.one_hot_encode()
        we.bigram()
        we.train()
        self.embeddings = we.extract_embeddings()
        self.vocab = we.vocab
        self.text = we.text
        self.one_hot = we.one_hot
        self.index = we.index
    
    def sentence_prep(self):
        self.sentence = []
        self.target = []
        for line in self.text:
            if len(line)<=self.units+1:
                self.sentence.append(line)
            elif len(line)>self.units+1:
                for i in range(self.units+1,len(line)):
                    self.sentence.append(line[i-(self.units+1):i])

    def cross_entropy_loss(self,y_pred, y_true):
        return -np.sum(y_true * np.log(y_pred + 1e-9))


    def compute(self):
        self.rnns = [None]*self.units
        self.dense = [None]*self.denseLayers
        # print(hs)
        for epoch in range(self.epochs):
            total_loss = 0
            step = 0
            for sentence in self.sentence:
                # print(sentence)
                for i in range(1,self.units+1):
                    hs = np.zeros((self.neurons,1))
                    if i>=len(sentence):
                        continue
                    # print("Training data:",sentence[:i])
                    target = sentence[i]
                    # print("Target:",target)
                    pred = None
                    for j in range(i):
                        inp = self.embeddings[self.vocab.get(sentence[j])].reshape(self.neurons,1)
                        if self.rnns[j]==None:
                            self.rnns[j] = RNNCell(inp=inp,predecessor=pred,hs=hs)
                        else:
                            self.rnns[j].inp = inp
                            self.rnns[j].hs = hs

                        hs = self.rnns[j].forward()
                        pred = self.rnns[j]

                    output = hs
                    #Dense Layers
                    for layer in range(len(self.dense)):
                        if self.dense[layer] is None:
                            if layer==self.denseLayers-1:
                                self.dense[layer] = Dense(output,len(self.vocab),True)
                            else:
                                self.dense[layer] = Dense(output,self.neurons,False)
                        else:
                            self.dense[layer].inp = output
                        output = self.dense[layer].forward()
                
                    # Calculate loss
                    total_loss+=self.cross_entropy_loss(output, self.one_hot.get(target))
                    step+=1
                    
                    gradient = output - self.one_hot.get(target)
                    for layer in reversed(self.dense):
                        gradient = layer.backward(gradient,self.learning_rate)
                    
                    for j in range(i-1,-1,-1):
                        gradient = self.rnns[j].backward(gradient,self.learning_rate)

            os.system('cls')
            print("RNN:\n\tAverage loss:",(total_loss/step))
            print("\tTraining epoch: ["+str(epoch)+"/"+str(self.epochs)+"]")

    def softmax(self, logits, temperature):
        temperature = max(0.01, min(temperature, 1.0))  # Clamp temperature to the range (0.01, 1.0)
        logits = logits / temperature  # Adjust logits based on temperature
        exp_logits = np.exp(logits - np.max(logits))  # For numerical stability
        return exp_logits / np.sum(exp_logits)

    def validate(self, validation_sentences):
        total_loss = 0
        step = 0
        for sentence in validation_sentences:
            hs = np.zeros((self.neurons, 1))
            for i in range(1, self.units + 1):
                if i >= len(sentence):
                    continue
                target = sentence[i]
                pred = None
                for j in range(i):
                    inp = self.embeddings[self.vocab.get(sentence[j])].reshape(self.neurons, 1)
                    if self.rnns[j] is None:
                        self.rnns[j] = RNNCell(inp=inp, predecessor=pred, hs=hs)
                    else:
                        self.rnns[j].inp = inp
                        self.rnns[j].hs = hs

                    hs = self.rnns[j].forward()
                    pred = self.rnns[j]

                output = hs
                for layer in self.dense:
                    layer.inp = output
                    output = layer.forward()

                total_loss += self.cross_entropy_loss(output, self.one_hot.get(target))
                step += 1
        
        return total_loss / step if step > 0 else float('inf')

    def autocomplete(self, sentence):
        self.sample = sentence.rstrip('\n').split(' ')
        generated = ""
        max_length = 50
        
        while generated != '<END>' and len(self.sample) < max_length:
            sample = self.sample[-self.units:] if len(self.sample) > self.units else self.sample
            hs = np.zeros((self.neurons, 1))
            
            for j in range(len(sample)):
                inp = self.embeddings[self.vocab.get(sample[j])].reshape(self.neurons, 1)
                self.rnns[j].inp = inp
                self.rnns[j].hs = hs
                hs = self.rnns[j].forward()
            
            output = hs
            for layer in self.dense:
                layer.inp = output
                output = layer.forward()
            
            if self.temperature!=0:
                probabilities = self.softmax(output.flatten(),self.temperature)  # Get probabilities from the output
                generated_index = np.random.choice(len(probabilities), p=probabilities)  # Sample based on probabilities
                generated = self.index.get(generated_index)
            else:
                generated = self.index.get(np.argmax(output))
            
            if generated == '<END>':
                break
            
            self.sample.append(generated)
            os.system('cls' if os.name == 'nt' else 'clear')
            print('SAMPLE:', sample)
            print(' '.join(self.sample))
            time.sleep(0.2)


    def check_weights(self,rnns,j):
        for x in range(1,j):
            if rnns[x].wh is rnns[x-1].wh:
                print('Weights',x,x-1,'are the same')
            else:
                print('Weights',x,x-1,'are NOT the same')

rnn = RNN(3,3,10,0.0007,3000,0)
x1 = time.time()
rnn.word_embeddings('training_data.txt')
print(time.time()-x1)
rnn.sentence_prep()
x1 = time.time()
rnn.compute()
print(time.time()-x1)
sent = ""
while sent!="stop":
    sent = input('Give a sentence to complete: ').lower()
    # os.system('cls' if os.name == 'nt' else 'clear')
    rnn.autocomplete(sent)