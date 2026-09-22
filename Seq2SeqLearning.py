import torch 
import torch.nn as nn
from torchtext.data import Field
from datasets import load_dataset
from collections import Counter
from torch.nn.utils.rnn import pad_sequence

#-------------DATA----------------------------------------
dataset = load_dataset("cfilt/iitb-english-hindi")
# print(dataset.shape, type(dataset))

#print(dataset['train'].shape, dataset['train'][0], type(dataset['train'][0]), type(dataset['train'][0]))
small_train = dataset["train"].select(range(10000))

english = []
hindi = []

for item in small_train:
    english.append(item['translation']['en'])
    hindi.append(item['translation']['hi'])

print(english[0], hindi[0])

#------------------


eng_vocab = {"<PAD>": 0,"<SOS>": 1,"<EOS>": 2,"<UNK>": 3}
hindi_vocab = {"<PAD>": 0,"<SOS>": 1,"<EOS>": 2,"<UNK>": 3}

def convert():
    counter = Counter()
    counter2 = Counter()

    for sentence in english:
        counter.update(sentence.split())

    for sentence in hindi:
        counter2.update(sentence.split())
    for word in counter2:
        hindi_vocab[word] = len(hindi_vocab)
    for word in counter:
        eng_vocab[word] = len(eng_vocab)

def sent_hindi(sentence):
    tokens = sentence.split()

    hindi_ids = [hindi_vocab["<SOS>"]]

    hindi_ids.extend(
        hindi_vocab.get(word, hindi_vocab["<UNK>"])
        for word in tokens
    )

    hindi_ids.append(hindi_vocab["<EOS>"])
    return hindi_ids

def sent_english(sentence):
    tokens = sentence.lower().split()

    eng_ids = [eng_vocab["<SOS>"]]

    eng_ids.extend(
        eng_vocab.get(word, eng_vocab["<UNK>"])
        for word in tokens
    )

    eng_ids.append(eng_vocab["<EOS>"])
    return eng_ids

def padding_tensor():
    eng_tensor = [torch.tensor(item) for item in english_sent]
    hindi_tensor = [torch.tensor(item) for item in hindi_sent]

    eng_padded = pad_sequence(
        eng_tensor,
        batch_first=True,
        padding_value=eng_vocab["<PAD>"]
    )

    hin_padded = pad_sequence(
        hindi_tensor,
        batch_first=True,
        padding_value=hindi_vocab["<PAD>"]
    )
    return eng_padded, hin_padded

convert()
print(len(eng_vocab), len(hindi_vocab))


# print(sent_english("Give your application an accessibility workout"))
# print(sent_hindi("अपने अनुप्रयोग को पहुंचनीयता व्यायाम का लाभ दें"))

english_sent = [sent_english(i) for i in english]
hindi_sent = [sent_hindi(i) for i in hindi]

print(english_sent[:10], hindi_sent[:10])

maxEnglish = max(len(item) for item in english_sent)
maxHindi = max(len(item) for item in hindi_sent)

print(maxEnglish, maxHindi)

english_sent, hindi_sent = padding_tensor()
# for i in range(10):
#     print(english_sent[i], hindi_sent[i])
    

    
    
#-------------DEVICE------------------------------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

#---------------MODEL BUILDING------------
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, embedding_size, dp):
        super(Encoder, self).__init__()
        
        self.hidden = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, embedding_dim=embedding_size)
        
        self.dropout = nn.Dropout(dp)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers=num_layers, dropout=dp, bidirectional=True)
    
    def forward(self, x):
        
        embedding=self.dropout(self.embedding(x))
        output, (hidden, cell) = self.rnn(embedding)
        
        return hidden, cell
        
    
class Decoder(nn.Module):
    def __init__(self):
        pass
    def forward(self):
        pass
    
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        pass
    def forward(self):
        pass

#------------------PARAMETERS--------------------

learning_rate = 0.01
epochs = 25
optimzer = None
criterion = nn.CrossEntropyLoss()

#-------------------TRAINING---------------------
for ep in range(epochs):
    pass



#------------------VALIDATION---------------------