import torch 
import torch.nn as nn
from torchtext.data import Field
from datasets import load_dataset
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torch import optim
import random
import time

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
print(f"eng vocab:{len(eng_vocab)}, hindi vocab:{len(hindi_vocab)}")


# print(sent_english("Give your application an accessibility workout"))
# print(sent_hindi("अपने अनुप्रयोग को पहुंचनीयता व्यायाम का लाभ दें"))

english_sent = [sent_english(i) for i in english]
hindi_sent = [sent_hindi(i) for i in hindi]

#print(english_sent[:10], hindi_sent[:10])

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

#---------------ENCODER-----------------
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, embedding_size, dp):
        super(Encoder, self).__init__()
        
        self.hidden = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, embedding_dim=embedding_size)
        
        self.dropout = nn.Dropout(dp)
        self.rnn = nn.LSTM(embedding_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dp, batch_first=True)
    
    def forward(self, x):
        
        embedding=self.dropout(self.embedding(x))
        output, (hidden, cell) = self.rnn(embedding)
        
        # print(f"embedding shape:{embedding.shape}")
        

        return hidden, cell
        
'''
-------------------TESTING SIZE/DIMENSIONS--------------------------
class EnglishDatasets(Dataset):
    
    def __init__(self, english):
        self.eng_data = english
    
    def __len__(self):
        return  self.eng_data.shape[0]       

    def __getitem__(self, index):
        return self.eng_data[index]
    
dataset = EnglishDatasets(english_sent)
loader = DataLoader(dataset, batch_size=100, shuffle=True)

ed = Encoder(input_size=len(eng_vocab), hidden_size=512, num_layers=2, embedding_size=256, dp=0.2)

for batch in loader:
    print(f"batch shape: {batch.shape}")
    hideenn, cell = ed(batch)
    print(f"hidden shape:{hidden.shape}")
    print(f"cell shape:{cell.shape}")
    break
'''
#------------------DECODER-------------------------
class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_size, num_layers, dp, output_size):
        super(Decoder, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.embedding = nn.Embedding(input_size, embedding_dim=embedding_size)
        self.dropout = nn.Dropout(dp)
        self.rnn = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dp, batch_first=True)
        
        self.fc = nn.Linear(hidden_size, output_size)
        
        
    def forward(self, x, hidden, cell):

        x = x.unsqueeze(1)

        embedding = self.dropout(
            self.embedding(x)
        )

        output, (hidden, cell) = self.rnn(
            embedding,
            (hidden, cell)
        )

        prediction = self.fc(output)

        prediction = prediction.squeeze(1)


        return prediction, hidden, cell


#-------------------TESTING SIZE/DIMENSIONS--------------------------
'''
class EnglishDatasets(Dataset):
    
    def __init__(self, english):
        self.eng_data = english
    
    def __len__(self):
        return  self.eng_data.shape[0]       

    def __getitem__(self, index):
        return self.eng_data[index]
    
dataset = EnglishDatasets(english_sent)
loader = DataLoader(dataset, batch_size=100, shuffle=True)

dd = Decoder(input_size=len(hindi_vocab), hidden_size=512, num_layers=2, embedding_size=256, dp=0.2, output_size=len(hindi_vocab)
)
ed = Encoder(input_size=len(eng_vocab), hidden_size=512, num_layers=2, embedding_size=256, dp=0.2)

for batch in loader:
    print(f"batch shape: {batch.shape}")
    hd, cll = ed(batch)
    output, hidden, cell = dd(batch, hd, cll)
    print(f"output shape: {output.shape}")
    print(f"hidden shape:{hidden.shape}")
    print(f"cell shape:{cell.shape}")
    break

    
'''
    
#-------------------------S2S-----------------------------------
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_force_ratio=0.5):
        batch_size = source.shape[0]
        target_size = target.shape[1]
        
        outputs = torch.zeros(
            batch_size,
            target_size,
            len(hindi_vocab),
            device=device
        )
        
        hidden, cell = self.encoder(source)
        
        x = target[:, 0]
        
        for i in range(1, target_size):
            
            output, hidden, cell = self.decoder(x, hidden, cell)
        
            outputs[:, i, :] = output
            
            best_guess = output.argmax(1)
            
            x = target[:,i] if random.random() < teacher_force_ratio else best_guess
            
        return outputs
        

#------------------PARAMETERS--------------------

learning_rate = 0.01
epochs = 25
batch_size = 100

input_size_encoder = len(eng_vocab)
input_size_decoder = len(hindi_vocab)
embedding = 256
output_size = len(hindi_vocab)

hidden_size = 512
num_layers = 2

dp = 0.5
#-------------------MODEL DEFINING---------------

encoder_model = Encoder(
    input_size=input_size_encoder, num_layers=num_layers, 
    dp=dp, embedding_size=embedding, hidden_size=hidden_size
)
decoder_model = Decoder(
    input_size=input_size_decoder, num_layers=num_layers, dp=dp, 
    embedding_size=embedding, hidden_size=hidden_size,
    output_size=output_size
)

model = Seq2Seq(encoder=encoder_model, decoder=decoder_model).to(device)
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

pad_idx = hindi_vocab['<PAD>']
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)


#-------------------TRAINING---------------------

    #data prepairing

class TrainDataset(Dataset):
    def __init__(self, source, target):
        self.source = source
        self.target = target
        
    def __len__(self):
        return len(self.source)
    
    def __getitem__(self, index):
        return self.source[index], self.target[index]  
    
train_dataset = TrainDataset(english_sent, hindi_sent)
loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

for ep in range(epochs):

    model.train()

    total_loss = 0

    for src, trg in loader:
        start = time.time()


        src = src.to(device)
        trg = trg.to(device)

        optimizer.zero_grad()

        outputs = model(src, trg)

        outputs = outputs[:, 1:, :]
        target = trg[:, 1:]

        outputs = outputs.reshape(-1, outputs.shape[2])
        target = target.reshape(-1)

        loss = criterion(outputs, target)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1
        )

        optimizer.step()

        total_loss += loss.item()
        print(loss.item())
        '''
                # print("src:", src.shape)
                        # print("trg:", trg.shape)
                
                        hidden, cell = encoder_model(src)
                
                        # print("encoder hidden:", hidden.shape)
                        # print("encoder cell:", cell.shape)
                
                        x = trg[:,0]
                
                        prediction, hidden, cell = decoder_model(x, hidden, cell)
                        
                
                        #print("prediction:", prediction.shape)
                '''
    print("Forward time:", time.time() - start)

    print(f"Epoch {ep+1}: {total_loss/len(loader):.4f}")
    
    break



#------------------VALIDATION---------------------