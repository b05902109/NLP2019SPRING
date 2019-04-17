import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # Set initial hidden and cell states 
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

class Net(nn.Module):
    def __init__(self, n_vocab, embedding_dim, hidden_dim, dropout=0.2):
        super(Net, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        # nn.Embedding 可以幫我們建立好字典中每個字對應的 vector
        self.embeddings = nn.Embedding(n_vocab, embedding_dim)
        # LSTM layer，形狀為 (input_size, hidden_size, ...)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, dropout=dropout)
        # Fully-connected layer，把 hidden state 線性轉換成 output
        self.hidden2out = nn.Linear(hidden_dim, n_vocab)
    def forward(self, seq_in):
        # LSTM 接受的 input 形狀為 (timesteps, batch, features)，
        # 即 (seq_length, batch_size, embedding_dim)
        # 所以先把形狀為 (batch_size, seq_length) 的 input 轉置後，
        # 再把每個 value (char index) 轉成 embedding vector
        embeddings = self.embeddings(seq_in.t())
        # LSTM 層的 output (lstm_out) 有每個 timestep 出來的結果
        #（也就是每個字進去都會輸出一個 hidden state）
        # 這邊我們取最後一層的結果，即最近一次的結果，來預測下一個字
        lstm_out, _ = self.lstm(embeddings)
        ht = lstm_out[-1]
        # 線性轉換至 output
        out = self.hidden2out(ht)
        return out