import torch

# https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
# https://discuss.pytorch.org/uploads/default/original/3X/6/8/68a2efdb5726be0e32abe75b65fcb8d92d101dff.png

class BiLSTM(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, num_classes: int, bias: bool = True):
        super().__init__()
        self.input_size: int = input_size
        self.hidden_size: int = hidden_size
        self.num_layers: int = num_layers
        self.bias: bool = bias
        self.lstm: torch.nn.LSTM = torch.nn.LSTM(self.input_size, self.hidden_size, self.num_layers, self.bias, batch_first=False, dropout=0, bidirectional=False, proj_size=0)
        self.outl: torch.nn.Linear = torch.nn.Linear(hidden_size, num_classes)
    def forward(self, input):
        out, (hn, cn) = self.lstm(input)
        out = out[-1,:,:] # only use last sequence output
        out = self.outl(out)
        return out
