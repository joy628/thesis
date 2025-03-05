import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class TimeSeriesAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, lstm_layers):
        super(TimeSeriesAutoencoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.encoder = nn.LSTM(
            input_dim, hidden_dim, 
            num_layers=lstm_layers, 
            batch_first=True, 
            bidirectional=False, 
        )
        self.decoder = nn.LSTM(
            hidden_dim, hidden_dim, 
            num_layers=lstm_layers, 
            batch_first=True, 
            bidirectional=False, 
        )
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, lengths):
        ## encoder 
        # pack, use the length of the sequence to avoid unnecessary computation

        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (h, c) = self.encoder(packed)  # packed_output: shape = (batch_size, seq_len, hidden_dim)
        
        # unpack, back to original shape
        encoder_output, _ = pad_packed_sequence(packed_output, batch_first=True)  # shape = (batch_size, seq_len, hidden_dim)
   
        
        ## decoder
        decoder_input = torch.zeros(x.size(0), x.size(1), self.hidden_dim).to(x.device)  # shape = (batch_size, seq_len, hidden_dim)
        
        packed_decoder_input = pack_padded_sequence(decoder_input, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.decoder(packed_decoder_input, (h, c))
        decoder_output, _ = pad_packed_sequence(packed_output, batch_first=True)  # shape = (batch_size, seq_len, hidden_dim)
                
        ## output layer
        output = self.output_layer(decoder_output)  # shape = (batch_size, seq_len, input_dim)
                
        return output, encoder_output

# class TimeSeriesAutoencoder(nn.Module):
#     def __init__(self, input_dim, hidden_dim, lstm_layers, dropout):
#         super(TimeSeriesAutoencoder, self).__init__()
#         self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers=lstm_layers, batch_first=True, bidirectional=False, dropout=dropout)
#         self.decoder = nn.LSTM(hidden_dim, input_dim, num_layers=lstm_layers, batch_first=True, bidirectional=False, dropout=dropout)
#         self.output_layer = nn.Linear(input_dim, input_dim)

#     def forward(self, x, lengths):
#         packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
#         _, (h, _) = self.encoder(packed)  
#         h = h.squeeze(0)  

#         h_repeated = h.unsqueeze(1).repeat(1, x.shape[1], 1)
#         output, _ = self.decoder(h_repeated)
#         output = self.output_layer(output)  

#         return output, h  
