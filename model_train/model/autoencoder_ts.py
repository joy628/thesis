import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn import Transformer
import math
import random

class TimeSeriesAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(TimeSeriesAutoencoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.encoder = nn.LSTM(
            input_dim, hidden_dim, 
            num_layers=4, 
            batch_first=True, 
            bidirectional=False, 
        )
        self.decoder = nn.LSTM(
            hidden_dim, hidden_dim, 
            num_layers=4, 
            batch_first=True, 
            bidirectional=False, 
        )
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, lengths):
        ## encoder 
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (h, c) = self.encoder(packed)  # packed_output: shape = (batch_size, seq_len, hidden_dim)
        encoder_output, _ = pad_packed_sequence(packed_output, batch_first=True)  # shape = (batch_size, seq_len, hidden_dim)
   
        ## decoder
        decoder_input = h[-1].unsqueeze(1).repeat(1, x.size(1), 1) # shape = (batch_size, seq_len, hidden_dim)    
        packed_decoder_input = pack_padded_sequence(decoder_input, lengths.cpu(), batch_first=True, enforce_sorted=False) # shape = (batch_size, seq_len, hidden_dim)
        packed_output, _ = self.decoder(packed_decoder_input, (h, c))
        decoder_output, _ = pad_packed_sequence(packed_output, batch_first=True)  # shape = (batch_size, seq_len, hidden_dim)
                
        ## output layer
        output = self.output_layer(decoder_output)  # shape = (batch_size, seq_len, input_dim)
                
        return output, encoder_output
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=6000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    
class TransformerAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers):
        super(TransformerAutoencoder, self).__init__()
        self.positional_encoding = PositionalEncoding(input_dim)
        self.encoder = Transformer(
            d_model=input_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=hidden_dim,
            dropout=0.1,
            batch_first=True
            )
        self.fc = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        x = self.positional_encoding(x)

        # Encoder
        memory = self.encoder(src=x, tgt=x)

        # Decoder
        output = self.fc(memory)
        return output



class lstm_encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(lstm_encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

    def forward(self, x_input, lengths):
        
        lengths = lengths.cpu()
    
        packed_input = pack_padded_sequence(x_input, lengths, batch_first=True, enforce_sorted=False)
        
        packed_output, hidden = self.lstm(packed_input)
        
        lstm_out, _ = pad_packed_sequence(packed_output, batch_first=True)
        
        return lstm_out, hidden


class lstm_decoder(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=1):

        super(lstm_decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, input_size)  # Linear layer to map hidden state to output

    def forward(self, x_input, encoder_hidden_states):

        lstm_out, hidden = self.lstm(x_input.unsqueeze(1), encoder_hidden_states)  # Add sequence dimension
        output = self.linear(lstm_out.squeeze(1))  # Remove sequence dimension
        return output, hidden


class lstm_autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(lstm_autoencoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Define encoder and decoder
        self.encoder = lstm_encoder(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.decoder = lstm_decoder(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)

    def forward(self, x_input, lengths, teacher_forcing_ratio=0.5):
        batch_size, seq_len, _ = x_input.shape

        # Encode the input sequence
        encoder_output, encoder_hidden = self.encoder(x_input, lengths)

        # Initialize decoder input with the last element of the input sequence
        decoder_input = x_input[:, -1, :]  # shape: (batch_size, input_size)
        decoder_hidden = encoder_hidden

        # Initialize tensor to store reconstructed outputs
        outputs = torch.zeros(batch_size, seq_len, self.input_size, device=x_input.device)

        # Decode step-by-step
        for t in range(seq_len):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            outputs[:, t, :] = decoder_output

            # Decide whether to use teacher forcing
            if random.random() < teacher_forcing_ratio:
                # Use ground truth as the next input
                decoder_input = x_input[:, t, :]
            else:
                # Use the decoder's output as the next input
                decoder_input = decoder_output

        return outputs, encoder_output