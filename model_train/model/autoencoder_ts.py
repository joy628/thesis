import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn import Transformer
import math
import random

class TimeSeriesAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(TimeSeriesAutoencoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.encoder = nn.LSTM(
            input_dim, hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            bidirectional=False 
        )
        self.decoder = nn.LSTM(
            hidden_dim, hidden_dim, 
            num_layers=4, 
            batch_first=True, 
            bidirectional=False
        )
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, lengths, teacher_forcing=True, teacher_forcing_ratio=0.5):
        
        _, seq_len, _ = x.shape
        
        ## encoder 
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (h, c) = self.encoder(packed)  # packed_output: shape = (batch_size, seq_len, hidden_dim)
        encoder_output, _ = pad_packed_sequence(packed_output, batch_first=True)  # shape = (batch_size, seq_len, hidden_dim)
        
        if teacher_forcing:   
            outputs = torch.zeros_like(x, device=x.device)  # (batch_size, seq_len, input_size)
            decoder_input = x[:, -1, :]  # shape: (batch_size, input_size)
            hidden = (h, c)
            for t in range(seq_len):
                lstm_out, hidden = self.decoder(decoder_input.unsqueeze(1), hidden)
                decoder_output = self.output_layer(lstm_out.squeeze(1))  
                outputs[:, t, :] = decoder_output
                # Teacher Forcing 
                decoder_input = x[:, t, :] if random.random() < teacher_forcing_ratio else decoder_output
        else:       
            ## decoder
            decoder_input = h[-1].unsqueeze(1).repeat(1, seq_len, 1) # shape = (batch_size, seq_len, hidden_dim)    
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
    
    
class LSTMTFAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(LSTMTFAutoencoder,self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True,bidirectional=False,)
        self.decoder_lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True,bidirectional=False,)

        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, lengths, teacher_forcing_ratio=0.5):
        _, seq_len, _ = x.shape

        # encoder
        packed_x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, hidden = self.encoder_lstm(packed_x)
        encoder_out, _ = pad_packed_sequence(packed_out, batch_first=True)

        # decoder
        outputs = torch.zeros_like(x, device=x.device)  # (batch_size, seq_len, input_size)
        decoder_input = x[:, -1, :]  # shape: (batch_size, input_size)

        for t in range(seq_len):
            lstm_out, hidden = self.decoder_lstm(decoder_input.unsqueeze(1), hidden)
            decoder_output = self.output_layer(lstm_out.squeeze(1))  
            outputs[:, t, :] = decoder_output

            # Teacher Forcing 
            decoder_input = x[:, t, :] if random.random() < teacher_forcing_ratio else decoder_output

        return outputs, encoder_out
    
    
    
######  transformer encoder decoder

class TransformerEncoderDecoder(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout=0.1):
        super(TransformerEncoderDecoder, self).__init__()
        self.d_model = d_model

        self.input_embedding = nn.Linear(input_dim, d_model)
        
        self.positional_encoding = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout,batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        
        # outputlayer
        self.output_layer = nn.Linear(d_model, input_dim)

    def forward(self, src):
        """
        :param src: shape = (batch_size, seq_len, input_dim)
        :param tgt: shape = (batch_size, seq_len, input_dim)
        :return output: shape = (batch_size, seq_len, input_dim)
        """
        batch_size, seq_len, input_dim = src.size()
        tgt = torch.cat([torch.zeros(batch_size, 1, input_dim, device=src.device), src[:, :-1, :]], dim=1) # shape 

        src = self.input_embedding(src) * math.sqrt(self.d_model)
        src = self.positional_encoding(src)
        
        tgt = self.input_embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.positional_encoding(tgt)
        
        # encoder
        memory = self.encoder(src)
        
        # decoder
        output = self.decoder(tgt, memory)
        
        #ouput layer
        output = self.output_layer(output)
        return output