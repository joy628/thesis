import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn import Transformer
import math
import random

class TimeSeriesAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads):
        super(TimeSeriesAutoencoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
       
        self.encoder = nn.LSTM(
            input_dim, hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            bidirectional=False 
        )
        
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=False,device="cuda", dropout=0.1)
        
        self.decoder = nn.LSTM(
            hidden_dim, hidden_dim, 
            num_layers=4, 
            batch_first=True, 
            bidirectional=False
        )
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, lengths, attention=True):
        
        _, seq_len, _ = x.shape
        
        ## encoder 
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (h, c) = self.encoder(packed)  # packed_output: shape = (batch_size, seq_len, hidden_dim)
        encoder_output, _ = pad_packed_sequence(packed_output, batch_first=True)  # shape = (batch_size, seq_len, hidden_dim)
          
        ## decoder
        if attention: 
            ## attention
            encoder_output_t = encoder_output.permute(1, 0, 2)  #(seq_len, batch_size, hidden_dim)
            attention_output, _ = self.attention(encoder_output_t, encoder_output_t, encoder_output_t)  
            attention_output = attention_output.permute(1, 0, 2)  # (batch_size, seq_len, hidden_dim)
            attention_encoder_output = encoder_output + attention_output # (batch_size, seq_len, hidden_dim)  
            decoder_input = attention_encoder_output  ## attention output as input to decoder
        else:
            decoder_input = h[-1].unsqueeze(1).repeat(1, seq_len, 1) # shape = (batch_size, seq_len, hidden_dim)    
        
        packed_decoder_input = pack_padded_sequence(decoder_input, lengths.cpu(), batch_first=True, enforce_sorted=False) # shape = (batch_size, seq_len, hidden_dim)
        packed_output, _ = self.decoder(packed_decoder_input, (h, c))
        decoder_output, _ = pad_packed_sequence(packed_output, batch_first=True)  # shape = (batch_size, seq_len, hidden_dim)            
        ## output layer
        output = self.output_layer(decoder_output)  # shape = (batch_size, seq_len, input_dim)
                
        return output, encoder_output
    
  
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

class TemporalConv(nn.Module):
    """
    time series data convolutional layer
    """
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super(TemporalConv, self).__init__()
        self.conv = nn.Conv1d(input_dim, hidden_dim, kernel_size, padding=kernel_size//2)
    
    def forward(self, x):
        """
        :param x:  (batch_size, seq_len, input_dim)
        :return: (batch_size, seq_len, hidden_dim)
        """
        x = x.transpose(1, 2)  #  (batch_size, input_dim, seq_len)
        x = self.conv(x)  # convolution
        x = x.transpose(1, 2)  # (batch_size, seq_len, hidden_dim)
        return x


class TransformerEncoderDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward):
        super(TransformerEncoderDecoder, self).__init__()
        
        self.d_model = hidden_dim

        self.input_embedding = nn.Linear(input_dim,hidden_dim)
        
        self.positional_encoding = PositionalEncoding(hidden_dim)
        
        self.temporal_conv = TemporalConv(hidden_dim, hidden_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(hidden_dim, nhead, dim_feedforward, dropout=0.1,batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(hidden_dim, nhead, dim_feedforward, dropout=0.1,batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        
        # outputlayer
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, src):
        """
        :param src: shape = (batch_size, seq_len, input_dim)
        :param tgt: shape = (batch_size, seq_len, input_dim)
        :return output: shape = (batch_size, seq_len, input_dim)
        """
        batch_size, seq_len, input_dim = src.size()
        
        tgt = torch.cat([torch.zeros(batch_size, 1, input_dim, device=src.device), src[:, :-1, :]], dim=1) # shape 
        
        
        src = self.input_embedding(src) * math.sqrt(self.d_model) # (batch_size, seq_len, hidden_dim)
        src = self.positional_encoding(src) # (batch_size, seq_len, hidden_dim)
        src = self.temporal_conv(src) # (batch_size, seq_len, hidden_dim)
        
        tgt = self.input_embedding(tgt) * math.sqrt(self.d_model)  # (batch_size, seq_len, hidden_dim)
        tgt = self.positional_encoding(tgt) # (batch_size, seq_len, hidden_dim)
        
        # encoder
        memory = self.encoder(src)
        
        # decoder
        output = self.decoder(tgt, memory)
        
        #ouput layer
        output = self.output_layer(output)
        return output
    