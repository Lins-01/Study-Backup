import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderLSTM(nn.Module):
    def __init__(self, input_dim=1, emb_dim=256, hidden_dim=512, n_layers=2):
        super(EncoderLSTM, self).__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hidden_dim, n_layers, batch_first=True)

    def forward(self, src):
        embedded = self.embedding(src)
        outputs, (hidden, cell) = self.rnn(embedded)
        return outputs, hidden, cell


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, encoder_outputs, decoder_hidden):
        # encoder_outputs: (batch_size, seq_len, hidden_dim)
        # decoder_hidden: (batch_size, hidden_dim)

        # Calculate the attention scores.
        scores = torch.bmm(encoder_outputs, decoder_hidden.unsqueeze(2)).squeeze(2)  # (batch_size, seq_len)

        attn_weights = F.softmax(scores, dim=1)  # (batch_size, seq_len)

        context_vector = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)  # (batch_size, hidden_dim)

        return context_vector, attn_weights


class DecoderLSTMWithAttention(nn.Module):
    def __init__(self, output_dim=1, emb_dim=256, hidden_dim=512, n_layers=2):
        super(DecoderLSTMWithAttention, self).__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim + hidden_dim, hidden_dim, n_layers, batch_first=True)
        self.out = nn.Linear(hidden_dim, output_dim)
        self.attention = Attention()

    def forward(self, input, encoder_outputs, hidden, cell):
        input = input.unsqueeze(1)  # (batch_size, 1)
        embedded = self.embedding(input)  # (batch_size, 1, emb_dim)

        context_vector, attn_weights = self.attention(encoder_outputs,
                                                      hidden[-1])  # using the last layer's hidden state

        rnn_input = torch.cat([embedded, context_vector.unsqueeze(1)], dim=2)  # (batch_size, 1, emb_dim + hidden_dim)

        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        prediction = self.out(output.squeeze(1))

        return prediction, hidden, cell


# # Example usage
# INPUT_DIM = 1000  # e.g., size of the source language vocabulary
# OUTPUT_DIM = 1000  # e.g., size of the target language vocabulary
# EMB_DIM = 256
# HIDDEN_DIM = 512
# N_LAYERS = 2
#
# encoder = EncoderLSTM(INPUT_DIM, EMB_DIM, HIDDEN_DIM, N_LAYERS)
# decoder = DecoderLSTMWithAttention(OUTPUT_DIM, EMB_DIM, HIDDEN_DIM, N_LAYERS)
#
# src_seq = torch.randint(0, INPUT_DIM, (32, 10))  # batch of 32, sequence length 10
# encoder_outputs, hidden, cell = encoder(src_seq)
#
# input = torch.randint(0, OUTPUT_DIM, (32,))  # batch of 32, single time step
# output, hidden, cell = decoder(input, encoder_outputs, hidden, cell)