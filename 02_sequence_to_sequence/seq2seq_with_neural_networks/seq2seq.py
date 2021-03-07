import random
import torch
import torck.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        src_ = src.transpose(1,0)
        input_lengths = torch.LongTensor([torch.max(src_[i,:].data.nonzero())+1 for i in range(src_.size(0))])
        input_lengths, sorted_idx = input_lengths.sort(0, descending=True)

        input_seq2idx = src_[sorted_idx]
        embedded = self.dropout(self.embedding(src))
        embedded = embedded.transpose(1,0)

        packed_input = nn.utils.rnn.pack_padded_sequence(embedded,\
                                                        input_lengths.tolist(), \
                                                        batch_first=True)
        packed_output, (hidden, cell) = self.lstm(packed_input)
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output,\
                                  batch_first=True)

        output = output.transpose(1,0)
        return output, hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.lstm = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        if input.dim() > 2:
            input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.fc_out(output.sequeeze(0))
        return prediction, hidden, cell



class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        assert encoder.hid_dim == decoder.hid_dim, \
        "hidden dimensions of encoder and decoder must be equal"
        assert encoder.n_layers == decoder.n_layers, \
        "encoder and decoder must have equal number of layers"

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # last hidden state of the encoder is used as the init hidden state of the decoder
        encoder_output, hidden, cell = self.encoder(src)

        # first input to the decoder is the <sos> token
        input = trg[0,:]

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)

            # receive output tensor (predictions) and new hidden and cell states
            outputs[t] = output

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from out predictions
            top1 = output.argmax(1)

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = trg[t] if teacher_force else top1

        return outputs


