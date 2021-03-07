import spacy
from torchtext.datasets import Multi30k
from torchtext.data import Field

class PreProcess:
    def __init__(self):
        self.spacy_fr = spacy.load('fr_core_news_sm')
        self.spacy_en = spacy.load('en_core_web_sm')

    def _tokenize_fr(self, text):
        return [tok.text for tok in self.spacy_fr.tokenizer(text)][::-1]

    def _tokenize_en(self, text):
        return [tok.text for tok in self.spacy_en.tokenizer(text)][::-1]

    def return_src_trg(self):
        SRC = Field(tokenize=self._tokenize_fr,
                    init_token="<sos>",
                    eos_token="<eos>",
                    lower=True)
        TRG = Field(tokenize=self._tokenize_en,
                    init_token="<sos>",
                    eos_token="<eos>",
                    lower=True)
        return SRC, TRG


    def split_data(self):
        SRC, TRG = self.return_src_trg()
        train_data, valid_data, test_data = Multi30k.splits(path="/home/soohyun/data/multi30k/",
                                            exts=(".fr",".en"),
                                            fields=(SRC, TRG))

        SRC.build_vocab(train_data, min_freq=2)
        TRG.build_vocab(train_data, min_freq=2)
        return train_data, valid_data, test_data, SRC, TRG