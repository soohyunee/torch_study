import re
import gensim
import numpy as np
import pandas as pd
from collections import defaultdict

## modified codes in https://github.com/yoonkim/CNN_sentence 
class preprocess:
    def __init__(self, data_path, save=False):
        self.k = 300     # the embedding dimension of pretrained vector noted at the paper
        self.revs = []
        self.vocab_size = 0
        self.max_len = 56 # the value what yoon used at his codes
        self.word_idx_map = dict()
        ## limit을 준 건, full로 올리면 메모리 에러가 나기 때문 
        self.word_vecs = gensim.models.KeyedVectors.load_word2vec_format("/kaggle/input/googlenewsvectorsnegative300/GoogleNews-vectors-negative300.bin", binary=True, limit=500000)
        # self.stop = set(stopwords.words('english'))
        self.save = save
        self.data_path = data_path

        
    def clean_str(self, string):
        ## string이 sentence로 들어와서, self.stop으로 못 거름...=_=;
        ## 우선 stopwords 안 거르는걸로 해서 함 돌려보자.
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
        string = re.sub(r"\'s", " \'s", string) 
        string = re.sub(r"\'ve", " \'ve", string) 
        string = re.sub(r"n\'t", " n\'t", string) 
        string = re.sub(r"\'re", " \'re", string) 
        string = re.sub(r"\'d", " \'d", string) 
        string = re.sub(r"\'ll", " \'ll", string) 
        string = re.sub(r",", " , ", string) 
        string = re.sub(r"!", " ! ", string) 
        string = re.sub(r"\(", " \( ", string) 
        string = re.sub(r"\)", " \) ", string) 
        string = re.sub(r"\?", " \? ", string) 
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()
    
    
    def build_data_cv(self, cv=10):
        pos_file = self.data_path + "rt-polarity.pos"
        neg_file = self.data_path + "rt-polarity.neg"

        file_list = [pos_file, neg_file]    
        self.vocab = defaultdict(float)

        for file in file_list:
            with open(file, "rb") as f:
                for line in f: 
                    try:
                        line = line.decode("utf-8")
                    except UnicodeDecodeError:
                        line = line.decode('latin-1')

                    rev = []
                    rev.append(line.strip())
                    orig_rev = self.clean_str(" ".join(rev))
        
                    words = set(orig_rev.split())
                    
                    if len(words) > self.max_len:
                        self.max_len = len(words)
                    
                    for word in words:
                        try:
                            self.vocab[word] += 1
                            self.vocab_size += 1
                        except:
                            self.vocab[word] = 1
                            self.vocab_size += 1
                            
                    if file[-3:] == "pos":        
                        datum = {"y": 1,
                                "text": orig_rev,
                                "split": np.random.randint(0,cv)}
                        ## np.random.randint는 discrete uniform distribution
                        
                    elif file[-3:] == "neg":
                        datum = {"y": 0,
                                "text": orig_rev,
                                "split": np.random.randint(0,cv)}
                    self.revs.append(datum)
        
        return self.revs, self.vocab, self.max_len
    # self.revs :  sentence와 label 데이터
    # self.vocab : vocabulary set with its frequency
    # self.word_vecs : google pre-trained word vector
    ### 단어는 없고 word_idx와 word_vector만 갖고 있음
    
    def add_unknown_words(self, min_df=1):
        cnt = 0
        for word in self.vocab:
            if word not in self.word_vecs.keys() and self.vocab[word] >= min_df:
                self.word_vecs[word] = np.random.uniform(-0.25,0.25,self.k)  
                self.vocab_size += 1
                cnt += 1
        print(cnt, ' of unknown words were here')          

    def get_W(self):
        self.revs, self.vocab, self.max_len = self.build_data_cv()
        self.add_unknown_words()
        self.W = np.zeros(shape=(self.vocab_size+1, self.k), dtype='float32')            
        
        i = 0
        for word in self.vocab:
            print(i, 'th word:', word)
            self.W[i] = self.word_vecs[word]
            self.word_idx_map[word] = i
            i += 1
            
        if self.save:
            self.save_file()

        return self.W, self.word_idx_map, self.revs, self.max_len
    
    def save_file(self):
        data = {'revs':self.revs,
               'w':self.W,
               'word_idx_map':self.word_idx_map,
               'vocab':self.vocab}
        
        pd.Series(data).to_json('data.json')
        print('making a json file completed!')