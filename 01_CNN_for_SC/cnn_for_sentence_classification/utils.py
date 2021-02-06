##### import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


class Make_Dataset(TensorDataset):
    def __init__(self, word_idx_map, xy):
        '''
        self.xy : a sentence, string type
        '''
        super().__init__()
        self.xy = xy
        self.max_len = 56
        self.word_idx_map = word_idx_map
        
    def __len__(self):
        return len(self.xy)
        
    def __getitem__(self, idx):
        splitted_sentence = self.xy[idx]['text'].split()
        tmp = []
        for word in splitted_sentence:
            tmp.append(self.word_idx_map[word])
            
        if len(tmp) < self.max_len:
            for _ in range(len(tmp), self.max_len):
                tmp.append(0)
                
        if self.xy[idx]['y'] in [0,1]: 
            return {'input_ids': torch.LongTensor(tmp).flatten(),
                    'target': torch.tensor(self.xy[idx]['y'])}
    
        else:       # mr-dataset에서는 여기에 걸리는 케이스가 없음. 전부 label이 존재. 
            return {'input_ids': torch.LongTensor(tmp).flatten()}


def Kfold_Split(revs, word_idx_map, test_fold_id):
    train = []
    test = []
    for datum in revs:
        if datum['split'] == test_fold_id:
            test.append(datum)
        else:
            train.append(datum)   

    train_dataset = Make_Dataset(word_idx_map=word_idx_map, xy=train)
    test_dataset = Make_Dataset(word_idx_map=word_idx_map, xy=test)

    train_batch = 50 
    test_batch = 50
        
    #TODO: batch_sizes 모두 collate_fn써서 적합하게 바꿔주기
    train_loader = DataLoader(train_dataset, batch_size=train_batch, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=test_batch, shuffle=False, drop_last=True)
    return train_loader, test_loader



### max-over-time pooling 과 max-pooling의 다른 점
# https://stackoverflow.com/questions/48549670/pooling-vs-pooling-over-time#:~:text=Max%20pooling%20typically%20applies%20to,along%20a%201d%20feature%20vector.&text=Max%20pooling%20over%20time%20takes,vector%20and%20computes%20the%20max.

## origin
# https://github.com/aisolab/nlp_classification/blob/master/Convolutional_Neural_Networks_for_Sentence_Classification/model/ops.py
def MaxOverTimePooling(x, status='non-static'):   # retrieve each feature map
    '''
    input : f3, f4, f5   # 각각 filter=3일때, 4일때, 5일때의 feature map
           ㄴ each shape : batch_size, # of kernels, strides된 이후의 size
          
    process : f3.max = [3,6,9], (f3이 [[1,2,3],[4,5,6],[7,8,9]]일 경우)

    output : 1-dim vector which containes a max value for each
    '''
    f3, f4, f5 = x  
    if status == 'multi':
        f3 = f3.max(dim=-1).values.view(2, 955, 100)
        f4 = f4.max(dim=-1).values.view(2, 955, 100)
        f5 = f5.max(dim=-1).values.view(2, 955, 100)
        
        return torch.cat([f3.max(dim=0).values,
                  f4.max(dim=0).values,
                  f5.max(dim=0).values], dim=-1)

    else:
        return torch.cat([
                f3.max(dim=-1).values,
                f4.max(dim=-1)[0],
                f5.max(dim=-1)[0],
            ], dim=-1,)


class Cnn_Model(nn.Module):
    def __init__(self, word_vecs, status='non-static'):
        super(Cnn_Model, self).__init__()
        assert status in ['random','static','non-static','multi']
        self.status = status
        self.freeze = True if status=='static' else False      # determining whether the W is static/non-static
        self.input_dim = 1   
        self.n_filters = 100
        self.kernel_sizes = [3,4,5]
        self.word_vecs = torch.FloatTensor(word_vecs)
        
        ## nn.embedding은 2-dim float tensor로 만들어지고, from_pretrained에서의 freeze는 기본적으로 True이다.
        ## static이 false면, freeze=False, 즉 non-static
        # https://github.com/aisolab/nlp_classification/blob/master/Convolutional_Neural_Networks_for_Sentence_Classification/model/ops.py
        if self.status == 'multi': 
            self._static = nn.Embedding.from_pretrained(self.word_vecs, freeze=True)
            self._non_static = nn.Embedding.from_pretrained(self.word_vecs, freeze=False)
            
        elif self.status == 'random':
            self.embedding = nn.Embedding(len(self.word_vecs), embedding_dim=300)

        else:
            self.embedding = nn.Embedding.from_pretrained(self.word_vecs, freeze=self.freeze)
        
        self.convs = nn.ModuleList([
                nn.Conv2d(in_channels=self.input_dim, out_channels=self.n_filters, kernel_size=(ks, 300), padding=0) for ks in self.kernel_sizes])
        
        self.fc = nn.Linear(len(self.kernel_sizes)*self.n_filters, 1) 
        self.dropout = nn.Dropout()
        
   
    def forward(self, x):                        # x.size :  50, 56                
        if self.status == 'multi':
            non_static = self._non_static(x).permute(0,1,2)
            static = self._static(x).permute(0,1,2)
            x = torch.cat([static, non_static], dim=0)
            x = x.unsqueeze(1)                   # 50, 1, 56, 300                   
            conved = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
            x = MaxOverTimePooling(conved, status=self.status)
                        
        else:
            x = self.embedding(x)                # 50, 56, 300    
            x = x.unsqueeze(1)                   # 50, 1, 56, 300                   
            conved = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
            x = MaxOverTimePooling(conved)
            ## make a feature map for each filter
            # x :::: [f3.size : 50, 10, 54], [f4.size : 50, 10, 53], [f5.size : 50, 10, 52]
    
        ### a penultimate layer
        x = self.dropout(x) * 0.5 # 50, 30
        output = self.fc(x)                          # 50, 30
        return output.squeeze()


def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc