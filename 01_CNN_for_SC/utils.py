import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# came from custom module
from preprocess import preprocess


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
            return {'input_ids':torch.LongTensor(tmp).flatten(),
                    'target':torch.tensor(self.xy[idx]['y'])}
    
        else:  # mr-dataset에는 여기에 걸리는 케이스가 없음. 전부 label 존재. 
            return {'input_ids':torch.LongTensor(tmp).flatten()}


    
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
    
    print('train length:', len(train_dataset))
    print('test length:', len(test_dataset))
## 우선 코드 전체적으로 쭈욱 한 번 돌리고 나서, 여길 수정하자.
#     proper_batch_size = int(len(vocab)/10)
#     print('proper:',proper_batch_size)

    train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=50, shuffle=True, drop_last=True)
    return train_loader, test_loader


### max-over-time pooling 과 max-pooling의 다른 점
# https://stackoverflow.com/questions/48549670/pooling-vs-pooling-over-time#:~:text=Max%20pooling%20typically%20applies%20to,along%20a%201d%20feature%20vector.&text=Max%20pooling%20over%20time%20takes,vector%20and%20computes%20the%20max.

## origin
# https://github.com/aisolab/nlp_classification/blob/master/Convolutional_Neural_Networks_for_Sentence_Classification/model/ops.py
def MaxOverTimePooling(f3, f4, f5):   # retrieve each feature map
    '''
    input : f3, f4, f4   # 각각 filter=3일때, 4일때, 5일때
          - each shape : batch_size, # of kernels, 3번째는 뭐지? 
          
    process : f3.max = [3,6,9], (f3이 [[1,2,3],[4,5,6],[7,8,9]]일 경우)

    output : 1-dim vector which containes a max value for each
    '''
    return torch.cat([
            f3.max(dim=-1).values,
            f4.max(dim=-1)[0],
            f5.max(dim=-1)[0],
        ], dim=-1,)



class Cnn_Model(nn.Module):
    def __init__(self):
        super(Cnn_Model, self).__init__()
        import gensim
        self.n_filters = 100
        self.input_dim = 1   # 그냥 text니깐 1임. vision일 경우 3 
        ### word_vecs 만들어주기, dict 자료형에, word와 wordvector가 함께 있어야 함
        word_vecs = gensim.models.KeyedVectors.load_word2vec_format("D:\\Notes\\data\\google_wordvector\\GoogleNews-vectors-negative300.bin", binary=True, limit=700000)
        self.word_vecs = torch.FloatTensor(word_vecs.vectors    
        ## nn.embedding은 2-dim float tensor로 만들어지고,
        ## from_pretrained에서의 freeze는 기본적으로 True이다.
        self.embedding = nn.Embedding.from_pretrained(self.word_vecs, freeze=False)
        self.conv3_layer = nn.Conv2d(self.input_dim, self.n_filters, kernel_size=(3,300))
        self.conv4_layer = nn.Conv2d(self.input_dim, self.n_filters, kernel_size=(4,300))
        self.conv5_layer = nn.Conv2d(self.input_dim, self.n_filters, kernel_size=(5,300))
        
        ## 우선 filter_size를 3으로 줬을 때
        self.fc = nn.Linear(3*self.n_filters, 1) 
        self.dropout = nn.Dropout()
        
    def forward(self, x):
#         print('first x:', x.size())                  # 50, 56                
        x = self.embedding(x)                       # 50, 56, 300    
        x = x.unsqueeze(1)                          # 50, 1, 56,, 300                   
        ## make a feature map for each filter
        f3 = F.relu(self.conv3_layer(x).squeeze(3))  # 50, 10, 54
        f4 = F.relu(self.conv4_layer(x).squeeze(3))  # 50, 10, 53
        f5 = F.relu(self.conv5_layer(x).squeeze(3))  # 50, 10, 52
        x = MaxOverTimePooling(f3,f4,f5)
    
        ### a penultimate layer
        x = self.dropout(x) * 0.5 # 50, 30
        #TODO: dropout 이게 test_loader가 돌아갈 땐 실행이 안 되어야 하는데...;;
        
#         x.size()  batch_size, n_filters*len(filter_sizes)

        output = self.fc(x)                          # 50, 30
        
        return output.squeeze()


def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc