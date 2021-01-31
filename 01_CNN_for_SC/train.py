import os, json
import torch
import gensim
import torch.nn as nn 
from torch import optim
# custom module
from utils import Make_Dataset, Cnn_Model, Kfold_Split, MaxOverTimePooling, binary_accuracy

###--------------------------- settings --------------------------------
local = True
make_json = False  # json 안 만듬
p_epochs = 5      # 25 is what a value of epoch Yoon used to his code.
p_static = False
p_multi = False
tpu = False
device = 'cuda' if torch.cuda.is_available() else 'cpu'

###----------------------- checking if local -------------------------###
if local:
    data_path = 'D:\\Notes\\git_repo\\torch_study\\01_CNN_for_SC\data\\'
    google_path = "D:\\Notes\\data\\google_wordvector\\"

else:
    data_path = "/kaggle/input/cnn-word-vector-json/" 
    google_path = "/kaggle/input/googlenewsvectorsnegative300/"

word_vecs = gensim.models.KeyedVectors.load_word2vec_format(google_path + "GoogleNews-vectors-negative300.bin", binary=True, limit=500000)
###--------------------- checking json existence ---------------------###
#TODO: 오랫동안 이 make_json을 안 써서 수정해줄 부분이 있을듯
if make_json:
    from preprocess import preprocess
    pre = preprocess(data_path=data_path, save=True)
    _, word_idx_map, revs, max_len = pre.get_W()

else:
    with open(data_path + "data.json") as json_file:
        files = json.load(json_file)    
    # w =  files['w']  # 자체 embedding 들어가서 w 만들 필요 없음
    revs, word_idx_map, vocab = files['revs'], files['word_idx_map'], files['vocab']
###-------------------------------------------------------------------###

model = Cnn_Model(word_vecs=word_vecs, static=p_static, multi=p_multi)
model.to(device)

criterion = nn.BCEWithLogitsLoss().to(device)
optimizer = optim.Adam(model.parameters())

for i in range(0,10):
    note = {}
    print('iiiiiiii:', i)
    train_loader, test_loader = Kfold_Split(revs, word_idx_map, test_fold_id=i, proper=False)
    ##------------------------- 그때그때 모델을 만들어주고 돌아가도록 해야지 cheating이 없다.
    model.train()
    optimizer.zero_grad()
    ##------------------------- 그래서 얘네 둘이 for loop 위에 있다. 
    
    ## Training
    for epoch in range(p_epochs):
        for idx, data in enumerate(train_loader):
            outputs = model(
                data['input_ids'].to(device))
            loss = criterion(outputs, data['target'].type_as(outputs))
#             loss = criterion(outputs, data['target'].type_as(outputs).squeeze(-1)) # tanh
            acc = binary_accuracy(outputs, data['target'].type_as(outputs))
            loss.backward()   
            if tpu:
                xm.optimizer_step(optimizer)
            optimizer.step()

            if idx % 50 == 0:
                print('epoch:', epoch,' current acc: {:.3f}'.format(float(acc.data.cpu().numpy())))
        print('training is done!')

    ## test 
    total = len(test_loader)
    correct = 0
    model.eval()

    ## .eval()이랑 no_grad()가 들어가면 자동으로 training=False가 되나?
    ## .eval()에서 dropout은 disable됨
    with torch.no_grad():
        ### L2 norm weight clipping
        for param in model.parameters():
            param.clamp_(min=-3, max=3)
            
        ### 위의 param.clamp_ 이게 gradient update를 못하게 하기 때문에
        ### train_loader에서는 쓰면 에러가 나는 것 같다.
        
        for data in test_loader:
            datas = data['input_ids'].to(device)
            outputs = model(datas)
#             print('outputs size:', outputs.size()) # the value should be [50,]            
#             _, predicted = torch.max(outputs.data, 1)
            labels = data['target'].to(device)
            
            acc = binary_accuracy(outputs, labels.type_as(outputs))
            print('test accuracy: {:.3f}'.format(float(acc.data.cpu().numpy())))
