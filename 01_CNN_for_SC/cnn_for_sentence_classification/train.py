import argparse
import os, json
import torch
import gensim
import torch.nn as nn 
from torch import optim
###--------------------------- settings --------------------------------
local = True
p_epochs = 25      # 25 is what a value of epoch Yoon used to his code.

status = 'random'     # 'random', 'static', 'non-static', 'multi'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# tpu = False
# if tpu:
#     !curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py
#     !python pytorch-xla-env-setup.py --version nightly --apt-packages libomp5 libopenblas-dev

#     import torch_xla
#     import torch_xla.core.xla_model as xm
#     os.environ['XLA_USE_BF16'] = '1'
#     os.environ['XLA_TENSOR_ALLOCATOR_MAXSIZE'] = '100000000'
#     device = xm.xla_device()
#     print('device check:', device)
###----------------------- checking if local -------------------------###

if local:
    data_path = 'D:\\Notes\\git_repo\\torch_study\\01_CNN_for_SC\\data\\'
    json_path = 'D:\\Notes\\git_repo\\torch_study\\01_CNN_for_SC\\data\\'
    google_path = "D:\\Notes\\data\\google_wordvector\\"
    from preprocess import preprocess
    from utils import Make_Dataset, Cnn_Model, Kfold_Split, MaxOverTimePooling, binary_accuracy
else:
    data_path = '/kaggle/input/posneg-for-cnn4sc/'
    json_path = '/kaggle/working/'
    google_path = "/kaggle/input/googlenewsvectorsnegative300/"

    
if os.path.isfile(json_path+"data_{}.json".format(status)):
    with open(json_path+"data_{}.json".format(status)) as json_file:
        files = json.load(json_file)
else:
    pre = preprocess(data_path=data_path, json_path=json_path, status=status, save=True)
    word_vecs, W, word_idx_map, revs, _ = pre.get_W()


with open(json_path+"data_{}.json".format(status)) as json_file:
    files = json.load(json_file)


revs, W, word_idx_map, vocab = files['revs'], files['w'], files['word_idx_map'], files['vocab']
model = Cnn_Model(word_vecs=W, status=status)
model.to(device)

criterion = nn.BCEWithLogitsLoss().to(device)
optimizer = optim.Adam(model.parameters())

for i in range(0,10):
    note = {}
    print('test_fold_id iii:', i)
    
    # 데이터로 쓸 revs는 여기서 들어감
    train_loader, test_loader = Kfold_Split(revs, word_idx_map, test_fold_id=i)
    ##------------------------- 그때그때 모델을 만들어주고 돌아가도록 해야지 cheating이 없다.
    model.train()
    optimizer.zero_grad()
    ##------------------------- 그래서 얘네 둘이 for loop 위에 있다. 

    ## Training
    for epoch in range(p_epochs):
        for idx, data in enumerate(train_loader):
            outputs = model(data['input_ids'].to(device))
            loss = criterion(outputs, data['target'].type_as(outputs))
#             loss = criterion(outputs, data['target'].type_as(outputs).squeeze(-1)) # tanh
            acc = binary_accuracy(outputs, data['target'].type_as(outputs))
            loss.backward()   
            # if tpu:
                # xm.optimizer_step(optimizer)
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
            labels = data['target'].to(device)

            acc = binary_accuracy(outputs, labels.type_as(outputs))
            print('test accuracy: {:.3f}'.format(float(acc.data.cpu().numpy())))