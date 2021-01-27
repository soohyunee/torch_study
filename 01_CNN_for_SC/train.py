import os
import torch
from preprocess import preprocess

local = True
json = True
####------------------------------------TPU--------------------------------------
if local:
    tpu = False

if tpu:
    !curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py
    !python pytorch-xla-env-setup.py --version nightly --apt-packages libomp5 libopenblas-dev
    import torch_xla
    import torch_xla.core.xla_model as xm

    os.environ['XLA_USE_BF16'] = '1'
    os.environ['XLA_TENSOR_ALLOCATOR_MAXSIZE'] = '100000000'
    device = xm.xla_device()
else:
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
####------------------------------------TPU--------------------------------------


###===================================JSON========================================
if json:
    if local:
        json_path = "D:\\Notes\\git_repo\\torch_study\\01_CNN_for_SC\\data\\data.json"
    else:
        json_path = "/kaggle/input/cnn-word-vector-json/data.json"
    
    with open(json_path) as json_file:
        files = json.load(json_file)
        
    # w =  files['w']  # 자체 embedding 들어가서 w 만들 필요 없음
    revs, word_idx_map, vocab = files['revs'], files['word_idx_map'], files['vocab']
    ## word_vecs : dictionary형, 벡터를 보고프면 word_vecs['word']

else:
    pre = preprocess(word_vecs, save=True)
    w, word_idx_map, revs, max_len = pre.get_W()
###===================================JSON========================================