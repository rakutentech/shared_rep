'''
The goal here is to covert data from citeulike dataset to our format
'''


from numpy import genfromtxt
import numpy as np
import scipy.sparse as sp
from sklearn import datasets
from sklearn import preprocessing as prep
import pickle


########################## generate val_grtr  ########################
#generate val_grtr
testset=genfromtxt('./eval/cold/test.csv', delimiter=',')


val_grtr = {} 

test_item_ids = np.unique( testset[:,1]).astype(int)
all_users = np.array(list(range(1, 5552)))
for test_iid in test_item_ids:
    pos_samples = testset[testset[:,1] == test_iid, 0]
    neg_samples = np.array(list(set(all_users.flat) - set(pos_samples.flat))) 
    val_grtr[test_iid] = [list( pos_samples), list( neg_samples)]                        


########################## generate val_grtr_userbased  ########################

val_grtr_userbased = {} 


test_user_ids = np.unique( testset[:,0]).astype(int)

for test_uid in test_user_ids:
    pos_samples = testset[testset[:,0] == test_uid, 1]
    neg_samples = np.array(list(set(test_item_ids.flat) - set(pos_samples.flat))) 
    val_grtr_userbased[test_uid] = [list( pos_samples), list( neg_samples)]                        
                                             

########################## generate IFeature ########################

def tfidf(R):
    row = R.shape[0]
    col = R.shape[1]
    Rbin = R.copy()
    Rbin[Rbin!=0]=1.0
    R = R + Rbin
    tf = R.copy()
    tf.data = np.log(tf.data)
    idf = np.sum(Rbin,0)
    idf = np.log(row/(1+idf))
    idf = sp.spdiags(idf,0,col,col)
    return tf * idf

def prep_standardize(x):
    """
    takes sparse input and compute standardized version

    Note:
        cap at 5 std

    :param x: 2D scipy sparse data array to standardize (column-wise), must support row indexing
    :return: the object to perform scale (stores mean/std) for inference, as well as the scaled x
    """
    x_nzrow = x.any(axis=1)
    scaler = prep.StandardScaler().fit(x[x_nzrow, :])
    x_scaled = np.copy(x)
    x_scaled[x_nzrow, :] = scaler.transform(x_scaled[x_nzrow, :])
    x_scaled[x_scaled > 5] = 5
    x_scaled[x_scaled < -5] = -5
    x_scaled[np.absolute(x_scaled) < 1e-5] = 0
    return scaler, x_scaled

item_content_file = './eval/item_features_0based.txt'
#generate IFeature, uFeature and trainset
item_content, _ = datasets.load_svmlight_file(item_content_file, zero_based=True, dtype=np.float32)

item_content = tfidf(item_content)

from sklearn.utils.extmath import randomized_svd
u,s,_ = randomized_svd(item_content, n_components=300, n_iter=5)
item_content = u * s
_, item_content = prep_standardize(item_content)

if sp.issparse(item_content):
    IFeature = item_content.tolil(copy=False)
else:
    IFeature = item_content



########################## generate negative samples and trainset ########################
trainset=genfromtxt('./eval/cold/train.csv', delimiter=',').astype(int)

index = np.random.choice(range(trainset.shape[0]), size=5000,  replace = False)
trainset = trainset[index]

user_ids = np.unique( trainset[:,0]).astype(int)

#### uFeature
uFeature = {}
for i in user_ids:
    uFeature[i] = trainset[trainset[:,0]==i,1]


#### trainset
item_ids = np.unique( trainset[:,1]).astype(int)
neg_trainset = []
for j in item_ids:
    if j in test_item_ids:
        print('here')
        continue
    index = trainset[:,1] == j
    negtive_user = np.zeros((10*np.sum(index), 3))  # 1 is the ratio of #NS/#PS
    negtive_user[:,0] = np.random.choice(trainset[np.logical_not( index) , 0], size=1* np.sum(index),  replace = False) # 1 is the ratio of #NS/#PS
    negtive_user[:,1] = np.repeat(j, 10*np.sum(index)) # 10 is the ratio of #NS/#PS
    neg_trainset.append( negtive_user)

neg_trainset = np.vstack(neg_trainset).astype(int)    
trainset_final = np.vstack((trainset, neg_trainset))
    
itm_keys = list(val_grtr.keys())
test_item_ids = np.array(itm_keys)


np.save('./trainset',trainset_final)

fName = open("./val_grtr.pkl", "wb")
pickle.dump(val_grtr_userbased, fName)
fName.close()

fName = open("./item_dict.pkl", "wb")
pickle.dump(IFeature, fName)
fName.close()

fName = open("./user_dict.pkl", "wb")
pickle.dump(uFeature, fName)
fName.close()

fName = open("./test_item_ids.pkl", "wb")
pickle.dump(test_item_ids, fName)
fName.close()