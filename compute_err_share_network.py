import numpy as np
import math
import torch

'''
Creates a batch of the user and items features for training

args:
    np_uid: user ids in the batch.
    np_iid: item ids in the batch.
    IFeature: a dictionary, where "key" is an item_id and the value is its feature vector. 
    uFeature: a dictionary, where "key" is an user_id and the value is id of the items purchased by that user.
    attn: True if we use an attention model, False otherwise.
    user_data: True if we need to return user features in addition to the item features, False otherwise.
    remove_item: True if we want to remove the item from the support set of the user, False otherwise.
    
'''

def create_batch(np_uid, np_iid, IFeature, uFeature, attn=False, user_data=True, remove_item=False):
    # size of the batch
    cu_size = np_iid.shape[0]

    max_vocab = len(IFeature[1])

    # This is the item feature vector.
    inp_i_vocab = np.zeros((cu_size, max_vocab))

    # Extract all the information from uFeature and IFeature
    total_len = 0 # total number of items in the support set of the users in this batch
    for ii in range(cu_size):
        iid = np.asscalar(np_iid[ii])
        uid = np.asscalar(np_uid[ii])

        if remove_item:
            user_supportset = np.setdiff1d(uFeature[uid], iid)
        else:
            user_supportset = uFeature[uid]

        total_len = total_len + len(user_supportset)

    # This is the user feature vector
    inp_u_vocab = np.zeros((total_len,max_vocab))
    inp_u_index = np.zeros((total_len,))

    # We create user and item feature vectors for all the users and items in this batch
    index = 0
    for ii in range(cu_size):

        iid = np.asscalar(np_iid[ii])
        inp_i_vocab[ii, 0:len(IFeature[iid])] = IFeature[iid]
        uid = np.asscalar(np_uid[ii])
        # print(len(uFeature[uid]))

        if user_data:
            if remove_item:
                # print("remove item in user's support set.")
                user_supportset = np.setdiff1d(uFeature[uid], iid)
            else:
                user_supportset = uFeature[uid]

            inp_u_index[index:index+len(user_supportset)] = ii

            for jj in user_supportset:
                inp_u_vocab[index, 0:len(IFeature[jj])] = IFeature[jj]
                index = index + 1

    # The "indext_mat" is a matrix, which determines which items correspond to each user
    if user_data:
        b_size = cu_size
        f_b_size = inp_u_index.shape[0]
        intex_mat = np.zeros((b_size, f_b_size))
        intex_mat[inp_u_index.astype(int), np.arange(0, f_b_size, 1).astype(int)] = 1
        if(not attn):
            m = np.sum(intex_mat, axis=1).reshape(-1, 1)
            intex_mat = intex_mat / m
    else:
        intex_mat = None

    return inp_u_index, inp_u_vocab, inp_i_vocab, intex_mat

'''
Computes recall for the shared representation model

args:
    model: the neural network model
    IFeature: a dictionary, where "key" is an item_id and the value is its feature vector. 
    uFeature: a dictionary, where "key" is an user_id and the value is id of the features purchased by that user.
    test_item_ids: id of the test items
    val_grtr_userbased: It's a dictionry, where the key is the user id. For each user id u, 
                        the value is a list of the format [a,b] 
                        a contains ids of the items that are purchased by that user
                        b contains ids of the items that are not purchased that user
    K: number of retrieved neighbors in computing the KNNs 
    device: cuda or cpu
    batch_size: Number of pairs given to the model at each time in computing the recall 
    attn: is True if we use an attention model, False otherwise.
    remove_item: True if we want to remove the item from the support set of the user    val_grtr_userbased: It's a dictionry, where the key is the user id. For each user id u, 
                        the value is a list of the format [a,b] 
                        a contains ids of the items that are purchased by that user
                        b contains ids of the items that are not purchased that user.  

'''
def comp_recall_userbased(model,IFeature, uFeature,test_item_ids,val_grtr_userbased, K, device, batch_size,attn=False,remove_item = False):

    user_keys = val_grtr_userbased.keys()
    user_ids = list(user_keys)
    t_itm_ids = np.ones_like(user_ids) #  we don't care about item ids.
    
    
    ############# calculate user representations ###################
    num_iters = math.ceil( len(user_ids) / batch_size)
    u_reps = []
    for batch_num in range(num_iters):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, len(user_ids))
        inp_u_index, inp_u_vocabs,  inp_i_vocab, intex_mat = create_batch(user_ids[start_index:end_index], t_itm_ids[start_index:end_index], IFeature, uFeature,attn=attn, remove_item = remove_item)

        if device == 'cuda':
            inp_u_index, inp_u_vocabs,inp_i_vocab = torch.from_numpy(inp_u_index).cuda().float(), torch.from_numpy(inp_u_vocabs).cuda().float(), torch.from_numpy(inp_i_vocab).cuda().float()
        else:    
            inp_u_index, inp_u_vocabs,inp_i_vocab= torch.from_numpy(inp_u_index).float(), torch.from_numpy(inp_u_vocabs).float(), torch.from_numpy(inp_i_vocab).float()
        

        #u_rep contains all user representations
        with torch.no_grad():
            pre_output1 = model.forward_once(inp_u_vocabs)
            output1 = [torch.mean(pre_output1[inp_u_index == i],dim=0) for i in range(len(torch.unique(inp_u_index)))]
                
            u_reps.append( torch.stack(output1) )
     
    u_reps = torch.cat(u_reps,0).cpu()
    

    
    ############# calculate i_reps ###################

    t_usr_ids = np.ones_like(test_item_ids) #We do not care about user_ids=fake ids

    #Get the items' input vectors
    inp_u_index, inp_u_vocabs, inp_i_vocab, intex_mat = create_batch(t_usr_ids, test_item_ids, IFeature, uFeature, attn=False,
                                 user_data = False, remove_item = remove_item )
 
    if device == 'cuda':
        inp_i_vocab = torch.from_numpy(inp_i_vocab).cuda().float()
    else:
        inp_i_vocab= torch.from_numpy(inp_i_vocab).float()

    with torch.no_grad():
        i_reps = model.forward_once(inp_i_vocab)
    i_reps = i_reps.cpu()
    recall = 0

    # loop over each cold-start item
    for count,usr in enumerate(user_keys):
        # usrs[0]= id of the users who purchased that item
        # usrs[1]= id of the users who did not purchased that item
        items = val_grtr_userbased[usr]

        u_rep = u_reps[count,:] # item representations
        i_rep = i_reps # user representation of all the users in usrs[0] and usrs[1]


        e_dist = torch.sum( torch.mul(u_rep, i_rep), 1)


        # Find the K smallest indics in e_dist
        midx = (np.argpartition(e_dist.flatten(), -K)).tolist()
        midx = test_item_ids[midx[-K:]]
            
        #indices 0 to len(usrs[0])-1 are the groundtruth indices
        grtr = items[0]        
        #compute the intersection of the K retrieved users and the groundtruth
        intersect = list(set(midx) & set(grtr))
        recall = recall + len(intersect)/len(grtr)


    recall = recall/len(val_grtr_userbased)

    return recall


'''
Computes recall for the shared attentional model

args:
    model: the neural network (attentional) model
    IFeature: a dictionary, where "key" is an item_id and the value is its feature vector. 
    uFeature: a dictionary, where "key" is an user_id and the value is id of the features purchased by that user.
    val_grtr_userbased: It's a dictionry, where the key is the user id. For each user id u, 
                        the value is a list of the format [a,b] 
                        a contains ids of the items that are purchased by that user
                        b contains ids of the items that are not purchased that user
    K: number of retrieved neighbors in computing the KNNs 
    device: cuda or cpu 
    attn: is True if we use an attention model, False otherwise.
    remove_item: True if we want to remove the item from the support set of the user, False otherwise.  

'''
    
def comp_recall_userbased_4attention(model, IFeature, uFeature,val_grtr_userbased, K, device,
                                     remove_item=False, attn=True):

    train_itm_keys = list(range(IFeature.shape[0]))

    train_item_ids = np.array(train_itm_keys)
    t_usr_ids = np.ones_like(train_item_ids)  # We do not care about user_ids=fake ids

    # Create the batch
    train_inp_u_index, train_inp_u_vocabs, train_inp_i_vocab, intex_mat = create_batch(t_usr_ids, train_item_ids,
                                                                                       IFeature, uFeature,
                                                                                       user_data=False,
                                                                                       remove_item=remove_item,
                                                                                       attn=attn)
    if device == 'cuda':
        train_inp_i_vocab = torch.from_numpy(train_inp_i_vocab).cuda().float()
    else:
        train_inp_i_vocab = torch.from_numpy(train_inp_i_vocab).float()

    with torch.no_grad():
        # ires contains all "cold-start" item representations
        train_i_reps = model.forward_once(train_inp_i_vocab)
    train_i_reps = train_i_reps

    user_keys = val_grtr_userbased.keys()
    user_ids = list(user_keys)
    user_ids = np.random.permutation(user_ids)

    recall = 0


    for count, uid in enumerate(user_ids):

        items = val_grtr_userbased[uid]
        all_items = items[0] + items[1]
        u_inp_pre_output1 = train_i_reps[
            uFeature[uid]]
        i_output2 = train_i_reps[items[0]]

        i_output2 = i_output2.cpu()
        u_inp_pre_output1 = u_inp_pre_output1.cpu()
        
        iR = i_output2.numpy().copy()
        uR = u_inp_pre_output1.numpy().copy()

        if (model.attention_type == 'cosin'):
            ciR = (iR / np.linalg.norm(iR, axis=1, keepdims=True))
            cuR = (uR / np.linalg.norm(uR, axis=1, keepdims=True))
            Z = np.matmul(cuR, ciR.T)
        elif(model.attention_type == 'dot'):
            Z = np.matmul(uR, iR.T)
        elif (model.attention_type == 'general'):
            if device == 'cuda':
                temp = torch.from_numpy(uR).cuda().float()
            else:
                temp = torch.from_numpy(uR).float()
            temp = model.attention(temp)
            temp = temp.cpu().detach().numpy()
            Z = np.matmul(temp, iR.T)
        
        
        Z = np.exp(Z)
        Z = Z / np.sum(Z, axis=0)

        output1_ps1 = np.matmul(Z.T, uR)

        if model.MLP_user:
            if device == 'cuda':
                output1_ps1 = torch.from_numpy(output1_ps1).cuda().float()
            else:
                output1_ps1 = torch.from_numpy(output1_ps1).float()
            output1_ps1 = model.fully_connected_4(output1_ps1)
            output1_ps1 = model.fully_connected_5(output1_ps1)
            output1_ps1 = output1_ps1.cpu().detach().numpy()
        
        e_dist_ps = np.sum((output1_ps1 * iR), axis=1)
        u_inp_pre_output1 = train_i_reps[uFeature[uid]]
        i_output2 = train_i_reps[items[1]]

        i_output2 = i_output2.cpu()
        u_inp_pre_output1 = u_inp_pre_output1.cpu()
        
        iR = i_output2.numpy().copy()
        uR = u_inp_pre_output1.numpy().copy()

        if (model.attention_type == 'cosin'):
            ciR = (iR / np.linalg.norm(iR, axis=1, keepdims=True))
            cuR = (uR / np.linalg.norm(uR, axis=1, keepdims=True))
            Z = np.matmul(cuR, ciR.T)
        elif(model.attention_type == 'dot'):
            Z = np.matmul(uR, iR.T)
        elif (model.attention_type == 'general'):
            if device == 'cuda':
                temp = torch.from_numpy(uR).cuda().float()
            else:
                temp = torch.from_numpy(uR).float()
            temp = model.attention(temp)
            temp = temp.cpu().detach().numpy()
            Z = np.matmul(temp, iR.T)
        
        Z = np.exp(Z)
        Z = Z / np.sum(Z, axis=0)

        output1_ns1 = np.matmul(Z.T,uR)

        if model.MLP_user:
            if device == 'cuda':
                output1_ns1 = torch.from_numpy(output1_ns1).cuda().float()
            else:
                output1_ns1 = torch.from_numpy(output1_ns1).float()
            output1_ns1 = model.fully_connected_4(output1_ns1)
            output1_ns1 = model.fully_connected_5(output1_ns1)
            output1_ns1 = output1_ns1.cpu().detach().numpy()
        
        e_dist_ns = np.sum((output1_ns1 * iR), axis=1)

        e_dist = np.concatenate([e_dist_ps, e_dist_ns])
        #
        midx = (np.argpartition(e_dist.flatten(), -K)).tolist()
        midx = [all_items[itm] for itm in midx[-K:]]

        # indices 0 to len(usrs[0])-1 are the groundtruth indices
        grtr = items[0]
        # compute the intersection of the K retrieved users and the groundtruth
        intersect = list(set(midx) & set(grtr))
        recall = recall + len(intersect) / len(grtr)
    recall = recall / len(val_grtr_userbased)

    return recall

