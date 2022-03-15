import torch
import torch.nn as nn

# Inner product of the user and item representations is used to predict the true label
class innerproduct(torch.nn.Module):
    def __init__(self):
        super(innerproduct, self).__init__()

    def forward(self, output1, output2, label):
        inner_product = torch.sum( torch.mul(output1, output2), 1).float()
        euclidean_distance = (label[:,0].float() - inner_product)**2
        loss_contrastive = torch.mean( euclidean_distance)
        return loss_contrastive


# We define the shared model here
class shared_model(nn.Module):

    def __init__(self, item_feature, MLP_user= False):
        super(shared_model, self).__init__()
        self.MLP_user = MLP_user

        # Three fully connected layer for the item input
        self.fully_connected_1 = nn.Sequential(torch.nn.Linear(item_feature, 100), nn.Tanh())
        self.fully_connected_2 = nn.Sequential(torch.nn.Linear(100, 100),nn.Tanh())
        self.fully_connected_3 = nn.Sequential(torch.nn.Linear(100, 100),nn.Tanh())

        # Two fully connected layer on top of the user representation
        self.fully_connected_4 = nn.Sequential(torch.nn.Linear(100, 100),nn.Tanh())
        self.fully_connected_5 = nn.Sequential(torch.nn.Linear(100, 100),nn.Tanh())

    '''
    Makes items representation
    
    aregs:
        inp_vocab: the items' feature vectors
    '''
    def forward_once(self, inp_vocab):

        f_irep = self.fully_connected_1(inp_vocab)
        f_irep = self.fully_connected_2(f_irep)
        f_irep = self.fully_connected_3(f_irep)
        return f_irep


    '''
    Returns both item and user representations
    
    args: 
        u_inp_vocabs: index of the items in the users' support set for each user in the batch
        i_inp_vocab: item features vectors of the items in the batch
        intex_mat: a matrix to combine item representations of the items each user interacted with
        
    '''
    def forward(self, u_inp_vocabs, i_inp_vocab,intex_mat):
        
        pre_output1 = self.forward_once(u_inp_vocabs)
        output2 = self.forward_once( i_inp_vocab)

        output1 = torch.matmul(intex_mat ,pre_output1)

        if self.MLP_user:
            output1 = self.fully_connected_4(output1)
            output1 = self.fully_connected_5(output1)

            
        return output1, output2        
 
 
# We define the shared attention model here
class shared_model_attention(nn.Module):

    def __init__(self, item_feature, attention_type = 'dot', MLP_user  = False):
        super(shared_model_attention, self).__init__()
        
        self.MLP_user = MLP_user

        # Layers for the attention module
        self.attention_type = attention_type
        self.attention = nn.Linear(100, 100)
        
        self.attention_concat = nn.Sequential(torch.nn.Linear(200, 100),nn.Tanh())
        self.vt_concat = nn.Linear(100, 1) #nn.Parameter(torch.FloatTensor(1, 100))

        # Three fully connected layer for the item input
        self.fully_connected_1 = nn.Sequential(torch.nn.Linear(item_feature, 100), nn.Tanh())
        self.fully_connected_2 = nn.Sequential(torch.nn.Linear(100, 100),nn.Tanh())
        self.fully_connected_3 = nn.Sequential(torch.nn.Linear(100, 100),nn.Tanh())
        
        #if self.MLP_user: Two fully connected layer on top of the user representation
        self.fully_connected_4 = nn.Sequential(torch.nn.Linear(100, 100),nn.Tanh())
        self.fully_connected_5 = nn.Sequential(torch.nn.Linear(100, 100),nn.Tanh())

    '''
    Makes items representation

    aregs:
        inp_vocab: the items' feature vectors
    '''
    def forward_once(self, inp_vocab):
        f_irep = self.fully_connected_1(inp_vocab)
        f_irep = self.fully_connected_2(f_irep)
        f_irep = self.fully_connected_3(f_irep)
        return f_irep

    '''
    Returns both item and user representations

    args: 
        u_inp_vocabs: index of the items in the users' support set for each user in the batch
        i_inp_vocab: item features vectors of the items in the batch
        intex_mat: a matrix to combine item representations of the items each user interacted with

    '''

    def forward(self, u_inp_vocabs, i_inp_vocab,intex_mat):
        
        f_urep = self.forward_once(u_inp_vocabs)
        f_irep = self.forward_once( i_inp_vocab)

        XX = torch.matmul(torch.transpose(intex_mat.clone(),0,1),f_irep)

        if(self.attention_type == 'cosin'):
            coss_loss = nn.CosineSimilarity(dim=1, eps=1e-6)
            Z = coss_loss(XX, f_urep)
        elif(self.attention_type == 'dot'):
            Z = torch.sum(torch.mul(XX, f_urep), 1)
        elif(self.attention_type == 'general'):
            u_a_rep = self.attention(f_urep)
            Z = torch.sum(torch.mul(XX, u_a_rep), 1)
            
        Z = torch.unsqueeze(Z,1)
        Z = torch.exp(Z)
        sumA = torch.matmul(intex_mat,Z)
        sumB = torch.matmul(torch.transpose(intex_mat.clone(),0,1),sumA)
        norm_Z = torch.div(Z,sumB)

        f_urep2 = norm_Z * f_urep  # multiply it by the user reps
        f_urep = torch.matmul(intex_mat, f_urep2)  # final user representations: weighted sum

        if self.MLP_user:
            f_urep = self.fully_connected_4(f_urep)
            f_urep = self.fully_connected_5(f_urep) 
        
        return f_urep, f_irep



