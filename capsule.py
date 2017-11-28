import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


## the comments are meant to be instructive to someone who is trying to 
## navigate the nuances of pytorch autograd and memory management
class CapsNet(nn.Module):
    
    def __init__(self):

        ## i still have no clue what super does, dont ask me
        super(CapsNet, self).__init__()

        ## first conv layer, standard issue, 256 channels, 9x9, stride 1
        self.conv1 = nn.Conv2d(1, 256, 9, stride = 1)

        ## second conv layer, pretending to be 32 capsules of 8 units each, 9x9, stride 2
        self.conv_prim_capsule = nn.Conv2d(256, 32*8, 9, stride = 2)

        ## capsule weights instantiated as a bunch of lists. why not just one giant matrix you ask?
        ## good question, seems like you can't do that without running into in-place problems from
        ## indexing and also pytorch doesn't seem to let you do multi-dimensional layers although
        ## we could implement this without too much hassle...for another time
        self.wij = nn.ModuleList([nn.ModuleList([nn.ModuleList([nn.ModuleList(
                                [nn.Linear(8,16) for l in range(6)]) for k in range(6)]) for j in range(32)])
                                                 for i in range(10)])

        ## log-priors for each of the capsules. note how we make a list instead of a 
        ## 4D tensor. This is because, once again, you can't have in-place operations
        ## and indexing into a tensor is considered an in-place operation
        self.bij = [Variable(torch.FloatTensor(32 * 6 * 6).zero_()) for i in range(10)]
        #self.bij = torch.FloatTensor(32, 6, 6, 10).zero_().view(10, -1)
        #self.bij = [Variable(self.bij[i].clone()) for i in range(10)]

    def squash(self, x):

        # our beloved non-linearity. squashes
        norm = x.norm()
        norm2 = norm ** 2
        x = (norm2/(1+norm2)) * (x/norm)
        return x
    
    def conv_to_prim(self, x):

        # recast primary capsule output with batch size explicitely
        x = self.conv_prim_capsule(x).view(-1, 32, 6, 6, 8)
        return x
    
    def prim_to_uhat(self, x):

        u_hat = Variable(torch.FloatTensor(x.size(0), 32, 6, 6, 10, 16))

        # this is where we could use some improvement...but how to avoid all the loops?
        # now, you might wonder, isn't u_hat being indexed into and thus in-place operation?
        # great question, the answer is that you can do this as long as nothing else depends
        # on u_hat in a horizontal fashion. i.e., besides the thing that depends on it 
        # immediately, there's not some other part of the graph that acceses mid-versions 
        # of the ever-changing variable
        for q in range(x.size(0)):
            for i in range(0, 10):
                for j in range(0, 32):
                    for k in range(0, 6):
                        for l in range(0, 6):
                            u_hat[q,j,k,l,i,:] = self.wij[i][j][k][l](x[q,j,k,l,:])
        
        return u_hat
        
    def route(self, x, n_iter = 3):

        x = x.view(-1, 10, 1152, 16)

        # we don't actually need outputs to be a list since nothing depends on it
        # but we did it anyways just cus...
        outputs = [Variable(torch.FloatTensor(x.size(0),16)) for i in range(10)]
        for r in range(n_iter):
            self.cij = [F.softmax(self.bij[i], dim=0) for i in range(len(self.bij))]
            for i in range(0,10):
                v = self.cij[i].matmul(x[:,i])
                v = self.squash(v)
                outputs[i] = v
                
                if r < n_iter-1:
                    for d in range(0, x.size(0)):
                        self.bij[i] = self.bij[i] + torch.matmul(x[d,i,:], v[d])

        # if you want to go on a wild ride, try removing this line
        # spoiler alert, if you allow a variable to persist in memory by
        # attaching it to an object, you'll notice that after the first batch,
        # pytorch will throw an error saying that your buffers have been freed.
        # we don't want to save buffers because that leads to explosive memory
        # so instead we make sure to detach our variables from the gradient
        # graph so it doesn't attempt to track them
        self.bij = [t.detach() for t in self.bij] 

        return torch.stack(outputs).view(-1,10,16)
    
    def forward(self, x):

        return self.route(self.prim_to_uhat(self.conv_to_prim(self.conv1(x))))

def debug_loss(input, target):

    return input.mean()

def margin_loss(input, target, size_average=True):
    
    """
    Class loss
    Implement section 3 'Margin loss for digit existence' in the paper.
    """
    batch_size = input.size(0)
    print(batch_size)
    # Implement equation 4 in the paper.

    # ||vc||
    v_c = torch.sqrt((input**2).sum(dim=2, keepdim=True))

    # Calculate left and right max() terms.
    zero = Variable(torch.zeros(1)).detach()
    m_plus = 0.9
    m_minus = 0.1
    loss_lambda = 0.5
    max_left = torch.max(m_plus - v_c, zero).view(batch_size, -1)**2
    max_right = torch.max(v_c - m_minus, zero).view(batch_size, -1)**2
    t_c = target.type(torch.FloatTensor)
    # Lc is margin loss for each digit of class c
    
    l_c = t_c * max_left + loss_lambda * (1.0 - t_c) * max_right
    l_c = l_c.sum(dim=1)

    if size_average:
        l_c = l_c.mean()

    return l_c