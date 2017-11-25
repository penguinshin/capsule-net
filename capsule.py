class CapsNet(nn.Module):
    
    def __init__(self):
        super(CapsNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 256, 9, stride = 1)
        self.conv_prim_capsule = nn.Conv2d(256, 32*8, 9, stride = 2)
        self.wij = nn.ModuleList([nn.ModuleList([nn.ModuleList([nn.ModuleList(
                                [nn.Linear(8,16) for l in range(6)]) for k in range(6)]) for j in range(32)])
                                                 for i in range(10)])
        
        self.bij = torch.FloatTensor(32, 6, 6, 10).zero_().view(10, -1)
        self.bij = [Variable(self.bij[i].clone()) for i in range(10)]
    
    def squash(self, x):
        norm = x.norm()
        norm2 = norm ** 2
        x = (norm2/(1+norm2)) * (x/norm)
        return x
    
    def conv_to_prim(self, x):
        x = self.conv_prim_capsule(x).view(-1, 32, 6, 6, 8)
        return x
    
    def prim_to_uhat(self, x):
        u_hat = Variable(torch.FloatTensor(x.size(0), 32, 6, 6, 10, 16))
        for q in range(x.size(0)):
            for i in range(0, 10):
                for j in range(0, 32):
                    for k in range(0, 6):
                        for l in range(0, 6):
                            u_hat[q,j,k,l,i,:] = self.wij[i][j][k][l](x[q,j,k,l,:])
        x = u_hat
        return x
        
    def route(self, x, n_iter = 3):
        x = x.view(-1, 10, 1152, 16)

        outputs = Variable(torch.FloatTensor(x.size(0), 10,16))
        self.bs = []
        for r in range(n_iter):
            self.bs = (self.bij)
        
            cij = [F.softmax(self.bij[i], dim=0) for i in range(len(self.bij))]
            for i in range(0,10):
                v = cij[i].matmul(x[:,i])
                v = self.squash(v)
                outputs[:,i] = v
                
                if r < n_iter-1:
                    for d in range(0, x.size(0)):
                        self.bij[i] = self.bij[i] + torch.matmul(x[d,i,:], v[d])
        return outputs
    
    def forward(self, x):
        return self.route(self.prim_to_uhat(self.conv_to_prim(self.conv1(x))))

def margin_loss(input, target, size_average=True):
    """
    Class loss
    Implement section 3 'Margin loss for digit existence' in the paper.
    """
    batch_size = input.size(0)

    # Implement equation 4 in the paper.

    # ||vc||
    v_c = torch.sqrt((input**2).sum(dim=2, keepdim=True))

    # Calculate left and right max() terms.
    zero = Variable(torch.zeros(1))
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