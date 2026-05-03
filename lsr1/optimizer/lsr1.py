import torch
import torch.nn as nn

MAX = 1e30
MAX_LR = 10.0

class LSR1Optimizer(nn.Module):
    def __init__(self,args):
        super(LSR1Optimizer,self).__init__()
        self.args = args
        self.ang_t = args.inner_ang_t
        self.dir_vec_eps = args.inner_dir_vec_eps
        self.enc = MLPBlock(input_dim=5,output_dim=args.inner_dim)
        self.Bk = InvHessianEstimator(L=args.inner_buffer_size)
        self.get_v = SR1DirectionVectorsGenerator(args,eps=self.dir_vec_eps)
        self.get_lr = LearningRateGenerator(args)
        self.iter = 0
        self.x = None
        self.gx = None
        return
    
    def reset(self):
        self.Bk.reset()
        self.x = None
        self.gx = None
        self.iter = 0
    
    def update(self,next_x,next_gx):
        dx = (next_x - self.x) if self.x is not None else next_x
        dg = (next_gx - self.gx) if self.gx is not None else next_gx
        bkdg = self.Bk(dg)
        features = torch.stack([next_x,dx,bkdg,next_gx,dg],dim=2)
        fin = self.enc(features.detach())
        v = self.get_v(fin)
        self.Bk.update_matrices(v)

        return bkdg, dx, fin
    
    def forward(self,x,grad):
        bkdg,dx,fin = self.update(x,grad)
        self.x = torch.clone(x)
        self.gx = torch.clone(grad)
        bkg = self.Bk(self.gx)
        dk = -bkg
        lr = self.get_lr(fin)
        lr = torch.clamp_max(lr,MAX_LR)
        update = lr * dk
        self.iter += 1
        return update, (bkdg,dx), lr

    def angle(self,A,B):
        return (A*B).sum() / ((A**2).sum() * (B**2).sum()).sqrt()


class LearningRateGenerator(nn.Module):
    def __init__(self,args) -> None:
        super().__init__()
        self.alpha1 = args.alpha1
        self.alpha2 = args.alpha2
        self.mlp = MLPBlock(input_dim=args.inner_dim,output_dim=1)

    def forward(self,F_in):
        out = self.mlp(F_in)
        out = out.squeeze()
        return self.alpha1 * torch.exp(self.alpha2 * out)


class InputEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim) -> None:
        super().__init__()
        self.fc_in = torch.nn.Linear(in_features=in_dim, out_features=hidden_dim)
    
    def forward(self,dX,BkdG,dG):
        feat_in = torch.cat([dX - BkdG, dX, BkdG, dG],axis=2)
        return self.fc_in(feat_in)

class SR1DirectionVectorsGenerator(nn.Module):
    def __init__(self,args,eps=1e-5) -> None:
        super().__init__()
        self.l_arch = args.inner_l_arch
        assert(self.l_arch == 'mlp')
        self.mlp = MLPBlock(input_dim=args.inner_dim,output_dim=1)
        self.eps = eps
    
    def forward(self,F_in):
        V = self.mlp(F_in)
        return V

class InvHessianEstimator:
    def __init__(self,L=100) -> None:
        self.L = L
        self.squarred = True
        self.reset()
        pass

    def __call__(self, X):
        if len(self.Vmats) == 0:
            return X #X
        
        elif not self.full:
            X = X.unsqueeze(2)
            
            if self.squarred:
                BkX_ = sum([X] + [V_ * bmatip(V_,X) for V_ in self.Vmats])
                X_n, BkX_n = (X**2).sum(dim=(1,2),keepdim=True).sqrt(), (BkX_**2).sum(dim=(1,2),keepdim=True).sqrt()
                Dk = BkX_ * X_n / BkX_n

            else:
                BkX_ = sum([X] + [U_ * bmatip(V_,X) for V_,U_ in zip(self.Vmats, self.Umats)])
                X_n, BkX_n = (X**2).sum(dim=(1,2),keepdim=True).sqrt(), (BkX_**2).sum(dim=(1,2),keepdim=True).sqrt()
                Dk = BkX_ * X_n / BkX_n
                
            return Dk.squeeze()
        
        else:
            X = X.unsqueeze(2)

            if self.squarred:
                BkX_ = sum([V_ * bmatip(V_,X) for V_ in self.Vmats])
                X_n, BkX_n = (X**2).sum(dim=(1,2),keepdim=True).sqrt(), (BkX_**2).sum(dim=(1,2),keepdim=True).sqrt()
                Dk = BkX_ * X_n / BkX_n

            else:
                BkX_ = sum([X] + [U_ * bmatip(V_,X) for V_,U_ in zip(self.Vmats, self.Umats)])
                X_n, BkX_n = (X**2).sum(dim=(1,2),keepdim=True).sqrt(), (BkX_**2).sum(dim=(1,2),keepdim=True).sqrt()
                Dk = BkX_ * X_n / BkX_n

            return Dk.squeeze()
    
    def reset(self,L=None):
        if L is not None:
            self.L = L
        self.counter = 0
        self.full = False
        self.Vmats = []
        if not self.squarred:
            self.Umats = []
        pass

    def resize_buffer(self,Lnew):
        if self.L < Lnew:
            self.full = False
            self.L = Lnew
        pass
    
    def update_matrices(self,uv):        
        if self.squarred:
            V = uv

            if self.full:
                self.Vmats[self.counter % self.L] = V
            else:
                self.Vmats.append(V)

        else:
            U,V = uv
            
            if self.full:
                self.Umats[self.counter % self.L] = U
                self.Vmats[self.counter % self.L] = V
            else:
                self.Umats.append(U)
                self.Vmats.append(V)
        
        self.counter+=1
        if self.counter == self.L:
            self.full = True
            self.counter = 0
        pass

    def pop_matrices(self):
        if not self.squarred:
            self.Umats[self.counter-1 % self.L] = torch.zeros_like(self.Umats[self.counter-1 % self.L])
        self.Vmats[self.counter-1 % self.L] = torch.zeros_like(self.Vmats[self.counter-1 % self.L])
        self.counter-=1
        pass

class BasicBlock(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, p_dropout=0.2):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.prelu = nn.PReLU()
        self.do1 = nn.Dropout(p_dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)
        self.do2 = nn.Dropout(p_dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x.permute(0,2,1)).permute(0,2,1)
        x = self.prelu(x)
        x = self.do1(x)
        x = self.fc2(x)
        x = self.bn2(x.permute(0,2,1)).permute(0,2,1)
        x = self.prelu(x)
        x = self.do2(x)
        return x


class MLPBlock(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, p_dropout=0.2):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.prelu = nn.PReLU()
        self.do = nn.Dropout(p_dropout)
        self.mlp1 = BasicBlock(hidden_dim, hidden_dim, hidden_dim)
        self.mlp2 = BasicBlock(hidden_dim, hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x.permute(0,2,1)).permute(0,2,1)
        x = self.prelu(x)
        x = self.do(x)
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.fc2(x)
        return x
    
def bmatip(bm1,bm2):
    assert(bm1.shape == bm2.shape)
    b = bm1.shape[0]
    return torch.stack([(v * dg).sum() for v,dg in zip(bm1,bm2)],dim=0).reshape(b,1,1)
