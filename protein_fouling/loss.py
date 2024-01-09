import torch

mean_squared_error=torch.nn.MSELoss()
class MyLoss(torch.nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()
    def forward(self, y_true, y_pred):

        scalar = 2.0*mean_squared_error(y_true[:,0],y_pred[:,0]) + 8.0*mean_squared_error(y_true[:,1], y_pred[:,1])
        return scalar