import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward_pass(self, x):
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # 将x传递给LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: (batch_size, seq_length, hidden_size)  (4,2,64)
        # print(out.shape)

        # 将LSTM的最后一个时间步输出传递到全连接层
        out = self.fc(out[:, -1, :])  # (4,256)
        return out

    def forward(self, z_t0, z_t1, z_t2, need_pre=False):
        B = z_t0.shape[0]
        z_t0, z_t1 = z_t0.view(B, -1), z_t1.view(B, -1)
        z_input = torch.stack((z_t0, z_t1), dim=1)  # (4, 2, 2304)
        z_var = torch.var(z_t2)
        z_predict = self.forward_pass(z_input)  # (2,2304)
        z_predict = z_predict.view(z_t2.shape)  # (2,64,6,6)
        loss = torch.mean((z_predict - z_t2)**2) / z_var
        # loss = self.calculate_loss(z_predict, z_t2)
        if need_pre:
            return z_predict, loss
        else:
            return loss
    def calculate_loss(self, inp, pre):
        loss = nn.MSELoss(reduction='mean')
        return loss(inp, pre)

    def prediction(self, z_t0, z_t1):
        z_shape = z_t0.shape
        z_t0, z_t1 = z_t0.view(z_shape[0], -1), z_t1.view(z_shape[0], -1)
        z_input = torch.stack((z_t0, z_t1), dim=1)  # (8,2,256)
        z_predict = self.forward_pass(z_input)  # (4,256)
        z_predict = z_predict.view(z_shape)  # (4,64,2,2)

        return z_predict

