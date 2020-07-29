import torch
from torch import nn


class BaseModel(nn.Module):

    def __init__(self):
        super().__init__()

    # 训练
    def fit(self,
            train_data_loader,
            eval_data,
            epochs,
            lr,
            model_path):
        eval_x, eval_y = eval_data

        eval_p = eval_x[0]
        eval_q = eval_x[1]

        eval_p = torch.from_numpy(eval_p)
        eval_q = torch.from_numpy(eval_q)
        if torch.cuda.is_available():
            eval_p = eval_p.cuda()
            eval_q = eval_q.cuda()
        if torch.cuda.is_available():
            self = self.cuda()

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_func = nn.CrossEntropyLoss()

        best_acc = 0

        for epoch in range(epochs):
            for step, (b_p, b_q, b_y) in enumerate(train_data_loader):
                if torch.cuda.is_available():
                    b_p = b_p.cuda()
                    b_q = b_q.cuda()
                    b_y = b_y.cuda()
                output = self(b_p, b_q)
                loss = loss_func(output, b_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if step % 20 == 0:
                    test_output = self(eval_p, eval_q)
                    pred_y = torch.max(test_output, 1)[1].cpu().data.numpy()
                    accuracy = float((pred_y == eval_y).astype(int).sum()) / float(eval_y.size)
                    if accuracy > best_acc:
                        best_acc = accuracy
                        self.save_model(model_path)
                        print('save model, accuracy: %.2f' % accuracy)
                    print('Epoch: ', epoch, '| train loss: %.4f' % loss.cpu().data.numpy(),
                          '| test accuracy: %.2f' % accuracy)

    # 保存模型
    def save_model(self, model_path):
        torch.save(self.state_dict(), model_path)
