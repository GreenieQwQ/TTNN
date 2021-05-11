from torch import nn

# 忽略pad的loss function
class TokenCrossEntropyLoss(nn.Module):

    def __init__(self, pad_index=0):
        super(TokenCrossEntropyLoss, self).__init__()

        self.pad_index = pad_index
        self.base_loss_function = nn.CrossEntropyLoss(reduction='sum', ignore_index=pad_index)

    def forward(self, outputs, targets):
        batch_size, seq_len, vocabulary_size = outputs.size()

        outputs_flat = outputs.reshape(batch_size * seq_len, vocabulary_size)
        targets_flat = targets.reshape(batch_size * seq_len)

        batch_loss = self.base_loss_function(outputs_flat, targets_flat)

        count = (targets != self.pad_index).sum().item()

        return batch_loss, count
