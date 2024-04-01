from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn

from logging_utils import logger


class FocalLoss(nn.Module):
    def __init__(
        self,
        alpha: Optional[float] = 0.5,
        gamma: float = 2.0,
        reduction: str = "mean",
        ignore_index: int = -100,
        epsilon=1.0e-9,
        activation_type="softmax",
    ) -> None:
        super(FocalLoss, self).__init__()
        if alpha is not None and not isinstance(alpha, float):
            raise ValueError(
                f"alpha value should be None, float value or list of real values. Got: {type(alpha)}"
            )
        self.alpha: Optional[Union[float, torch.Tensor]] = (
            alpha
            if alpha is None or isinstance(alpha, float)
            else torch.FloatTensor(alpha)
        )
        if isinstance(alpha, float) and not 0.0 <= alpha <= 1.0:
            logger.warn("[Focal Loss] alpha value is to high must be between [0, 1]")

        self.gamma: float = gamma
        self.reduction: str = reduction
        self.ignore_index: int = ignore_index
        self.epsilon = epsilon
        self.activation_type: str = activation_type

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the criterion.
        @param input: model's output, shape [N, C] meaning N: batch_size and C: num_classes
        @param target: ground truth, shape [N], N: batch_size
        @return: shape of [N]
        """
        if not torch.is_tensor(input):
            raise TypeError(
                "Input type is not a torch.Tensor. Got {}".format(type(input))
            )
        if input.device != target.device:
            raise ValueError(
                "input and target must be in the same device. Got: {}".format(
                    input.device, target.device
                )
            )

        # filter labels
        target = target.type(torch.long)

        if self.activation_type == "sigmoid":
            multi_hot_key = target
            logits = torch.sigmoid(input)
            zero_hot_key = 1 - multi_hot_key
            focal_loss = (
                -self.alpha
                * multi_hot_key
                * torch.pow((1 - logits), self.gamma)
                * (logits + self.epsilon).log()
            )
            focal_loss += (
                -(1 - self.alpha)
                * zero_hot_key
                * torch.pow(logits, self.gamma)
                * (1 - logits + self.epsilon).log()
            )
            weights = torch.ones_like(
                focal_loss, dtype=focal_loss.dtype, device=focal_loss.device
            )
        else:
            input_mask = target != self.ignore_index
            target = target[input_mask]
            input = input[input_mask]
            # compute softmax over the classes axis
            pt = F.softmax(input, dim=1)
            logpt = F.log_softmax(input, dim=1)

            # compute focal loss
            pt = pt.gather(1, target.unsqueeze(-1)).squeeze()
            logpt = logpt.gather(1, target.unsqueeze(-1)).squeeze()
            focal_loss = -1 * (1 - pt) ** self.gamma * logpt

            weights = torch.ones_like(
                focal_loss, dtype=focal_loss.dtype, device=focal_loss.device
            )
            if self.alpha is not None:
                if isinstance(self.alpha, float):
                    alpha = torch.tensor(self.alpha, device=input.device)
                    weights = torch.where(target > 0, 1 - alpha, alpha)
                elif torch.is_tensor(self.alpha):
                    alpha = self.alpha.to(input.device)
                    weights = alpha.gather(0, target)

        tmp_loss = focal_loss * weights
        if self.reduction == "none":
            loss = tmp_loss
        elif self.reduction == "mean":
            loss = (
                tmp_loss.sum() / weights.sum()
                if torch.is_tensor(self.alpha)
                else torch.mean(tmp_loss)
            )
        elif self.reduction == "sum":
            loss = tmp_loss.sum()
        else:
            raise NotImplementedError(
                "Invalid reduction mode: {}".format(self.reduction)
            )
        return loss


def multilabel_categorical_crossentropy(y_pred, y_true):
    """多标签分类的交叉熵
        y_true 和 y_pred 的 shape 一致、y_true的元素非0即1
        1表示对应的类为目标类、0表示对应的类为非目标类。
    警告：请保证 y_pred 的值域是全体实数，换言之一般情况下 y_pred
        不用加激活函数，尤其是不能加 sigmoid或者softmax
        预测阶段则输出 y_pred 大于 0 的类。
    """
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = torch.zeros_like(y_pred[:, :1])
    output_neg = torch.cat([y_pred_neg, zeros], dim=1)
    output_pos = torch.cat([y_pred_pos, zeros], dim=1)
    loss_neg = torch.logsumexp(output_neg, dim=1)
    loss_pos = torch.logsumexp(output_pos, dim=1)
    loss = (loss_neg + loss_pos).sum()
    return loss


def compute_kl_loss(p, q, pad_mask=None, r=4):
    # 计算KL散度
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction="none")
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction="none")

    # pad_mask is for seq-level tasks
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.0)
        q_loss.masked_fill_(pad_mask, 0.0)

    # You can choose whether to use function "sum" and "mean" depending on your task
    p_loss = p_loss.sum()
    q_loss = q_loss.sum()

    loss = (p_loss + q_loss) / r
    return loss
