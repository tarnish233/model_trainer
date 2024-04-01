from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import BertModel, BertPreTrainedModel
from transformers.modeling_outputs import (
    SequenceClassifierOutput,
    TokenClassifierOutput,
)

from model_loss import FocalLoss, multilabel_categorical_crossentropy
from model_outputs import MultiTaskBertForSequenceAndTokenClassifierOutput


class CustomerBertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.loss_type = config.loss_type
        self.bert = BertModel(config)
        self.label_weights = config.label_weights
        self.label_weights = (
            torch.tensor(self.label_weights).cuda()
            if self.label_weights is not None
            else None
        )
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                if self.loss_type == "FocalLoss":
                    loss_fct = FocalLoss(alpha=0.75, gamma=2)
                    loss = loss_fct(logits, labels)
                elif self.loss_type == "CrossEntropyLoss":
                    loss_fct = CrossEntropyLoss(weight=self.label_weights)
                    # loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                else:
                    raise ValueError("loss_type must be focal_loss or cross_entropy")

            elif self.config.problem_type == "multi_label_classification":
                if self.loss_type == "ZLPR":
                    loss = multilabel_categorical_crossentropy(logits, labels)
                elif self.loss_type == "FocalLoss":
                    loss_fct = FocalLoss(alpha=0.75, gamma=2, is_multilabel=True)
                    loss = loss_fct(logits, labels)
                elif self.loss_type == "BCEWithLogitsLoss":
                    loss_fct = BCEWithLogitsLoss(weight=self.label_weights)
                    loss = loss_fct(logits, labels)
                else:
                    raise ValueError("loss_type must be ZLPR or BCE")

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class BertForTokenClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class MultiTaskBertForSequenceAndTokenClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.num_token_labels = config.num_token_labels
        self.config = config
        self.loss_type = config.loss_type
        # self.use_r_drop = config.args.use_r_drop
        self.bert = BertModel(config)
        self.label_weights = config.label_weights
        self.label_weights = (
            torch.tensor(self.label_weights).cuda()
            if self.label_weights is not None
            else None
        )
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.token_classifier = nn.Linear(config.hidden_size, config.num_token_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        token_labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Tuple[
        Union[Tuple[torch.Tensor], SequenceClassifierOutput],
        Union[Tuple[torch.Tensor], TokenClassifierOutput],
    ]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        token_labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output, pooled_output = outputs[:2]

        # For sequence classification
        pooled_output = self.dropout(pooled_output)
        cls_logits = self.classifier(pooled_output)

        # For token classification
        sequence_output = self.dropout(sequence_output)
        token_logits = self.token_classifier(sequence_output)
        total_loss, cls_loss, token_loss = None, None, None

        # Loss
        if labels is not None and token_labels is not None:
            # For sequence cls loss
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    cls_loss = loss_fct(cls_logits.squeeze(), labels.squeeze())
                else:
                    cls_loss = loss_fct(cls_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                if self.loss_type == "FocalLoss":
                    loss_fct = FocalLoss(alpha=0.75, gamma=2)
                    cls_loss = loss_fct(cls_logits, labels)
                elif self.loss_type == "CrossEntropyLoss":
                    loss_fct = CrossEntropyLoss()
                    cls_loss = loss_fct(
                        cls_logits.view(-1, self.num_labels), labels.view(-1)
                    )
                else:
                    raise ValueError("loss_type must be focal_loss or cross_entropy")
            elif self.config.problem_type == "multi_label_classification":
                if self.loss_type == "ZLPR":
                    cls_loss = multilabel_categorical_crossentropy(cls_logits, labels)
                elif self.loss_type == "FocalLoss":
                    loss_fct = FocalLoss(alpha=0.75, gamma=2, is_multilabel=True)
                    cls_loss = loss_fct(cls_logits, labels)
                elif self.loss_type == "BCEWithLogitsLoss":
                    loss_fct = BCEWithLogitsLoss(weight=self.label_weights)
                    cls_loss = loss_fct(cls_logits, labels)
                else:
                    raise ValueError("loss_type must be ZLPR or BCE")

            # For token cls loss
            loss_fct = CrossEntropyLoss()
            token_loss = loss_fct(
                token_logits.view(-1, self.num_token_labels), token_labels.view(-1)
            )

            total_loss = cls_loss + token_loss

        if not return_dict:
            output = (cls_logits, token_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return MultiTaskBertForSequenceAndTokenClassifierOutput(
            loss=total_loss,
            cls_logits=cls_logits,
            token_logits=token_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# 新增标签
class UpdateLabelBertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels1 = config.num_labels1
        self.num_labels2 = config.num_labels2
        self.config = config
        self.loss_type = config.loss_type
        self.bert = BertModel(config)
        self.label_weights = config.label_weights
        self.label_weights = (
            torch.tensor(self.label_weights).cuda()
            if self.label_weights is not None
            else None
        )
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier1 = nn.Linear(config.hidden_size, config.num_labels1)
        self.classifier2 = nn.Linear(config.hidden_size, config.num_labels2)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels1: Optional[torch.Tensor] = None,
        labels2: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)

        loss = None
        labels = None
        if labels1 is not None:
            labels = labels1
            self.num_labels = self.num_labels1
            logits = self.classifier1(pooled_output)
        else:
            labels = labels2
            self.num_labels = self.num_labels2
            logits = self.classifier2(pooled_output)
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels1.dtype == torch.long or labels1.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                if self.loss_type == "FocalLoss":
                    loss_fct = FocalLoss(alpha=0.75, gamma=2)
                    loss = loss_fct(logits, labels)
                elif self.loss_type == "CrossEntropyLoss":
                    loss_fct = CrossEntropyLoss(weight=self.label_weights)
                    # loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                else:
                    raise ValueError("loss_type must be focal_loss or cross_entropy")

            elif self.config.problem_type == "multi_label_classification":
                if self.loss_type == "ZLPR":
                    loss = multilabel_categorical_crossentropy(logits, labels)
                elif self.loss_type == "FocalLoss":
                    loss_fct = FocalLoss(alpha=0.75, gamma=2, is_multilabel=True)
                    loss = loss_fct(logits, labels)
                elif self.loss_type == "BCEWithLogitsLoss":
                    loss_fct = BCEWithLogitsLoss(weight=self.label_weights)
                    loss = loss_fct(logits, labels)
                else:
                    raise ValueError("loss_type must be ZLPR or BCE")

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
