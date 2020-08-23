import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers.modeling_bert import BertPreTrainedModel, BertModel
import torch.nn.functional as F

class BertForSequenceClassification(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    Examples::
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]
    """

    def __init__(self, config, weight=None):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.reduce = nn.Linear(self.config.visual_features_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        self.classifier2 = nn.Linear(2*config.hidden_size, self.config.num_labels)
        self.weight = weight

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None, vis=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)
        # Complains if input_embeds is kept

        pooled_output = outputs[1] # size (batch_size, 768)
        pooled_output = self.dropout(pooled_output) # size (batch_size, 768)

        if self.config.visual:
            reduced_vis = self.reduce(vis) # size (batch_size, 768)
            if self.config.codebase == 'concatenation':
                combine = torch.cat((pooled_output,reduced_vis),1) # size (batch_size, 1536)
                logits = self.classifier2(combine)
            else:
                combine = torch.mul(pooled_output, reduced_vis)
                logits = self.classifier(combine)
        else:
            logits = self.classifier(pooled_output)
        if !self.config.regression:
            logits = F.sigmoid(logits)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        # if labels is not None:
        #     # if self.num_labels == 1:
        #     #     #  We are doing regression
        #     #     # logits = torch.sigmoid(logits)
        #     #     loss_fct = nn.BCEWithLogitsLoss()
        #     #     loss = loss_fct(logits.view(-1), labels.view(-1))
        #     #     print(logits.view(-1))
        #     # else:
        #       loss_fct = nn.BCELoss(weight=self.weight)
        #       # print(logits.view(-1, self.num_labels))
        #       # print(labels.view(-1).long())
        #       loss = loss_fct(logits.view(-1), labels.view(-1))
        #       outputs = (loss,) + outputs
        if labels is not None:
            if self.config.regression:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.BCELoss(weight=self.weight)
                loss = loss_fct(logits.view(-1), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
