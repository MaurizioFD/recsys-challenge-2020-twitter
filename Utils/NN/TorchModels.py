import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss
from transformers import DistilBertModel

from Utils.Eval.Metrics import ComputeMetrics as CoMe


class FFNNMulti(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, hidden_dropout_prob_1, hidden_dropout_prob_2):
        super(FFNNMulti, self).__init__()

        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.hidden_dropout_prob_1 = hidden_dropout_prob_1
        self.hidden_dropout_prob_2 = hidden_dropout_prob_2

        self.dropout_1 = nn.Dropout(hidden_dropout_prob_1)
        self.dropout_2 = nn.Dropout(hidden_dropout_prob_2)
        self.first_layer = nn.Linear(input_size, hidden_size_1)
        self.second_layer = nn.Linear(hidden_size_1, hidden_size_2)
        self.classifier = nn.Linear(hidden_size_2, 4)

    def forward(self, x):
        x = self.first_layer(x)
        x = nn.ReLU()(x)
        x = self.dropout_1(x)
        x = self.second_layer(x)
        x = nn.ReLU()(x)
        x = self.dropout_2(x)
        x = self.classifier(x)
        return x

    def __str__(self):
        return f"Input size: {self.input_size} \nHidden size 1: {self.hidden_size_1} \nHidden size 2: {self.hidden_size_2} \nDropout 1: {self.hidden_dropout_prob_1} \nDropout 2: {self.hidden_dropout_prob_2} \nOutput Size: 4 \n"

    def get_params_string(self):
        return f"multi_output_{self.input_size}_{self.hidden_size_1}_{self.hidden_size_2}_{self.hidden_dropout_prob_1}_{self.hidden_dropout_prob_2}"


class FFNNDual(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, hidden_dropout_prob_1, hidden_dropout_prob_2):
        super(FFNNDual, self).__init__()

        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.hidden_dropout_prob_1 = hidden_dropout_prob_1
        self.hidden_dropout_prob_2 = hidden_dropout_prob_2

        self.dropout_1 = nn.Dropout(hidden_dropout_prob_1)
        self.dropout_2 = nn.Dropout(hidden_dropout_prob_2)
        self.first_layer = nn.Linear(input_size, hidden_size_1)
        self.second_layer = nn.Linear(hidden_size_1, hidden_size_2)
        self.classifier = nn.Linear(hidden_size_2, 2)

    def forward(self, x):
        x = self.first_layer(x)
        x = nn.ReLU()(x)
        x = self.dropout_1(x)
        x = self.second_layer(x)
        x = nn.ReLU()(x)
        x = self.dropout_2(x)
        x = self.classifier(x)
        return x

    def __str__(self):
        return f"Input size: {self.input_size} \nHidden size 1: {self.hidden_size_1} \nHidden size 2: {self.hidden_size_2} \nDropout 1: {self.hidden_dropout_prob_1} \nDropout 2: {self.hidden_dropout_prob_2} \nOutput Size: 2 \n"

    def get_params_string(self):
        return f"dual_output_{self.input_size}_{self.hidden_size_1}_{self.hidden_size_2}_{self.hidden_dropout_prob_1}_{self.hidden_dropout_prob_2}"


class FFNN2(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, hidden_dropout_prob_1, hidden_dropout_prob_2):
        super(FFNN2, self).__init__()

        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.hidden_dropout_prob_1 = hidden_dropout_prob_1
        self.hidden_dropout_prob_2 = hidden_dropout_prob_2

        self.dropout_1 = nn.Dropout(hidden_dropout_prob_1)
        self.dropout_2 = nn.Dropout(hidden_dropout_prob_2)
        self.first_layer = nn.Linear(input_size, hidden_size_1)
        self.second_layer = nn.Linear(hidden_size_1, hidden_size_2)
        self.classifier = nn.Linear(hidden_size_2, 1)

    def forward(self, x):
        x = self.first_layer(x)
        x = nn.ReLU()(x)
        x = self.dropout_1(x)
        x = self.second_layer(x)
        x = nn.ReLU()(x)
        x = self.dropout_2(x)
        x = self.classifier(x)
        return x

    def __str__(self):
        return f"Input size: {self.input_size} \nHidden size 1: {self.hidden_size_1} \nHidden size 2: {self.hidden_size_2} \nDropout 1: {self.hidden_dropout_prob_1} \nDropout 2: {self.hidden_dropout_prob_2} \nOutput Size: 1 \n"
    
    def get_params_string(self):
        return f"{self.input_size}_{self.hidden_size_1}_{self.hidden_size_2}_{self.hidden_dropout_prob_1}_{self.hidden_dropout_prob_2}"


class FFNN1(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_dropout_prob):
        super(FFNN1, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_dropout_prob = hidden_dropout_prob

        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.first_layer = nn.Linear(input_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.first_layer(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

    def __str__(self):
        return f"Input size: {self.input_size} \nHidden size: {self.hidden_size} \nDropout: {self.hidden_dropout_prob} \nOutput Size: 1 \n"    

    def get_params_string(self):
        return f"{self.input_size}_{self.hidden_size}_{self.hidden_dropout_prob_1}"



class DistilBertMultiClassifier(nn.Module):

    def __init__(self, ffnn_params):
        super().__init__()

        self.bert = DistilBertModel.from_pretrained("distilbert-base-multilingual-cased")
        self.ffnn = FFNNMulti(**ffnn_params)

    def forward(
            self,
            input_ids=None,
            input_features=None,  # the second input
            attention_mask=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None, ):

        distilbert_output = self.bert(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = torch.cat([pooled_output, input_features.float()], dim=1)

        logits = self.ffnn(pooled_output)  # (bs, dim)

        preds = torch.sigmoid(logits)

        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1), labels.view(-1).float())
            # Declaring the class containing the metrics

            preds_arr = preds.detach().cpu().numpy()
            labels_arr = labels.detach().cpu().numpy()
            output_list = []
            for i in range(4):
                #print(preds_arr[:, i])
                #print(labels_arr[:, i])

                outputs = (loss, logits[:, i], preds_arr[:, i],)
                output_list.append(outputs)

            return output_list  # (loss), logits, (hidden_states), (attentions), (preds, prauc, rce, conf, max_pred, min_pred, avg)
        else:
            return (logits,)

    def __str__(self):
        return str(self.ffnn)

    def get_params_string(self):
        return self.ffnn.get_params_string()


class DistilBertDualClassifier(nn.Module):

    def __init__(self, ffnn_params):
        super().__init__()

        self.bert = DistilBertModel.from_pretrained("distilbert-base-multilingual-cased")
        self.ffnn = FFNNDual(**ffnn_params)

    def forward(
            self,
            input_ids=None,
            input_features=None,  # the second input
            attention_mask=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None, ):

        distilbert_output = self.bert(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = torch.cat([pooled_output, input_features.float()], dim=1)

        logits = self.ffnn(pooled_output)  # (bs, dim)

        preds = torch.sigmoid(logits)

        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1), labels.view(-1).float())
            # Declaring the class containing the metrics

            preds_arr = preds.detach().cpu().numpy()
            labels_arr = labels.detach().cpu().numpy()
            output_list = []
            for i in range(2):
                #print(preds_arr[:, i])
                #print(labels_arr[:, i])

                outputs = (loss, logits[:, i], preds_arr[:, i],)
                output_list.append(outputs)

            return output_list  # (loss), logits, (hidden_states), (attentions), (preds, prauc, rce, conf, max_pred, min_pred, avg)
        else:
            return (logits,)

    def __str__(self):
        return str(self.ffnn)

    def get_params_string(self):
        return self.ffnn.get_params_string()


class DistilBertClassifierDoubleInput(nn.Module):

    def __init__(self, ffnn_params):
        super().__init__()

        self.bert = DistilBertModel.from_pretrained("distilbert-base-multilingual-cased")
        self.ffnn = FFNN2(**ffnn_params)

    def forward(
            self,
            input_ids=None,
            input_features=None,  # the second input
            attention_mask=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None, ):

        distilbert_output = self.bert(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = torch.cat([pooled_output, input_features.float()], dim=1)

        logits = self.ffnn(pooled_output)  # (bs, dim)

        preds = torch.sigmoid(logits)

        outputs = (logits,) + distilbert_output[1:]

        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1), labels.view(-1).float())
            # Declaring the class containing the metrics
            #cm = CoMe(preds.detach().cpu().numpy(), labels.detach().cpu().numpy())
            # Evaluating
            #prauc = cm.compute_prauc()
            #rce = cm.compute_rce()
            # Confusion matrix
            #conf = cm.confMatrix()
            # Prediction stats
            #max_pred, min_pred, avg = cm.computeStatistics()

            outputs = (loss,) + outputs + (preds,) #, prauc, rce, conf, max_pred, min_pred, avg)

        return outputs  # (loss), logits, (hidden_states), (attentions), (preds, prauc, rce, conf, max_pred, min_pred, avg)

    def __str__(self):
        return str(self.ffnn)

    def get_params_string(self):
        return self.ffnn.get_params_string()
