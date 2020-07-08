import pathlib
import torch
from transformers import AdamW
# from torchviz import make_dot
import numpy as np
from torch.utils.data import DataLoader, SequentialSampler
from transformers import get_linear_schedule_with_warmup
import random
import time
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import PowerTransformer
from Utils.Eval.Metrics import ComputeMetrics as CoMe
from Utils.NN.TorchModels import DistilBertDualClassifier
from Utils.NN.NNUtils import HIDDEN_SIZE_BERT
from Utils.Base.RecommenderBase import RecommenderBase
from Utils.NN.CustomDatasets import *
from Utils.NN.NNUtils import flat_accuracy, create_data_loaders, format_time
from abc import ABC, abstractmethod

from Utils.TelegramBot import telegram_bot_send_update


# abstract class for nn recommenders
class DualNNRec(RecommenderBase, ABC):

    # TODO add support for Early Stopping
    def __init__(self, weight_decay: float = 0.0,
                 lr: float = 2e-5,
                 cap_length: int = 128,
                 eps: float = 1e-8,
                 num_warmup_steps: int = 0,
                 epochs: int = 4,
                 ffnn_params: dict = None,):
        super().__init__()
        self.device = None
        self.device = self._find_device()
        self.weight_decay = weight_decay
        self.lr = lr
        self.eps = eps
        self.num_warmup_steps = num_warmup_steps
        self.epochs = epochs

        self.cap_length = cap_length

        self.scaler = PowerTransformer(method='yeo-johnson')

        self.ffnn_params = ffnn_params

        self.model = None

        # Set the seed value all over the place to make this reproducible.
        self.seed_val = 42
        random.seed(self.seed_val)
        np.random.seed(self.seed_val)
        torch.manual_seed(self.seed_val)

    @abstractmethod
    def _get_model(self, ffnn_input_size):
        pass

    def _find_device(self):
        # If there's a GPU available...
        if torch.cuda.is_available():

            # Tell PyTorch to use the GPU.
            device = torch.device("cuda")

            print('There are %d GPU(s) available.' % torch.cuda.device_count())

            print('We will use the GPU:', torch.cuda.get_device_name(0))

        # If not...
        else:
            print('No GPU available, using the CPU instead.')
            device = torch.device("cpu")

        return device

    def _normalize_features(self, df, is_train=False):
        if is_train == True:
            print("Fitting yeo-jhonson scaler")
            self.scaler.fit(df)
            # print(self.scaler.scale_, self.scaler.mean_, self.scaler.var_, self.scaler.n_samples_seen_)
        return pd.DataFrame(self.scaler.transform(df), columns=df.columns)

    def load_model(self):
        pass

    # TODO add support for cat features
    def fit(self,
            df_train_features: pd.DataFrame,
            df_train_tokens_reader: pd.io.parsers.TextFileReader,
            df_train_label: pd.DataFrame,
            df_val_features: pd.DataFrame,
            df_val_tokens_reader: pd.io.parsers.TextFileReader,
            df_val_label: pd.DataFrame,
            save_filename : str,
            cat_feature_set: set,
            normalize: bool = True,
            train_batches_to_skip: int = 0,
            val_batches_to_skip: int = 0,
            pretrained_model_dict_path : str = None,
            pretrained_optimizer_dict_path : str = None):

        self.df_train_label = df_train_label
        self.df_val_label = df_val_label

        print(df_train_features)
        print(df_val_features)

        assert len(df_train_label.columns) == 2, "it needs 2 labels in train df."

        assert len(df_val_label.columns) == 2, "it needs 2 labels in val df."

        assert len(df_train_features.columns) == len(df_val_features.columns), \
            "df_train_features and df_val_features have different number of columns"

        if normalize:
            df_train_features = self._normalize_features(df_train_features, is_train=True)
            df_val_features = self._normalize_features(df_val_features)
            print(df_train_features)
            print(df_val_features)

        gpu = torch.cuda.is_available()
        if gpu:
            torch.cuda.manual_seed_all(self.seed_val)

        ffnn_input_size = HIDDEN_SIZE_BERT + df_train_features.shape[1]

        self.model = self._get_model(ffnn_input_size=ffnn_input_size)

        if pretrained_model_dict_path is not None:
            print(f"Loading pretrained model : {pretrained_model_dict_path}")
            self.model.load_state_dict(torch.load(pretrained_model_dict_path))

        if gpu:
            self.model.cuda()

        # freeze all bert layers
        # for param in self.model.bert.parameters():
        #     param.requires_grad = False
        train_dataset = CustomDatasetDualCap(df_features=df_train_features, df_tokens_reader=df_train_tokens_reader,
                                        df_label=df_train_label, cap=self.cap_length, batches_to_skip=train_batches_to_skip)
        val_dataset = CustomDatasetDualCap(df_features=df_val_features, df_tokens_reader=df_val_tokens_reader,
                                    df_label=df_val_label, cap=self.cap_length, batches_to_skip=val_batches_to_skip)

        train_dataloader, validation_dataloader = create_data_loaders(train_dataset, val_dataset,
                                                                        batch_size=df_train_tokens_reader.chunksize)

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]

        # Note: AdamW is a class from the huggingface library (as opposed to pytorch)
        # I believe the 'W' stands for 'Weight Decay fix"
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.lr,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                          eps=self.eps  # args.adam_epsilon  - default is 1e-8.
                          )

        if pretrained_optimizer_dict_path is not None:
            print(f"Loading pretrained optimizer : {pretrained_optimizer_dict_path}")
            optimizer.load_state_dict(torch.load(pretrained_optimizer_dict_path))

        # Total number of training steps is [number of batches] x [number of epochs].
        # (Note that this is not the same as the number of training samples).
        total_steps = len(train_dataloader) * self.epochs

        # Create the learning rate scheduler.
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0,  # Default value in run_glue.py
                                                    num_training_steps=total_steps)

        # We'll store a number of quantities such as training and validation loss,
        # validation accuracy, and timings.
        training_stats = []

        # Measure the total training time for the whole run.
        total_t0 = time.time()

        # For each epoch...
        for epoch_i in range(0, self.epochs):
            # ========================================
            #               Training
            # ========================================

            # Perform one full pass over the training set.

            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, self.epochs))
            print('Training...')
            avg_train_loss, training_time, = self.train(self.model, train_dataloader, optimizer,
                                                                               scheduler)

            # ========================================
            #               Validation
            # ========================================
            # After the completion of each training epoch, measure our performance on
            # our validation set.

            print("")
            print("Running Validation...")

            avg_val_loss, validation_time = self.validation(model=self.model,
                                                            validation_dataloader=validation_dataloader)

            # Record all statistics from this epoch.
            curr_stats = {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                # 'PRAUC train': prauc_train,
                # 'RCE train': rce_train,
                # 'PRAUC val': prauc_val,
                # 'RCE val': rce_val,
                'Valid. Loss': avg_val_loss,
                # 'Valid. Accur.': avg_val_accuracy,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
            training_stats.append(curr_stats)

            pathlib.Path('./saved_models').mkdir(parents=True, exist_ok=True)

            model_path = f"./saved_models/saved_model_{save_filename}"
            optimizer_path = f"./saved_models/saved_optimizer_{save_filename}"

            print(f"Saving model : {model_path}")

            torch.save(self.model.state_dict(), model_path)
            torch.save(optimizer.state_dict(), optimizer_path)

            bot_string = f"DistilBertDoubleInput NN - dual_label \n ---------------- \n"
            bot_string = bot_string + str(self.model)
            bot_string = bot_string + "Weight decay: " + str(self.weight_decay) + "\n"
            bot_string = bot_string + "Learning rate: " + str(self.lr) + "\n"
            bot_string = bot_string + "Epsilon: " + str(self.eps) + "\n ---------------- \n"
            bot_string = bot_string + "\n".join([key + ": " + str(curr_stats[key]) for key in curr_stats]) + "\n\n"
            bot_string = bot_string + "Saved to : " + model_path
            #telegram_bot_send_update(bot_string)

        print("")
        print("Training complete!")

        print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))

        return training_stats

    def train(self, model, train_dataloader, optimizer, scheduler):

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_train_loss = 0
        # total_train_prauc = 0
        # total_train_rce = 0

        # Put the model into training mode. Don't be mislead--the call to
        # `train` just changes the *mode*, it doesn't *perform* the training.
        # `dropout` and `batchnorm` layers behave differently during training
        # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
        model.train()
        preds_list = [None] * 2
        labels_list = [None] * 2

        # For each batch of training data...
        for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):

            # Progress update every 40 batches.
            #if step % 40 == 0 and not step == 0:
            #    # Calculate elapsed time in minutes.
            #    elapsed = format_time(time.time() - t0)
            #
            #    # Report progress.
            #    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            # Unpack this training batch from our dataloader.
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using the
            # `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids
            #   [1]: attention masks
            #   [2]: features
            #   [3]: labels
            b_input_ids = batch[0].to(self.device)
            b_input_mask = batch[1].to(self.device)
            b_features = batch[2].to(self.device)
            b_labels = batch[3].to(self.device)

            #print("b_labels")
            #print(b_labels)
            #print(b_labels.shape)

            # print("b_labels:",b_labels.shape)

            # Always clear any previously calculated gradients before performing a
            # backward pass. PyTorch doesn't do this automatically because
            # accumulating the gradients is "convenient while training RNNs".
            # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
            model.zero_grad()

            # Perform a forward pass (evaluate the model on this training batch).
            # The documentation for this `model` function is here:
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            # It returns different numbers of parameters depending on what arguments
            # arge given and what flags are set. For our useage here, it returns
            # the loss (because we provided labels) and the "logits"--the model
            # outputs prior to activation.
            output_list = model(input_ids=b_input_ids,
                                input_features=b_features,
                                # token_type_ids=None,
                                attention_mask=b_input_mask,
                                labels=b_labels)

            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value
            # from the tensor.
            loss = output_list[0][0]
            total_train_loss += loss.item()

            for i in range(2):
                curr_preds = output_list[i][2]

                if preds_list[i] is None:
                    preds_list[i] = curr_preds
                else:
                    preds_list[i] = np.hstack([preds_list[i], curr_preds])

                curr_labels = b_labels.detach().cpu().numpy()[:, i]

                if labels_list[i] is None:
                    labels_list[i] = curr_labels
                else:
                    labels_list[i] = np.hstack([labels_list[i], curr_labels])

            # print(f"batch {step} RCE: {rce}")
            # print(f"batch {step} PRAUC: {prauc}")

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)

        print(f"TRAINING STATISTICS FOR EPOCH")
        for i in range(2):
            prauc, rce, conf, max_pred, min_pred, avg = self.evaluate(preds=preds_list[i], labels=labels_list[i])
            if i == 0:
                print("\n------- LABEL 1 -------")
            elif i == 1:
                print("\n------- LABEL 2 -------")

            print(f"PRAUC : {prauc}"
                  f"\nRCE : {rce}"
                  f"\nMIN : {min_pred}"
                  f"\nMAX : {max_pred}"
                  f"\nAVG : {avg}")

        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(training_time))

        return avg_train_loss, training_time

    def validation(self, model, validation_dataloader):

        t0 = time.time()
        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        model.eval()

        # Tracking variables
        total_eval_loss = 0
        preds_list = [None] * 2
        labels_list = [None] * 2

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Evaluate data for one epoch
        for step, batch in tqdm(enumerate(validation_dataloader), total=len(validation_dataloader)):

            # Progress update every 40 batches.
            #if step % 40 == 0 and not step == 0:
            #    # Calculate elapsed time in minutes.
            #    elapsed = format_time(time.time() - t0)
            #
            #    # Report progress.
            #    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(validation_dataloader), elapsed))

            # Unpack this training batch from our dataloader.
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using
            # the `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids
            #   [1]: attention masks
            #   [2]: features
            #   [3]: labels
            b_input_ids = batch[0].to(self.device)
            b_input_mask = batch[1].to(self.device)
            b_features = batch[2].to(self.device)
            b_labels = batch[3].to(self.device)
            # print("b_labels:",b_labels.shape)

            # Tell pytorch not to bother with constructing the compute graph during
            # the forward pass, since this is only needed for backprop (training).
            with torch.no_grad():
                # Forward pass, calculate logit predictions.
                # token_type_ids is the same as the "segment ids", which
                # differentiates sentence 1 and 2 in 2-sentence tasks.
                # The documentation for this `model` function is here:
                # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                # Get the "logits" output by the model. The "logits" are the output
                # values prior to applying an activation function like the softmax.
                output_list = model(input_ids=b_input_ids,
                                    input_features=b_features,
                                    attention_mask=b_input_mask,
                                    labels=b_labels,
                                    )
            loss = output_list[0][0]
            total_eval_loss += loss.item()

            for i in range(2):
                curr_preds = output_list[i][2]

                if preds_list[i] is None:
                    preds_list[i] = curr_preds
                else:
                    preds_list[i] = np.hstack([preds_list[i], curr_preds])

                curr_labels = b_labels.detach().cpu().numpy()[:, i]

                if labels_list[i] is None:
                    labels_list[i] = curr_labels
                else:
                    labels_list[i] = np.hstack([labels_list[i], curr_labels])


        # Calculate the average loss over all of the batches.
        avg_val_loss = total_eval_loss / len(validation_dataloader)

        print(f"VALIDATION STATISTICS FOR EPOCH")
        for i in range(2):
            prauc, rce, conf, max_pred, min_pred, avg = self.evaluate(preds=preds_list[i], labels=labels_list[i])
            if i == 0:
                print("\n------- LABEL 1 -------")
            elif i == 1:
                print("\n------- LABEL 2 -------")

            print(f"PRAUC : {prauc}"
                  f"\nRCE : {rce}"
                  f"\nMIN : {min_pred}"
                  f"\nMAX : {max_pred}"
                  f"\nAVG : {avg}")

        # Measure how long the validation run took.
        validation_time = format_time(time.time() - t0)

        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))

        return avg_val_loss, validation_time

    def evaluate(self, preds, labels=None):

        # print(preds)
        # print(preds.shape)
        # print(labels)
        # print(labels.shape)

        # Tries to load X and Y if not directly passed
        if (labels is None):
            print("No labels passed, cannot perform evaluation.")

        if (self.model is None):
            print("No model trained, cannot to perform evaluation.")

        else:
            #print("preds")
            #print(preds)
            #print(preds.shape)
            #print("labels")
            #print(labels)
            #print(labels.shape)

            # Declaring the class containing the metrics
            cm = CoMe(preds, labels)

            # Evaluating
            prauc = cm.compute_prauc()
            rce = cm.compute_rce()
            # Confusion matrix
            conf = cm.confMatrix()
            # Prediction stats
            max_pred, min_pred, avg = cm.computeStatistics()

            return prauc, rce, conf, max_pred, min_pred, avg

    def get_prediction(self,
                       df_test_features: pd.DataFrame,
                       df_test_tokens_reader: pd.io.parsers.TextFileReader,
                       pretrained_model_dict_path: str = None,
                       normalize: bool = True
                       ):

        if normalize:
            df_test_features = self._normalize_features(df_test_features)

        if pretrained_model_dict_path is None:
            assert self.model is not None, "You are trying to predict without training."
        else:
            ffnn_input_size = HIDDEN_SIZE_BERT + df_test_features.shape[1]
            self.model = self._get_model(ffnn_input_size=ffnn_input_size)
            self.model.load_state_dict(torch.load(pretrained_model_dict_path))

        self.model.cuda()
        self.model.eval()

        preds = None

        test_dataset = CustomTestDatasetCap(df_features=df_test_features, df_tokens_reader=df_test_tokens_reader,
                                            cap=self.cap_length)
        test_dataloader = DataLoader(test_dataset,  # The test samples.
                                     sampler=SequentialSampler(test_dataset),  # Select batches sequentially
                                     batch_size=df_test_tokens_reader.chunksize
                                     # Generates predictions with this batch size.
                                     )

        # Evaluate data for one epoch
        for step, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
            # Unpack this training batch from our dataloader.
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using
            # the `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids
            #   [1]: attention masks
            #   [2]: features
            #   [3]: labels
            b_input_ids = batch[0].to(self.device)
            b_input_mask = batch[1].to(self.device)
            b_features = batch[2].to(self.device)

            # Tell pytorch not to bother with constructing the compute graph during
            # the forward pass, since this is only needed for backprop (training).
            with torch.no_grad():
                # Forward pass, calculate logit predictions.
                # token_type_ids is the same as the "segment ids", which
                # differentiates sentence 1 and 2 in 2-sentence tasks.
                # The documentation for this `model` function is here:
                # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                # Get the "logits" output by the model. The "logits" are the output
                # values prior to applying an activation function like the softmax.
                curr_logits = self.model(input_ids=b_input_ids,
                                         input_features=b_features,
                                         # token_type_ids=None, --> missing in distilbert
                                         attention_mask=b_input_mask)

            curr_logits = curr_logits[0]

            #print(curr_logits)
            #print(curr_logits.shape)

            curr_preds = torch.sigmoid(curr_logits)

            curr_preds = curr_preds.detach().cpu().numpy()

            if preds is None:
                preds = curr_preds
            else:
                preds = np.vstack([preds, curr_preds])

        return preds


class DualDistilBertRec(DualNNRec):

    def _get_model(self, ffnn_input_size):
        self.ffnn_params['input_size'] = ffnn_input_size
        return DistilBertDualClassifier(self.ffnn_params)
