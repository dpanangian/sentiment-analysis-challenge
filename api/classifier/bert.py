import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AdamW, BertModel, BertTokenizer, get_linear_schedule_with_warmup

from api.classifier.preprocessing import bert_preprocessing

from .model import Model

with open("config.json") as json_file:
    config = json.load(json_file)


class BERT(Model):
    def __init__(self, pre_trained=True):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.tokenizer = BertTokenizer.from_pretrained(config["PARAMS"]["BERT_MODEL"])


        classifier = BertClassifier(freeze_bert=False)

        classifier.load_state_dict(
            state_dict=torch.load(config["PARAMS"]["PRE_TRAINED_MODEL"], map_location=self.device)
        )
        classifier = classifier.eval()
        self.classifier = classifier.to(self.device)

    def predict_on_batch(self, texts):
        return super().predict_on_batch(texts)

    def predict(self, text):
        return super().predict(text)

    def predict_probabilities(self, text):
        encoded_text = self.tokenizer.encode_plus(
            bert_preprocessing(text),
            max_length=config["PARAMS"]["MAX_SEQUENCE_LEN"],
            add_special_tokens=True,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = encoded_text["input_ids"].to(self.device)
        attention_mask = encoded_text["attention_mask"].to(self.device)

        with torch.no_grad():
            probabilities = F.softmax(self.classifier(input_ids, attention_mask), dim=1)
        return probabilities



    def init_for_train(self, train_dataloader, epochs=4):
        """Initialize the Bert Classifier, the optimizer and the learning rate scheduler.
        """
        # Instantiate Bert Classifier
        bert_classifier = BertClassifier(freeze_bert=False)

        # Tell PyTorch to run the model on GPU
        bert_classifier.to(self.device)

        # Create the optimizer
        optimizer = AdamW(bert_classifier.parameters(),
                        lr=5e-5,    # Default learning rate
                        eps=1e-8    # Default epsilon value
                        )

        # Total number of training steps
        total_steps = len(train_dataloader) * epochs

        # Set up the learning rate scheduler
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0, # Default value
                                                    num_training_steps=total_steps)
        return bert_classifier, optimizer, scheduler





# Create the BertClassfier class
class BertClassifier(nn.Module):
    """Bert Model for Classification Tasks.
    """
    def __init__(self, freeze_bert=False):
        """
        @param    bert: a BertModel object
        @param    classifier: a torch.nn.Module classifier
        @param    freeze_bert (bool): Set `False` to fine-tune the BERT model
        """
        super(BertClassifier, self).__init__()
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        D_in, H, D_out = 768, 50, len(config["CLASS_NAMES"])

        # Instantiate BERT model
        self.bert = BertModel.from_pretrained(config["PARAMS"]["BERT_MODEL"])

        # Instantiate an one-layer feed-forward classifier
        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            #nn.Dropout(0.5),
            nn.Linear(H, D_out)
        )

        # Freeze the BERT model
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
    def forward(self, input_ids, attention_mask):
        """
        Feed input to BERT and the classifier to compute logits.
        @param    input_ids (torch.Tensor): an input tensor with shape (batch_size,
                      max_length)
        @param    attention_mask (torch.Tensor): a tensor that hold attention mask
                      information with shape (batch_size, max_length)
        @return   logits (torch.Tensor): an output tensor with shape (batch_size,
                      num_labels)
        """
        # Feed input to BERT
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)
        
        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = outputs[0][:, 0, :]

        # Feed input to classifier to compute logits
        logits = self.classifier(last_hidden_state_cls)

        return logits
