import streamlit as st
from transformers import BertTokenizerFast
import torch
from sklearn.externals import joblib
import numpy as np
import torch.nn as nn

class BERT_Arch(nn.Module):
    def __init__(self, bert):
        super(BERT_Arch, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(768, 256)  # Change the number of hidden units
        self.fc2 = nn.Linear(256, 2)   # Change the number of hidden units
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sent_id, mask):
      cls_hs = self.bert(sent_id, attention_mask=mask)['pooler_output']
      x = self.fc1(cls_hs)
      x = self.relu(x)
      x = self.dropout(x)
      x = self.fc2(x)
      x = self.softmax(x)
      return x


def predict_news(news):
    bert_tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    MAX_LENGHT = 15
    tokens_unseen = bert_tokenizer.batch_encode_plus(
        news,
        max_length=MAX_LENGHT,
        pad_to_max_length=True,
        truncation=True
    )
    unseen_seq = torch.tensor(tokens_unseen['input_ids'])
    unseen_mask = torch.tensor(tokens_unseen['attention_mask'])

    clf = joblib.load("Model/fakenews_model.sav")
    with torch.no_grad():
        preds = clf(unseen_seq, unseen_mask)
        preds = preds.detach().cpu().numpy()

    preds = np.argmax(preds, axis=1)
    return preds

def main():
    st.title(":red[FakeNews]_Detecter_")
    st.subheader("Find out the actual truth behind the news :sunglasses:")
    n = st.text_input("Enter news here 👇", placeholder = "News.....")

    if st.button("Validate"):
        n11 = [n]
        pred = predict_news(n11)
        if pred[0] == 1:
            st.success("News is Not Fake!!")
        else:
            st.warning("News is Fake News!!")


if __name__ == '__main__':
    main()
