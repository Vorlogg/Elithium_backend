import sys
import numpy as np
import pandas as pd
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, SequentialSampler
import os
import gc


# proxy = 'http://LebedevDV:57dwop@10.0.4.255:3128'
#
# os.environ['http_proxy'] = proxy
# os.environ['HTTP_PROXY'] = proxy
# os.environ['https_proxy'] = proxy
# os.environ['HTTPS_PROXY'] = proxy


class BertPredictor:

    def __init__(self) -> None:
        super().__init__()
        self.__pretrained_model = "bert-base-multilingual-uncased"
        self.__label_dict = {
            'HOT_RAISE': 4,
            'RAISE': 3,
            'HOLD': 2,
            'FALL': 1,
            'HOT_FALL': 0,
        }
        self.__batch_size = 3
        self.__model = None
        self.__device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.__model_path = os.path.join(sys.path[0], "bert.model")

    def __evaluate(self, dataloader_val):
        self.__model.eval()

        predictions = []

        for batch in dataloader_val:
            batch = tuple(b.to(self.__device) for b in batch)

            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'labels': batch[2],
                      }

            with torch.no_grad():
                outputs = self.__model(**inputs)

            logits = outputs[1]

            logits = logits.detach().cpu().numpy()
            predictions.append(logits)

        predictions = np.concatenate(predictions, axis=0)

        return predictions

    def __handle_preds(self, preds):
        label_dict_inverse = {v: k for k, v in self.__label_dict.items()}

        preds_flat = np.argmax(preds, axis=1).flatten()

        result = []

        for pred in preds_flat:
            result.append(label_dict_inverse[pred])

        return result

    def predict(self, news):
        df = pd.DataFrame.from_dict(news, orient='columns')
        df['predicted'] = 2     # заглушка
        tokenizer = BertTokenizer.from_pretrained(self.__pretrained_model, do_lower_case=True)

        encoded_data = tokenizer.batch_encode_plus(
            df.text.values,
            add_special_tokens=True,
            return_attention_mask=True,
            padding="max_length",
            truncation=True,
            max_length=256,
            return_tensors='pt'
        )

        input_ids = encoded_data['input_ids']
        attention_masks = encoded_data['attention_mask']
        plugs = torch.tensor(df.predicted.values)

        dataset = TensorDataset(input_ids, attention_masks, plugs)

        dataloader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=self.__batch_size)

        self.__model = BertForSequenceClassification.from_pretrained(self.__pretrained_model, num_labels=len(self.__label_dict), output_attentions=False, output_hidden_states=False)
        self.__model.to(self.__device)
        self.__model.load_state_dict(torch.load(self.__model_path, map_location=self.__device))

        predictions = self.__evaluate(dataloader)

        # print(predictions)
        # print(self.__handle_preds(predictions))

        df['predicted'] = self.__handle_preds(predictions)
        df = df[['id', 'predicted']]

        result_json = df.to_json(force_ascii=False, orient="records")

        del self.__model
        gc.collect()

        return result_json
