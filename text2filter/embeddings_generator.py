import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn

class TwoHeadClassifier(nn.Module):
    def __init__(self, transformer_name, num_filters, num_intensities):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(transformer_name)
        hidden_size = self.transformer.config.hidden_size
        self.filter_head = nn.Linear(hidden_size, num_filters)
        self.intensity_head = nn.Linear(hidden_size, num_intensities)

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        filter_logits = self.filter_head(cls_output)
        intensity_logits = self.intensity_head(cls_output)
        return filter_logits, intensity_logits

class Predictor:
    def __init__(self, model_path, filter_encoder, intensity_encoder, transformer_name, device=None):
        import pickle
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TwoHeadClassifier(transformer_name, len(filter_encoder.classes_), len(intensity_encoder.classes_)).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(transformer_name)
        self.filter_encoder = filter_encoder
        self.intensity_encoder = intensity_encoder

    def predict(self, phrase):
        encoding = self.tokenizer.encode_plus(
            phrase, add_special_tokens=True, max_length=32, padding="max_length", truncation=True, return_tensors="pt"
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            filter_logits, intensity_logits = self.model(input_ids, attention_mask)
            pred_filter_idx = filter_logits.argmax(dim=1).item()
            pred_intensity_idx = intensity_logits.argmax(dim=1).item()

        pred_filter = self.filter_encoder.inverse_transform([pred_filter_idx])[0]
        pred_intensity = self.intensity_encoder.inverse_transform([pred_intensity_idx])[0]
        return pred_filter, pred_intensity
