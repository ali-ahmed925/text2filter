import torch
from transformers import AutoModel, AutoTokenizer
import pickle

class TwoHeadClassifier(torch.nn.Module):
    def __init__(self, transformer_name, num_filters, num_intensities):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(transformer_name)
        hidden_size = self.transformer.config.hidden_size
        self.filter_head = torch.nn.Linear(hidden_size, num_filters)
        self.intensity_head = torch.nn.Linear(hidden_size, num_intensities)

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        filter_logits = self.filter_head(cls_output)
        intensity_logits = self.intensity_head(cls_output)
        return filter_logits, intensity_logits


transformer_name = 'sentence-transformers/paraphrase-mpnet-base-v2'
model_path = 'twohead_model.pt'
filter_encoder_path = 'encoders/filter_encoder.pkl'
intensity_encoder_path = 'encoders/intensity_encoder.pkl'

with open(filter_encoder_path, 'rb') as f:
    filter_encoder = pickle.load(f)
with open(intensity_encoder_path, 'rb') as f:
    intensity_encoder = pickle.load(f)

num_filters = len(filter_encoder.classes_)
num_intensities = len(intensity_encoder.classes_)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = TwoHeadClassifier(transformer_name, num_filters, num_intensities).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

tokenizer = AutoTokenizer.from_pretrained(transformer_name)


def predict_phrase(phrase):
    encoding = tokenizer.encode_plus(
        phrase,
        add_special_tokens=True,
        max_length=32,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        filter_logits, intensity_logits = model(input_ids, attention_mask)
        pred_filter_idx = filter_logits.argmax(dim=1).item()
        pred_intensity_idx = intensity_logits.argmax(dim=1).item()

    pred_filter = filter_encoder.inverse_transform([pred_filter_idx])[0]
    pred_intensity = intensity_encoder.inverse_transform([pred_intensity_idx])[0]

    return pred_filter, pred_intensity


if __name__ == "__main__":
    test_phrases = [
        "remove small noise but keep edges sharp",
        "apply strong smoothing",
        "sharpen the image slightly",
        "blur the background softly",
        "enhance details in the image",
        "remove heavy noise completely",
        "detect strong edges",
        "apply subtle denoising",
        "sharpen features without overdoing it",
        "smooth skin and reduce minor blemishes"
    ]

    for phrase in test_phrases:
        filt, intensity = predict_phrase(phrase)
        print(f"Phrase: '{phrase}'")
        print(f"Predicted Filter: {filt}, Intensity: {intensity}\n")
