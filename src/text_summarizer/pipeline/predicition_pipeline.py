from src.text_summarizer.config.configuration import ConfigurationManager
from transformers import AutoTokenizer
from transformers import pipeline


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


class PredictionPipeline:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load tokenizer from original model
        self.tokenizer = AutoTokenizer.from_pretrained("google/pegasus-cnn_dailymail")

        # Load your fine-tuned model
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            "artifacts/model_trainer/pegasus-samsum-model"
        ).to(self.device)

        self.model.eval()

    def predict(self, text):
        inputs = self.tokenizer(
            text,
            max_length=512,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=60,
                min_length=10,
                num_beams=4,
                length_penalty=1.0,
                no_repeat_ngram_size=3,
                early_stopping=True
            )

            # summary_ids = self.model.generate(
            #     inputs["input_ids"],
            #     attention_mask=inputs["attention_mask"],
            #     max_length=64,
            #     num_beams=2
            # )

        summary = self.tokenizer.decode(
            summary_ids[0],
            skip_special_tokens=True
        )

        return summary


