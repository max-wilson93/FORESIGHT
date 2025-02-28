"""
Sentiment Analysis with FinBERT

Purpose:
This module uses FinBERT, a pre-trained BERT model fine-tuned for financial text, to perform
sentiment analysis on financial news, earnings calls, and social media data. It classifies text
into positive, negative, or neutral sentiment.

Role in FORESIGHT:
- Provides sentiment scores for textual data.
- Enhances predictive models by incorporating sentiment analysis.
- Integrates with the broader financial forecasting pipeline.

Key Features:
- FinBERT model for financial sentiment analysis.
- Batch processing for large datasets.
- Integration with the data collection and preprocessing modules.
"""

from transformers import BertTokenizer, BertForSequenceClassification
import torch

class FinBERTSentimentAnalyzer:
    def __init__(self):
        """
        Initialize the FinBERT model and tokenizer.
        """
        self.model_name = "yiyanghkust/finbert-tone"
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = BertForSequenceClassification.from_pretrained(self.model_name)

    def analyze_sentiment(self, text: str) -> str:
        """
        Analyze the sentiment of a given text.

        Args:
            text (str): Text to analyze.

        Returns:
            str: Sentiment label ("Positive", "Negative", or "Neutral").
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        outputs = self.model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        sentiment = torch.argmax(probs, dim=1).item()
        # Map sentiment to labels
        labels = ["Negative", "Neutral", "Positive"]
        return labels[sentiment]

# Example usage
if __name__ == "__main__":
    analyzer = FinBERTSentimentAnalyzer()
    text = "Apple Inc. reported record-breaking earnings this quarter!"
    sentiment = analyzer.analyze_sentiment(text)
    print("Sentiment:", sentiment)