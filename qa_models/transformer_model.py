# qa_models/transformer_model.py

from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
from .base_model import QAModel

class TransformerModel(QAModel):
    """QA Model using Hugging Face Transformers."""

    def __init__(self, model_name='bert-large-uncased-whole-word-masking-finetuned-squad'):
        """
        Initialize the transformer model.

        Args:
            model_name (str): The Hugging Face model to use.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    def get_answer(self, question, context):
        """
        Get an answer using the transformer model.

        Args:
            question (str): The question to answer.
            context (str): The context to use for answering.

        Returns:
            str: The answer to the question.
        """
        inputs = self.tokenizer.encode_plus(
            question,
            context,
            add_special_tokens=True,
            return_tensors='pt',
            truncation=True,
            max_length=512
        )
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

        # Get the most likely start and end of answer
        answer_start = torch.argmax(start_logits, dim=1).item()
        answer_end = torch.argmax(end_logits, dim=1).item() + 1

        # Convert token ids to tokens
        all_tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])

        # Extract the answer tokens and skip special tokens
        answer_tokens = all_tokens[answer_start:answer_end]
        answer = self.tokenizer.convert_tokens_to_string(answer_tokens).strip()

        if not answer:
            answer = "Unable to find an answer."

        return answer
