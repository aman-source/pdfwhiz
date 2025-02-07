# qa_models/openai_model.py

from openai import OpenAI
from .base_model import QAModel
import os

class OpenAIModel(QAModel):
    """QA Model using OpenAI's GPT API."""

    def __init__(self, api_key=None, model_name='gpt-3.5-turbo'):
        """
        Initialize the OpenAI model.

        Args:
            api_key (str, optional): OpenAI API key. Defaults to environment variable.
            model_name (str): The OpenAI model to use.
        """
        self.api_key =  os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key is not set.")
        
        # Initialize the OpenAI client
        self.client = OpenAI(api_key=self.api_key)
        self.model_name = model_name

        # Set instructions for each use case
        self.instructions = '''
            You are a helpful assistant designed to answer questions based on provided PDF content.
            Your answers should be clear and concise, providing information relevant to the context.
        '''

    def get_answer(self, question, context):
        """
        Get an answer from the OpenAI API based on context and question.

        Args:
            question (str): The question to answer.
            context (str): The context to use for answering.

        Returns:
            str: The answer to the question.
        """
        # Create messages for the conversation
        messages = [
            {"role": "system", "content": self.instructions},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
        ]

        try:
            # Generate a response using the OpenAI client
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.2,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error with OpenAI API: {e}"
