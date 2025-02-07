from abc import ABC, abstractmethod

class QAModel(ABC):
    """Abstract base class for question-answering models."""

    @abstractmethod
    def get_answer(self, question, context):
        """Get an answer to the question based on the context."""
        pass
