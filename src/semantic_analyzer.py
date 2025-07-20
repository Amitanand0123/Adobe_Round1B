import logging
import spacy
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict

logger = logging.getLogger(__name__)

class SemanticAnalyzer:
    """Handles text embedding and persona analysis."""

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        logger.info(f"Loading sentence transformer model: {model_name}")
        self.model = SentenceTransformer(model_name)
        logger.info("Loading spaCy model for keyword extraction...")
        # MODIFIED: Handle potential download error for spacy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("Spacy model 'en_core_web_sm' not found. Downloading...")
            spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
        logger.info("SemanticAnalyzer initialized.")

    # MODIFIED: Make this a public method by removing the underscore
    def extract_persona_keywords(self, text: str) -> List[str]:
        """Extracts key nouns, proper nouns, and verbs from text."""
        doc = self.nlp(text.lower())
        keywords = [
            token.lemma_ for token in doc 
            if not token.is_stop and not token.is_punct and token.pos_ in ["NOUN", "PROPN", "VERB"]
        ]
        return list(set(keywords))

    # REMOVED: This function is no longer needed with the multi-query approach
    # def create_persona_vector(...)

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generates embeddings for a list of text chunks."""
        if not texts:
            return np.array([])
        return self.model.encode(texts, show_progress_bar=False)