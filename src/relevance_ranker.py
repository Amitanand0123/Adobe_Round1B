import logging
import numpy as np
from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity
from src.semantic_analyzer import SemanticAnalyzer

logger = logging.getLogger(__name__)

class RelevanceRanker:
    """
    Ranks document chunks using a multi-aspect relevance model.
    """

    def __init__(self, analyzer: SemanticAnalyzer):
        self.analyzer = analyzer

    def rank_content(self, aspect_vectors: np.ndarray, chunks: List[Dict], persona_keywords: List[str]) -> List[Dict]:
        """
        Ranks chunks based on cosine similarity to aspects and boosts with keywords.
        """
        if not chunks:
            return []
        
        # Handle case where no aspect vectors are generated
        if aspect_vectors.size == 0:
            for chunk in chunks:
                chunk['relevance'] = 0.0
            return chunks

        chunk_texts = [chunk['text'] for chunk in chunks]
        chunk_embeddings = self.analyzer.generate_embeddings(chunk_texts)

        if chunk_embeddings.size == 0:
            return []
        
        similarity_matrix = cosine_similarity(aspect_vectors, chunk_embeddings)
        max_similarity_scores = np.max(similarity_matrix, axis=0)
        
        ranked_chunks = []
        for i, chunk in enumerate(chunks):
            base_score = max_similarity_scores[i]
            
            # MODIFIED: Increase the bonus slightly for more impactful keyword matching
            bonus = 0.0
            text_lower = chunk['text'].lower()
            keyword_hits = sum(1 for keyword in persona_keywords if keyword in text_lower)
            
            if keyword_hits > 0:
                # Cap the bonus to prevent it from overpowering the semantic score entirely
                bonus = min(0.2, 0.075 * keyword_hits) 
            
            chunk['relevance'] = base_score + bonus
            ranked_chunks.append(chunk)
                
        return sorted(ranked_chunks, key=lambda x: x['relevance'], reverse=True)