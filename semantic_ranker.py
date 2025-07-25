# semantic_ranker.py

import logging
import numpy as np
from typing import List, Dict
# NEW: Import TfidfVectorizer and cosine_similarity from scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class SemanticRanker:
    """
    Enhanced semantic ranking system using TF-IDF with improved scoring.
    This version is lightweight and has a minimal footprint.
    """

    def __init__(self):
        """Initialize the semantic ranker with a TF-IDF Vectorizer."""
        logger.info("Initializing TF-IDF based Semantic Ranker")
        # Initialize the vectorizer. We can tune parameters here.
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2), # Consider both single words and two-word phrases
            max_df=0.85,        # Ignore terms that appear in more than 85% of documents
            min_df=2            # Ignore terms that appear in less than 2 documents
        )
        logger.info("TF-IDF Semantic Ranker initialized successfully")

    # All your private methods (_extract_domain_keywords, _calculate_persona_alignment, etc.)
    # remain EXACTLY THE SAME. No changes needed there.
    def _extract_domain_keywords(self, persona_text: str, task_text: str) -> List[str]:
        """Extract domain-specific keywords based on persona and task"""
        keywords = {
            'academic': ['methodology', 'research', 'study', 'analysis', 'findings', 'results',
                        'literature', 'review', 'data', 'experiment', 'hypothesis', 'conclusion'],
            'business': ['revenue', 'profit', 'growth', 'market', 'financial', 'strategy',
                        'performance', 'trends', 'analysis', 'investment', 'ROI', 'metrics'],
            'student': ['concept', 'definition', 'example', 'theory', 'principle', 'study',
                       'exam', 'practice', 'key points', 'understanding', 'learning'],
            'technical': ['implementation', 'algorithm', 'approach', 'method', 'technique',
                         'framework', 'model', 'system', 'process', 'architecture'],
            'medical': ['treatment', 'diagnosis', 'symptoms', 'therapy', 'patient', 'clinical',
                       'disease', 'medicine', 'healthcare', 'medical'],
            'legal': ['law', 'regulation', 'compliance', 'legal', 'court', 'case', 'statute',
                     'contract', 'agreement', 'policy']
        }
        combined_text = (persona_text + " " + task_text).lower()
        relevant_keywords = []
        if any(word in combined_text for word in ['researcher', 'phd', 'academic', 'scientist']):
            relevant_keywords.extend(keywords['academic'])
        if any(word in combined_text for word in ['analyst', 'business', 'investment', 'financial']):
            relevant_keywords.extend(keywords['business'])
        if any(word in combined_text for word in ['student', 'undergraduate', 'learning', 'exam']):
            relevant_keywords.extend(keywords['student'])
        if any(word in combined_text for word in ['engineer', 'developer', 'technical', 'algorithm']):
            relevant_keywords.extend(keywords['technical'])
        if any(word in combined_text for word in ['doctor', 'medical', 'clinical', 'patient']):
            relevant_keywords.extend(keywords['medical'])
        if any(word in combined_text for word in ['lawyer', 'legal', 'law', 'compliance']):
            relevant_keywords.extend(keywords['legal'])
        return relevant_keywords

    def _calculate_persona_alignment(self, chunk: Dict, persona: str, task: str) -> float:
        """Calculate how well a chunk aligns with the specific persona and task"""
        text_lower = chunk.get('text', '').lower()
        section_title_lower = chunk.get('section_title', '').lower()
        score = 0.0
        if any(word in persona.lower() for word in ['researcher', 'phd', 'academic']):
            academic_indicators = ['methodology', 'research', 'study', 'analysis', 'findings', 'results', 'literature', 'data', 'experiment']
            score += sum(0.1 for indicator in academic_indicators if indicator in text_lower)
            if any(word in section_title_lower for word in ['method', 'result', 'conclusion', 'discussion']):
                score += 0.3
        if any(word in persona.lower() for word in ['analyst', 'business', 'investment']):
            business_indicators = ['revenue', 'profit', 'growth', 'market', 'financial', 'strategy', 'performance', 'trends']
            score += sum(0.1 for indicator in business_indicators if indicator in text_lower)
            if any(word in section_title_lower for word in ['financial', 'revenue', 'performance', 'analysis']):
                score += 0.3
        if any(word in persona.lower() for word in ['student', 'undergraduate']):
            student_indicators = ['concept', 'definition', 'example', 'theory', 'principle', 'key', 'important', 'summary']
            score += sum(0.1 for indicator in student_indicators if indicator in text_lower)
            if any(word in section_title_lower for word in ['introduction', 'summary', 'overview', 'key']):
                score += 0.3
        task_words = [word for word in task.lower().split() if len(word) > 3]
        title_matches = sum(1 for word in task_words if word in section_title_lower)
        text_matches = sum(1 for word in task_words if word in text_lower)
        score += title_matches * 0.15
        score += min(text_matches * 0.05, 0.2)
        return min(score, 1.0)

    def _calculate_content_specificity(self, chunk: Dict) -> float:
        """Calculate how specific and actionable the content is"""
        text = chunk.get('text', '')
        specificity_indicators = [(':', 0.1), ('-', 0.05), ('•', 0.05), ('1.', 0.1), ('2.', 0.05), ('Table', 0.15), ('Figure', 0.15), ('Algorithm', 0.2), ('Equation', 0.15)]
        specificity_score = 0.0
        for indicator, weight in specificity_indicators:
            count = text.count(indicator)
            specificity_score += min(count * weight, weight * 3)
        import re
        numbers = re.findall(r'\d+(?:\.\d+)?%?', text)
        if len(numbers) > 3:
            specificity_score += 0.2
        citations = re.findall(r'\[\d+\]|\(\d+\)', text)
        if len(citations) > 2:
            specificity_score += 0.15
        return min(specificity_score, 1.0)

    def _calculate_section_importance(self, chunk: Dict) -> float:
        """Calculate importance based on section title and position"""
        section_title = chunk.get('section_title', '').lower()
        important_sections = {'high': ['result', 'conclusion', 'finding', 'summary', 'discussion', 'analysis'], 'medium': ['method', 'approach', 'implementation', 'experiment', 'evaluation'], 'low': ['introduction', 'background', 'related work', 'acknowledgment', 'reference']}
        if any(keyword in section_title for keyword in important_sections['high']):
            return 0.3
        elif any(keyword in section_title for keyword in important_sections['medium']):
            return 0.2
        elif any(keyword in section_title for keyword in important_sections['low']):
            return 0.1
        return 0.15

    def rank_chunks(self, chunks: List[Dict], queries: List[str], keywords: List[str],
                   persona: str = "", task: str = "") -> List[Dict]:
        """
        Ranks chunks using TF-IDF for semantic relevance, combined with persona-specific scoring.
        """
        if not chunks:
            logger.warning("No chunks provided for ranking")
            return []
        if not queries:
            logger.warning("No queries provided for ranking")
            return chunks

        logger.info(f"Ranking {len(chunks)} chunks against {len(queries)} queries using TF-IDF")

        domain_keywords = self._extract_domain_keywords(persona, task)
        all_keywords = keywords + domain_keywords
        chunk_texts = [chunk.get('text', '') for chunk in chunks]

        # --- TF-IDF Calculation ---
        # Combine queries and chunk texts to build a shared vocabulary
        all_texts = queries + chunk_texts
        tfidf_matrix = self.vectorizer.fit_transform(all_texts)

        # Split the matrix back into query vectors and chunk vectors
        query_vectors = tfidf_matrix[:len(queries)]
        chunk_vectors = tfidf_matrix[len(queries):]

        # Calculate cosine similarity
        similarity_matrix = cosine_similarity(query_vectors, chunk_vectors)
        max_similarity_scores = np.max(similarity_matrix, axis=0)
        avg_similarity_scores = np.mean(similarity_matrix, axis=0)
        # --- End of TF-IDF Calculation ---

        scored_chunks = []
        for i, chunk in enumerate(chunks):
            # The rest of your brilliant scoring logic remains the same!
            # We are just replacing the source of the semantic scores.
            semantic_score = max_similarity_scores[i]
            avg_semantic_score = avg_similarity_scores[i]

            text_lower = chunk.get('text', '').lower()
            keyword_matches = sum(1 for keyword in all_keywords if keyword.lower() in text_lower)
            keyword_score = min(0.4, keyword_matches * 0.03)

            section_title = chunk.get('section_title', '').lower()
            title_keyword_matches = sum(1 for keyword in all_keywords if keyword.lower() in section_title)
            title_score = min(0.3, title_keyword_matches * 0.1)

            persona_score = self._calculate_persona_alignment(chunk, persona, task)
            specificity_score = self._calculate_content_specificity(chunk)
            importance_score = self._calculate_section_importance(chunk)
            density_score = chunk.get('content_density', 0.5) * 0.15
            word_count = len(chunk.get('text', '').split())
            length_bonus = min(0.15, word_count / 800)

            # Same final score calculation
            final_score = (
                semantic_score * 0.22 +
                avg_semantic_score * 0.08 +
                keyword_score * 0.18 +
                title_score * 0.15 +
                persona_score * 0.15 +
                specificity_score * 0.08 +
                importance_score * 0.05 +
                density_score * 0.02 +
                length_bonus * 1.5
            )

            chunk_copy = chunk.copy()
            chunk_copy['relevance_score'] = final_score
            chunk_copy['semantic_score'] = semantic_score
            chunk_copy['keyword_matches'] = keyword_matches
            chunk_copy['persona_alignment'] = persona_score
            chunk_copy['specificity_score'] = specificity_score
            chunk_copy['importance_score'] = importance_score
            scored_chunks.append(chunk_copy)

        ranked_chunks = sorted(scored_chunks, key=lambda x: x['relevance_score'], reverse=True)
        
        logger.info(f"Ranking complete. Top score: {ranked_chunks[0]['relevance_score']:.3f}")
        logger.info(f"Average score: {np.mean([c['relevance_score'] for c in ranked_chunks]):.3f}")

        return ranked_chunks