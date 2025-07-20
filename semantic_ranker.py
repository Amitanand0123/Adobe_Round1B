# semantic_ranker.py

import logging
import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class SemanticRanker:
    """
    Enhanced semantic ranking system using sentence transformers with improved scoring
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize the semantic ranker with a sentence transformer model"""
        logger.info(f"Loading sentence transformer model: {model_name}")
        self.model = SentenceTransformer(model_name)
        logger.info("Semantic ranker initialized successfully")
    
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
        
        # Identify relevant domains
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
        
        # Academic researcher specific alignment
        if any(word in persona.lower() for word in ['researcher', 'phd', 'academic']):
            academic_indicators = ['methodology', 'research', 'study', 'analysis', 'findings', 
                                 'results', 'literature', 'data', 'experiment']
            score += sum(0.1 for indicator in academic_indicators if indicator in text_lower)
            
            # Bonus for methodology and results sections
            if any(word in section_title_lower for word in ['method', 'result', 'conclusion', 'discussion']):
                score += 0.3
        
        # Business analyst specific alignment
        if any(word in persona.lower() for word in ['analyst', 'business', 'investment']):
            business_indicators = ['revenue', 'profit', 'growth', 'market', 'financial', 
                                 'strategy', 'performance', 'trends']
            score += sum(0.1 for indicator in business_indicators if indicator in text_lower)
            
            # Bonus for financial sections
            if any(word in section_title_lower for word in ['financial', 'revenue', 'performance', 'analysis']):
                score += 0.3
        
        # Student specific alignment
        if any(word in persona.lower() for word in ['student', 'undergraduate']):
            student_indicators = ['concept', 'definition', 'example', 'theory', 'principle',
                                'key', 'important', 'summary']
            score += sum(0.1 for indicator in student_indicators if indicator in text_lower)
            
            # Bonus for introductory and summary sections
            if any(word in section_title_lower for word in ['introduction', 'summary', 'overview', 'key']):
                score += 0.3
        
        # Task-specific alignment
        task_words = [word for word in task.lower().split() if len(word) > 3]
        title_matches = sum(1 for word in task_words if word in section_title_lower)
        text_matches = sum(1 for word in task_words if word in text_lower)
        
        score += title_matches * 0.15
        score += min(text_matches * 0.05, 0.2)  # Cap text matches contribution
        
        return min(score, 1.0)  # Cap at 1.0

    def _calculate_content_specificity(self, chunk: Dict) -> float:
        """Calculate how specific and actionable the content is"""
        text = chunk.get('text', '')
        
        # Look for specific details and structured content
        specificity_indicators = [
            (':', 0.1),     # Lists or detailed descriptions
            ('-', 0.05),    # Bullet points
            ('•', 0.05),    # Unicode bullet points
            ('1.', 0.1),    # Numbered lists
            ('2.', 0.05),   # Additional numbered items
            ('Table', 0.15), # Tables
            ('Figure', 0.15), # Figures
            ('Algorithm', 0.2), # Algorithms
            ('Equation', 0.15), # Mathematical content
        ]
        
        specificity_score = 0.0
        for indicator, weight in specificity_indicators:
            count = text.count(indicator)
            specificity_score += min(count * weight, weight * 3)  # Cap contribution per indicator
        
        # Check for numerical data
        import re
        numbers = re.findall(r'\d+(?:\.\d+)?%?', text)
        if len(numbers) > 3:
            specificity_score += 0.2
        
        # Check for citations or references
        citations = re.findall(r'\[\d+\]|\(\d+\)', text)
        if len(citations) > 2:
            specificity_score += 0.15
        
        return min(specificity_score, 1.0)

    def _calculate_section_importance(self, chunk: Dict) -> float:
        """Calculate importance based on section title and position"""
        section_title = chunk.get('section_title', '').lower()
        
        # Important section keywords
        important_sections = {
            'high': ['result', 'conclusion', 'finding', 'summary', 'discussion', 'analysis'],
            'medium': ['method', 'approach', 'implementation', 'experiment', 'evaluation'],
            'low': ['introduction', 'background', 'related work', 'acknowledgment', 'reference']
        }
        
        if any(keyword in section_title for keyword in important_sections['high']):
            return 0.3
        elif any(keyword in section_title for keyword in important_sections['medium']):
            return 0.2
        elif any(keyword in section_title for keyword in important_sections['low']):
            return 0.1
        
        return 0.15  # Default score

    def rank_chunks(self, chunks: List[Dict], queries: List[str], keywords: List[str], 
                   persona: str = "", task: str = "") -> List[Dict]:
        """
        Enhanced ranking with persona-specific and task-specific scoring
        """
        if not chunks:
            logger.warning("No chunks provided for ranking")
            return []
        
        if not queries:
            logger.warning("No queries provided for ranking")
            return chunks
        
        logger.info(f"Ranking {len(chunks)} chunks against {len(queries)} queries")
        
        # Extract domain-specific keywords
        domain_keywords = self._extract_domain_keywords(persona, task)
        all_keywords = keywords + domain_keywords
        
        # Extract text from chunks
        chunk_texts = [chunk.get('text', '') for chunk in chunks]
        
        # Generate embeddings for chunks and queries
        logger.info("Generating embeddings...")
        chunk_embeddings = self.model.encode(chunk_texts, show_progress_bar=False)
        query_embeddings = self.model.encode(queries, show_progress_bar=False)
        
        # Calculate cosine similarity between queries and chunks
        similarity_matrix = cosine_similarity(query_embeddings, chunk_embeddings)
        max_similarity_scores = np.max(similarity_matrix, axis=0)
        avg_similarity_scores = np.mean(similarity_matrix, axis=0)
        
        # Score chunks with enhanced criteria
        scored_chunks = []
        for i, chunk in enumerate(chunks):
            # Base semantic scores
            semantic_score = max_similarity_scores[i]
            avg_semantic_score = avg_similarity_scores[i]
            
            # Keyword matching score
            text_lower = chunk.get('text', '').lower()
            keyword_matches = sum(1 for keyword in all_keywords if keyword.lower() in text_lower)
            keyword_score = min(0.4, keyword_matches * 0.03)  # Increased weight for keywords
            
            # Section title relevance
            section_title = chunk.get('section_title', '').lower()
            title_keyword_matches = sum(1 for keyword in all_keywords if keyword.lower() in section_title)
            title_score = min(0.3, title_keyword_matches * 0.1)  # Increased weight for title matches
            
            # Persona and task alignment
            persona_score = self._calculate_persona_alignment(chunk, persona, task)
            
            # Content specificity and actionability
            specificity_score = self._calculate_content_specificity(chunk)
            
            # Section importance
            importance_score = self._calculate_section_importance(chunk)
            
            # Content density from chunker
            density_score = chunk.get('content_density', 0.5) * 0.15
            
            # Content length bonus (prefer substantial content)
            word_count = len(chunk.get('text', '').split())
            length_bonus = min(0.1, word_count / 1000)  # Bonus for longer, detailed content
            
            # Calculate final score with weighted components
            final_score = (
                semantic_score * 0.25 +           # Max semantic similarity
                avg_semantic_score * 0.10 +       # Average semantic similarity  
                keyword_score * 0.20 +            # Keyword matching
                title_score * 0.15 +              # Section title relevance
                persona_score * 0.15 +            # Persona/task alignment
                specificity_score * 0.08 +        # Content specificity
                importance_score * 0.05 +         # Section importance
                density_score * 0.02 +            # Content density
                length_bonus                      # Length bonus
            )
            
            # Add scores to chunk
            chunk_copy = chunk.copy()
            chunk_copy['relevance_score'] = final_score
            chunk_copy['semantic_score'] = semantic_score
            chunk_copy['keyword_matches'] = keyword_matches
            chunk_copy['persona_alignment'] = persona_score
            chunk_copy['specificity_score'] = specificity_score
            chunk_copy['importance_score'] = importance_score
            
            scored_chunks.append(chunk_copy)
        
        # Sort by relevance score
        ranked_chunks = sorted(scored_chunks, key=lambda x: x['relevance_score'], reverse=True)
        
        logger.info(f"Ranking complete. Top score: {ranked_chunks[0]['relevance_score']:.3f}")
        logger.info(f"Average score: {np.mean([c['relevance_score'] for c in ranked_chunks]):.3f}")
        
        return ranked_chunks