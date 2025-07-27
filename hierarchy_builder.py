import logging
import re
from typing import List, Dict, Any
from collections import defaultdict
import numpy as np
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)

class HierarchyBuilder:
    """Builds a refined hierarchical outline with multi-stage filtering and validation."""

    def _is_header_or_footer(self, block: Dict, page_height: float) -> bool:
        """Checks if a block is likely a header or footer based on its vertical position."""
        y0 = block['bbox'][1]
        return y0 < page_height * 0.08 or y0 > page_height * 0.92

    def _is_toc_entry(self, text: str) -> bool:
        """Checks if a line of text is likely a Table of Contents entry."""
        return bool(re.search(r'\.{5,}\s*\d+$', text))

    def _is_full_sentence(self, text: str, word_count: int) -> bool:
        """Checks if a block of text is likely a full sentence rather than a heading."""
        return word_count > 20 and text.endswith('.') and text[0].isupper()

    def _get_block_features(self, block: Dict, page_width: float) -> Dict:
        """Extracts key features from a text block for classification."""
        text = block['text'].strip()
        x0, y0, x1, y1 = block['bbox']
        word_count = len(text.split())

        return {
            "text": text,
            "font_size": block['font_size'],
            "is_bold": "bold" in block['font_name'].lower() or "black" in block['font_name'].lower(),
            "is_all_caps": text.isupper() and len(text) > 1,
            "starts_with_number": bool(re.match(r'^\d+(\.\d+)*\s*(\w+)', text)),
            "is_centered": abs((x0 + x1) / 2 - page_width / 2) < page_width * 0.1,
            "word_count": word_count,
            "is_sentence": self._is_full_sentence(text, word_count),
            "is_toc": self._is_toc_entry(text),
            "original_block": block
        }

    def _is_potential_heading(self, features: Dict) -> bool:
        """Determines if a block is a potential heading using stricter rules."""
        if not features['text'] or features['word_count'] > 25:
            return False
        if features['is_toc'] or features['is_sentence']:
            return False
        if features['text'].isdigit():
            return False
        
        if features['is_bold'] and features['word_count'] < 8 and features['font_size'] >= 11:
            return True

        if features['font_size'] > 20 and features['word_count'] < 15:
            return True
        if features['starts_with_number'] and features['word_count'] < 20:
            return True
        if features['is_bold'] and features['is_centered'] and features['word_count'] < 20:
            return True
        if features['is_all_caps'] and features['font_size'] > 13 and features['word_count'] < 20:
             return True

        return False

    def _extract_title(self, text_blocks: List[Dict], page_width: float) -> str:
        """More robust title extraction focusing on the top-center of the first page."""
        first_page_blocks = [b for b in text_blocks if b['page_num'] == 1 and len(b['text'].strip()) > 3]
        
        if not first_page_blocks:
            return "Untitled Document"

        candidates = []
        for block in first_page_blocks:
            if not block['text'].strip() or len(block['text'].strip()) > 250:
                continue
            
            x0, y0, x1, y1 = block['bbox']
            
            score = block['font_size'] * 1.5
            if abs((x0 + x1) / 2 - page_width / 2) < page_width * 0.2:
                score += 15
            if y0 < 200:
                score += 10

            candidates.append((score, block['text'].strip()))

        return max(candidates, key=lambda item: item[0])[1] if candidates else "Untitled Document"

    def _assign_heading_levels(self, heading_candidates: List[Dict]) -> List[Dict]:
        """Assigns heading levels (H1, H2, H3) using clustering on a clean set of font sizes."""
        # BUG FIX: Handle the case of 0 or 1 headings by building the correct output structure.
        if len(heading_candidates) < 2:
            outline = []
            for features in heading_candidates:
                block = features['original_block']
                outline.append({
                    "level": 'H1',  # Assign H1 if it's the only heading
                    "text": features['text'],
                    "page": block['page_num'],
                    "y_pos": block['bbox'][1]
                })
            return outline

        font_sizes = np.array([[h['font_size']] for h in heading_candidates])
        unique_sizes = np.unique(font_sizes)
        
        n_clusters = min(len(unique_sizes), 3)
        if n_clusters == 0:
            return []

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto').fit(font_sizes)
        cluster_centers = sorted(kmeans.cluster_centers_.flatten(), reverse=True)
        
        size_to_level = {center: f"H{i+1}" for i, center in enumerate(cluster_centers)}
        
        outline = []
        for i, features in enumerate(heading_candidates):
            cluster_label = kmeans.labels_[i]
            cluster_center = kmeans.cluster_centers_[cluster_label][0]
            level = size_to_level[cluster_center]
            
            block = features['original_block']
            outline.append({
                "level": level,
                "text": features['text'],
                "page": block['page_num'],
                "y_pos": block['bbox'][1]
            })
            
        return outline

    def build(self, processed_pages: List[Dict]) -> Dict[str, Any]:
        """Constructs the final document outline using a multi-stage process."""
        if not processed_pages:
            return {"title": "No Content Found", "outline": []}

        all_blocks = [block for page in processed_pages for block in page['text_blocks']]
        page_height = processed_pages[0]['height']
        page_width = processed_pages[0]['width']
        
        core_content_blocks = [b for b in all_blocks if not self._is_header_or_footer(b, page_height)]
        title = self._extract_title(core_content_blocks, page_width)
        block_features = [self._get_block_features(b, page_width) for b in core_content_blocks]
        heading_candidates = [f for f in block_features if self._is_potential_heading(f)]
        outline = self._assign_heading_levels(heading_candidates)
        
        # BUG FIX: Sort here, but DO NOT delete y_pos. It's needed for section content grouping later.
        outline.sort(key=lambda x: (x['page'], x['y_pos']))

        return {"title": title.replace("\n", " ").strip(), "outline": outline}