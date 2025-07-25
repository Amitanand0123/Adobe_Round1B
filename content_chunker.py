# content_chunker.py

import logging
import re
from typing import List, Dict

logger = logging.getLogger(__name__)

class AdvancedContentChunker:
    """
    Intelligently groups raw text blocks into meaningful content chunks
    """
    MIN_CHUNK_WORD_COUNT = 50
    MAX_CHUNK_WORD_COUNT = 700

    def __init__(self):
        logger.info("Advanced Content Chunker initialized")

    def _is_bullet_point(self, text: str) -> bool:
        """Checks if text starts with a common bullet marker."""
        return bool(re.match(r'^\s*[•*-]', text.strip()))

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Replace bullet points with dashes
        text = re.sub(r'^\s*[•*]\s*', '- ', text)
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _calculate_content_density(self, text: str) -> float:
        """Calculate how content-rich a text block is"""
        words = text.split()
        if len(words) < 5:
            return 0.0
        
        # Count meaningful content indicators
        content_indicators = 0
        
        # Check for descriptive content
        if any(word.lower() in ['include', 'such', 'example', 'like', 'features', 'offers', 'provides'] for word in words):
            content_indicators += 1
        
        # Check for specific details
        if re.search(r'\d+', text):  # Contains numbers
            content_indicators += 1
        
        # Check for location/place names (capitalized words)
        capitalized_words = [word for word in words if word[0].isupper() and len(word) > 2]
        if len(capitalized_words) > 2:
            content_indicators += 1
        
        # Check for lists or enumerations
        if ':' in text and len(text.split(':')) > 1:
            content_indicators += 1
        
        return min(content_indicators / 4.0, 1.0)  # Normalize to 0-1

    def _is_meaningful_content(self, text: str) -> bool:
        """Determine if text contains meaningful, specific content"""
        if len(text.split()) < 25:
            return False
        
        # Avoid generic introductory text
        generic_starters = [
            'this guide', 'this section', 'introduction', 'overview', 
            'in this chapter', 'welcome to', 'planning a trip'
        ]
        
        text_lower = text.lower()
        if any(starter in text_lower[:50] for starter in generic_starters):
            return False
        
        # Look for specific, actionable content
        specific_indicators = [
            ':', '-', 'visit', 'try', 'enjoy', 'explore', 'experience',
            'located', 'offers', 'features', 'includes', 'recommended'
        ]
        
        return any(indicator in text_lower for indicator in specific_indicators)

    def _finalize_chunk(self, blocks: List[Dict], section_info: Dict) -> Dict:
        """Creates a chunk dictionary from a list of blocks."""
        if not blocks:
            return None
        
        # Combine text from all blocks
        chunk_text = " ".join([self._clean_text(b['text']) for b in blocks if b['text'].strip()])
        
        # Check minimum word count
        if len(chunk_text.split()) < self.MIN_CHUNK_WORD_COUNT:
            return None

        # Check if this is meaningful content
        if not self._is_meaningful_content(chunk_text):
            return None

        # Truncate if too long
        words = chunk_text.split()

        # Calculate content quality score
        content_density = self._calculate_content_density(chunk_text)

        return {
            'text': chunk_text,
            'page_number': blocks[0]['page_num'],
            'document': section_info['document'],
            'section_title': section_info['title'],
            'bbox': blocks[0]['bbox'],
            'content_density': content_density,
            'word_count': len(words)
        }

    def chunk_document_content(self, all_blocks: List[Dict], outline: List[Dict]) -> List[Dict]:
        """Groups text blocks into sections and creates meaningful chunks."""
        if not all_blocks or not outline:
            logger.warning("No blocks or outline provided for chunking")
            return []

        # Group blocks by section
        sections = self._group_blocks_by_section(all_blocks, outline)
        
        all_chunks = []
        for section in sections:
            section_chunks = self._create_chunks_from_blocks(section['content_blocks'], section)
            all_chunks.extend(section_chunks)
            
        # Filter out low-quality chunks
        quality_chunks = [chunk for chunk in all_chunks if chunk.get('content_density', 0) > 0.2]
        
        logger.info(f"Created {len(quality_chunks)} quality chunks from {len(all_blocks)} blocks")
        return quality_chunks

    def _group_blocks_by_section(self, all_blocks: List[Dict], outline: List[Dict]) -> List[Dict]:
        """Assigns each text block to its corresponding section from the outline."""
        sections = []
        
        # Sort outline by page and position
        sorted_outline = sorted(outline, key=lambda x: (x.get('page', 0), x.get('y_pos', 0)))
        
        for i, heading in enumerate(sorted_outline):
            start_page = heading.get('page', 1)
            start_y = heading.get('y_pos', 0)
            
            # Determine end boundaries
            if i + 1 < len(sorted_outline):
                next_heading = sorted_outline[i + 1]
                end_page = next_heading.get('page', float('inf'))
                end_y = next_heading.get('y_pos', float('inf'))
            else:
                end_page = float('inf')
                end_y = float('inf')

            # Find blocks that belong to this section
            content_blocks = []
            for block in all_blocks:
                block_page = block.get('page_num', 1)
                block_y = block.get('bbox', [0, 0, 0, 0])[1]  # y0 coordinate
                
                # Skip if block text matches heading text (case insensitive)
                if block['text'].strip().lower() == heading['text'].strip().lower():
                    continue
                
                # Check if block falls within section boundaries
                in_section = False
                
                if block_page > start_page and block_page < end_page:
                    in_section = True
                elif block_page == start_page and block_page == end_page:
                    in_section = (block_y >= start_y and block_y < end_y)
                elif block_page == start_page:
                    in_section = (block_y >= start_y)
                elif block_page == end_page:
                    in_section = (block_y < end_y)
                
                if in_section:
                    content_blocks.append(block)

            # Only create section if it has meaningful content
            if content_blocks and len(content_blocks) > 0:
                section_context = {
                    "title": heading['text'],
                    "page": start_page,
                    "document": heading.get('document', 'unknown'),
                    "content_blocks": content_blocks
                }
                sections.append(section_context)
        
        return sections

    def _create_chunks_from_blocks(self, blocks: List[Dict], section_context: Dict) -> List[Dict]:
        """
        Converts a list of text blocks into meaningful chunks within a section.
        """
        if not blocks:
            return []

        chunks = []
        current_chunk_blocks = []
        
        for i, block in enumerate(blocks):
            text = block['text'].strip()
            if not text:
                continue
                
            # Check if this starts a list (ends with colon and next block is bullet)
            is_list_intro = (
                text.endswith(':') and 
                i + 1 < len(blocks) and 
                self._is_bullet_point(blocks[i + 1]['text'])
            )
            
            if is_list_intro:
                # Finalize current chunk if exists
                if current_chunk_blocks:
                    chunk = self._finalize_chunk(current_chunk_blocks, section_context)
                    if chunk:
                        chunks.append(chunk)
                
                # Start new chunk with list intro
                current_chunk_blocks = [block]
                
                # Add following bullet points
                j = i + 1
                while j < len(blocks) and self._is_bullet_point(blocks[j]['text']):
                    current_chunk_blocks.append(blocks[j])
                    j += 1
                
                # Finalize list chunk
                chunk = self._finalize_chunk(current_chunk_blocks, section_context)
                if chunk:
                    chunks.append(chunk)
                
                current_chunk_blocks = []
                
            else:
                # Regular text block
                if current_chunk_blocks:
                    # Check if we should start a new chunk
                    prev_block = current_chunk_blocks[-1]
                
                    # Calculate vertical gap
                    vertical_gap = block['bbox'][1] - prev_block['bbox'][3]
                    avg_font_size = (block.get('font_size', 12) + prev_block.get('font_size', 12)) / 2
                
                    # Start new chunk if large vertical gap or significant content change
                    is_new_chunk = (
                        vertical_gap > avg_font_size * 2.5 or  # Changed from 1.8 to 2.5
                        abs(block.get('font_size', 12) - prev_block.get('font_size', 12)) > 3 or  # Changed from 2 to 3
                        len(current_chunk_blocks) > 15  # Changed from 8 to 15
                    )
                    
                    if is_new_chunk:
                        # Finalize current chunk
                        chunk = self._finalize_chunk(current_chunk_blocks, section_context)
                        if chunk:
                            chunks.append(chunk)
                        current_chunk_blocks = [block]
                    else:
                        current_chunk_blocks.append(block)
                else:
                    current_chunk_blocks = [block]

        # Finalize remaining chunk
        if current_chunk_blocks:
            chunk = self._finalize_chunk(current_chunk_blocks, section_context)
            if chunk:
                chunks.append(chunk)

        return chunks