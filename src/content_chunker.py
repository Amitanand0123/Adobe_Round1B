# content_chunker.py

import logging
import re
from typing import List, Dict

logger = logging.getLogger(__name__)

class AdvancedContentChunker:
    """
    Intelligently groups raw text blocks into meaningful paragraphs and lists ("chunks")
    using context-aware block grouping.
    """
    MIN_CHUNK_WORD_COUNT = 20  # Increased slightly to favor more substantial chunks
    MAX_CHUNK_WORD_COUNT = 400

    def _is_bullet_point(self, text: str) -> bool:
        """Checks if text starts with a common bullet marker."""
        return bool(re.match(r'^\s*[•*-]', text))

    def _finalize_chunk(self, blocks: List[Dict]) -> Dict:
        """Creates a chunk dictionary from a list of blocks."""
        if not blocks:
            return None
        
        chunk_text = " ".join([b['text'] for b in blocks]).strip()
        chunk_text = re.sub(r'\s*([•*-])\s+', ' - ', chunk_text)
        chunk_text = re.sub(r'\s{2,}', ' ', chunk_text)

        # FIX: Ensure the finalized chunk meets the minimum word count
        if len(chunk_text.split()) >= self.MIN_CHUNK_WORD_COUNT:
            return {
                'text': chunk_text,
                'page_number': blocks[0]['page_num'],
            }
        return None

    def chunk_document_content(self, all_blocks: List[Dict], outline: List[Dict]) -> List[Dict]:
        """Groups text blocks into sections and then creates meaningful chunks."""
        if not all_blocks or not outline:
            return []

        sections = self._group_blocks_by_section(all_blocks, outline)
        
        all_chunks = []
        for section in sections:
            # Pass the section context to the chunking method
            section_chunks = self._create_chunks_from_blocks(section['content_blocks'], section)
            all_chunks.extend(section_chunks)
            
        return all_chunks

    def _group_blocks_by_section(self, all_blocks: List[Dict], outline: List[Dict]) -> List[Dict]:
        """Assigns each text block to its corresponding heading from the outline."""
        sections = []
        sorted_outline = sorted(outline, key=lambda x: (x['page'], x.get('y_pos', 0)))

        for i, heading in enumerate(sorted_outline):
            start_page, start_y = heading['page'], heading.get('y_pos', 0)
            end_page, end_y = (float('inf'), float('inf'))
            if i + 1 < len(sorted_outline):
                next_heading = sorted_outline[i+1]
                end_page, end_y = next_heading['page'], next_heading.get('y_pos', 0)

            content_blocks = [
                block for block in all_blocks
                if (block['page_num'] > start_page or (block['page_num'] == start_page and block['bbox'][1] >= start_y))
                and (block['page_num'] < end_page or (block['page_num'] == end_page and block['bbox'][1] < end_y))
                and block['text'].strip().lower() != heading['text'].strip().lower()
            ]
            
            # Attach document context to the section
            section_context = {
                "title": heading['text'],
                "page": heading['page'],
                "document": heading['document'],
                "content_blocks": content_blocks
            }
            sections.append(section_context)
        return sections

    def _create_chunks_from_blocks(self, blocks: List[Dict], section_context: Dict) -> List[Dict]:
        """
        Converts a list of text blocks into paragraph-like and list-like chunks.
        """
        if not blocks:
            return []

        chunks = []
        current_chunk_blocks = []
        i = 0
        while i < len(blocks):
            block = blocks[i]
            
            is_list_intro = block['text'].strip().endswith(':') and (i + 1 < len(blocks)) and self._is_bullet_point(blocks[i+1]['text'])
            
            if is_list_intro:
                if current_chunk_blocks:
                    final_chunk = self._finalize_chunk(current_chunk_blocks)
                    if final_chunk: chunks.append(final_chunk)
                
                current_chunk_blocks = [block]
                i += 1
                
                while i < len(blocks) and self._is_bullet_point(blocks[i]['text']):
                    current_chunk_blocks.append(blocks[i])
                    i += 1
                
                final_chunk = self._finalize_chunk(current_chunk_blocks)
                if final_chunk: chunks.append(final_chunk)
                current_chunk_blocks = []
            else:
                if not current_chunk_blocks:
                    current_chunk_blocks.append(block)
                else:
                    prev_block = current_chunk_blocks[-1]
                    vertical_gap = block['bbox'][1] - prev_block['bbox'][3]
                    is_new_para = vertical_gap > (prev_block['font_size'] * 0.75)
                    
                    if is_new_para:
                        final_chunk = self._finalize_chunk(current_chunk_blocks)
                        if final_chunk: chunks.append(final_chunk)
                        current_chunk_blocks = [block]
                    else:
                        current_chunk_blocks.append(block)
                i += 1

        if current_chunk_blocks:
            final_chunk = self._finalize_chunk(current_chunk_blocks)
            if final_chunk: chunks.append(final_chunk)

        # Attach document and section context to each created chunk
        for chunk in chunks:
            chunk['document'] = section_context['document']
            chunk['section_title'] = section_context['title']

        return chunks