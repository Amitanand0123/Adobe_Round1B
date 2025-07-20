import logging
import pdfplumber
from pdf2image import convert_from_path
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

class PDFProcessor:
    """Processes a PDF to extract text elements and page images with improved text block handling."""

    def process_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extracts structured text data and images from each page of a PDF with better text grouping.
        """
        pages_data = []
        
        try:
            logger.info(f"Rendering PDF pages to images for '{pdf_path}'...")
            page_images = convert_from_path(pdf_path, dpi=150)
        except Exception as e:
            logger.error(f"Failed to render PDF to images: {e}")
            return []

        logger.info("Opening PDF with pdfplumber to extract text...")
        with pdfplumber.open(pdf_path) as pdf:
            for i, (page, image) in enumerate(zip(pdf.pages, page_images)):
                page_num = i + 1
                logger.info(f"Processing Page {page_num}...")
                
                words = page.extract_words(
                    x_tolerance=1.5,
                    y_tolerance=3,
                    keep_blank_chars=False,
                    use_text_flow=True,
                    horizontal_ltr=True,
                    extra_attrs=["fontname", "size", "mcid"]
                )
                
                # Add page number to each word for consistency
                for word in words:
                    word['page_num'] = page_num

                text_blocks = self._group_words_into_blocks(words)
                
                pages_data.append({
                    'page_num': page_num,
                    'width': float(page.width),
                    'height': float(page.height),
                    'image': image,
                    'text_blocks': text_blocks,
                })
        
        logger.info("Finished processing all pages.")
        return pages_data

    def _unify_bbox(self, bboxes: List[Tuple[float, float, float, float]]) -> Tuple[float, float, float, float]:
        """Calculates a bounding box that encompasses all given boxes."""
        x0 = min(b[0] for b in bboxes)
        y0 = min(b[1] for b in bboxes)
        x1 = max(b[2] for b in bboxes)
        y1 = max(b[3] for b in bboxes)
        return (x0, y0, x1, y1)

    def _group_words_into_blocks(self, words: List[Dict]) -> List[Dict]:
        """Groups words into logical text blocks based on lines and paragraphs."""
        if not words:
            return []
        
        lines = []
        if words:
            sorted_words = sorted(words, key=lambda w: (w['top'], w['x0']))
            current_line = [sorted_words[0]]
            for word in sorted_words[1:]:
                if abs(word['top'] - current_line[-1]['top']) < 5: # Tolerance for same line
                    current_line.append(word)
                else:
                    lines.append(sorted(current_line, key=lambda w: w['x0']))
                    current_line = [word]
            lines.append(sorted(current_line, key=lambda w: w['x0']))

        blocks = []
        if lines:
            current_block_lines = [lines[0]]
            for line in lines[1:]:
                prev_line = current_block_lines[-1]
                
                vertical_gap = line[0]['top'] - prev_line[0]['top']
                font_name_match = line[0]['fontname'] == prev_line[0]['fontname']
                font_size_match = abs(line[0]['size'] - prev_line[0]['size']) < 1

                is_new_block = (vertical_gap > prev_line[0]['size'] * 1.5) or not font_name_match or not font_size_match

                if is_new_block:
                    text = " ".join(" ".join(w['text'] for w in ln) for ln in current_block_lines)
                    bboxes = [ (w['x0'], w['top'], w['x1'], w['bottom']) for ln in current_block_lines for w in ln ]
                    blocks.append({
                        "text": text,
                        "bbox": self._unify_bbox(bboxes),
                        "font_name": current_block_lines[0][0]['fontname'],
                        "font_size": round(current_block_lines[0][0]['size'], 2),
                        "page_num": current_block_lines[0][0]['page_num']
                    })
                    current_block_lines = [line]
                else:
                    current_block_lines.append(line)

            text = " ".join(" ".join(w['text'] for w in ln) for ln in current_block_lines)
            bboxes = [ (w['x0'], w['top'], w['x1'], w['bottom']) for ln in current_block_lines for w in ln ]
            blocks.append({
                "text": text,
                "bbox": self._unify_bbox(bboxes),
                "font_name": current_block_lines[0][0]['fontname'],
                "font_size": round(current_block_lines[0][0]['size'], 2),
                "page_num": current_block_lines[0][0]['page_num']
            })

        return blocks