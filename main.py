# main.py

import json
import time
import logging
import datetime
from pathlib import Path
from collections import OrderedDict
from typing import List, Dict

from pdf_processor import PDFProcessor
from hierarchy_builder import HierarchyBuilder
from content_chunker import AdvancedContentChunker
from semantic_ranker import SemanticRanker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

def extract_persona_keywords(persona_text: str) -> List[str]:
    """Extract relevant keywords from persona description"""
    keywords = []
    text_lower = persona_text.lower()
    
    # Domain-specific keywords based on persona
    if any(term in text_lower for term in ['travel', 'planner', 'tourism']):
        keywords.extend(['destination', 'attractions', 'activities', 'sightseeing', 'experience', 
                        'visit', 'explore', 'itinerary', 'recommendations', 'guide'])
    
    if any(term in text_lower for term in ['researcher', 'phd', 'academic', 'research']):
        keywords.extend(['methodology', 'research', 'study', 'analysis', 'findings', 'results',
                        'literature', 'review', 'data', 'experiment'])
    
    if any(term in text_lower for term in ['student', 'undergraduate', 'learning']):
        keywords.extend(['concept', 'definition', 'example', 'theory', 'principle', 'study',
                        'exam', 'practice', 'key points'])
    
    if any(term in text_lower for term in ['analyst', 'business', 'investment']):
        keywords.extend(['revenue', 'profit', 'growth', 'market', 'financial', 'strategy',
                        'performance', 'trends', 'analysis'])
    
    # Extract key terms from the persona text itself
    words = persona_text.lower().split()
    important_words = [w.strip('.,!?()[]') for w in words if len(w) > 3 and w.isalpha()]
    keywords.extend(important_words[:8])
    
    return list(set(keywords))

def extract_task_keywords(task_text: str) -> List[str]:
    """Extract relevant keywords from job-to-be-done description"""
    keywords = []
    text_lower = task_text.lower()
    
    # Task-specific keywords
    if 'trip' in text_lower or 'travel' in text_lower:
        keywords.extend(['travel', 'trip', 'vacation', 'journey', 'destination', 'places'])
    
    if 'group' in text_lower or 'friends' in text_lower:
        keywords.extend(['group', 'friends', 'together', 'social', 'team', 'collective'])
    
    if 'plan' in text_lower:
        keywords.extend(['plan', 'planning', 'organize', 'schedule', 'itinerary', 'prepare'])
    
    if 'days' in text_lower:
        keywords.extend(['days', 'duration', 'time', 'schedule', 'daily'])
    
    if 'literature review' in text_lower:
        keywords.extend(['literature', 'review', 'methodology', 'research', 'papers'])
    
    if 'financial' in text_lower or 'revenue' in text_lower:
        keywords.extend(['financial', 'revenue', 'profit', 'cost', 'investment', 'money'])
    
    if 'exam' in text_lower or 'study' in text_lower:
        keywords.extend(['study', 'exam', 'preparation', 'concepts', 'key', 'important'])
    
    # Extract important words from task description
    words = task_text.lower().split()
    important_words = [w.strip('.,!?()[]') for w in words if len(w) > 3 and w.isalpha()]
    keywords.extend(important_words[:12])
    
    return list(set(keywords))

def filter_and_rank_sections(ranked_chunks: List[Dict], max_sections: int = 5) -> List[Dict]:
    """Filter and rank sections ensuring diversity and relevance"""
    seen_sections = set()
    unique_sections = []
    
    for chunk in ranked_chunks:
        section_key = (chunk['document'], chunk['section_title'])
        
        if section_key not in seen_sections:
            seen_sections.add(section_key)
            unique_sections.append({
                'document': chunk['document'],
                'section_title': chunk['section_title'],
                'page_number': chunk['page_number'],
                'relevance_score': chunk['relevance_score'],
                'importance_rank': len(unique_sections) + 1,
                'chunk': chunk
            })
            
            if len(unique_sections) >= max_sections:
                break
    
    return unique_sections

def select_best_subsections(ranked_chunks: List[Dict], max_subsections: int = 5) -> List[Dict]:
    """Select the best subsections ensuring diversity across documents"""
    selected_subsections = []
    document_count = {}
    
    for chunk in ranked_chunks:
        doc_name = chunk['document']
        
        # Limit subsections per document to ensure diversity
        if document_count.get(doc_name, 0) >= 2:
            continue
        
        # Ensure the chunk has good content quality
        if (chunk.get('relevance_score', 0) > 0.3 and 
            chunk.get('specificity_score', 0) > 0.2 and
            len(chunk.get('text', '').split()) >= 20):
            
            selected_subsections.append({
                'document': chunk['document'],
                'section_title': chunk['section_title'],
                'refined_text': chunk['text'][:500] + "..." if len(chunk['text']) > 500 else chunk['text'],
                'page_number': chunk['page_number'],
                'relevance_score': chunk['relevance_score']
            })
            document_count[doc_name] = document_count.get(doc_name, 0) + 1
            
            if len(selected_subsections) >= max_subsections:
                break
    
    return selected_subsections

def process_collection(input_json_path: Path, base_pdf_path: Path, output_dir: Path):
    """Enhanced processing function for document collection"""
    logger.info(f"--- Starting Enhanced Analysis for {input_json_path.name} ---")
    start_time = time.time()

    # Load configuration
    with open(input_json_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # Initialize processors
    pdf_processor = PDFProcessor()
    hierarchy_builder = HierarchyBuilder()
    content_chunker = AdvancedContentChunker()
    semantic_ranker = SemanticRanker()

    # Extract configuration details
    job_task = config.get("job_to_be_done", {}).get('task', '')
    persona_info = config.get("persona", {})
    persona_role = persona_info.get('role', '')
    persona_description = persona_info.get('description', '')

    logger.info(f"Processing for persona: {persona_role}")
    logger.info(f"Task: {job_task}")

    # Extract keywords from persona and task
    persona_keywords = extract_persona_keywords(f"{persona_role} {persona_description}")
    task_keywords = extract_task_keywords(job_task)
    all_keywords = list(set(persona_keywords + task_keywords))
    
    logger.info(f"Extracted keywords: {all_keywords[:15]}...")

    # Process all documents
    all_subsection_chunks = []
    document_sections = {}
    
    for doc_info in config.get("documents", []):
        filename = doc_info.get("filename")
        pdf_path = base_pdf_path / filename
        
        if not pdf_path.exists():
            logger.warning(f"PDF not found: {pdf_path}")
            continue

        logger.info(f"Processing document: {filename}")
        
        # Process PDF
        processed_pages = pdf_processor.process_pdf(str(pdf_path))
        if not processed_pages:
            logger.warning(f"No pages extracted from {filename}")
            continue

        # Build document hierarchy
        outline_data = hierarchy_builder.build(processed_pages)
        
        # Add document name to outline entries
        for heading in outline_data['outline']:
            heading['document'] = filename
        
        # Store sections for this document
        document_sections[filename] = outline_data['outline']
        
        # Extract all text blocks
        all_blocks = []
        for page in processed_pages:
            for block in page['text_blocks']:
                all_blocks.append(block)
        
        # Create content chunks
        doc_chunks = content_chunker.chunk_document_content(all_blocks, outline_data['outline'])
        all_subsection_chunks.extend(doc_chunks)

    if not all_subsection_chunks:
        logger.error("No content chunks extracted from any document")
        return

    logger.info(f"Total content chunks extracted: {len(all_subsection_chunks)}")

    # Create enhanced search queries
    search_queries = [
        f"{persona_role} {job_task}",
        job_task,
        f"{persona_role} recommendations",
        f"practical {job_task}",
        f"guide for {job_task}"
    ]

    # Rank content using enhanced semantic similarity
    logger.info("Ranking content chunks with enhanced scoring...")
    ranked_chunks = semantic_ranker.rank_chunks(
        chunks=all_subsection_chunks,
        queries=search_queries,
        keywords=all_keywords,
        persona=f"{persona_role} {persona_description}",
        task=job_task
    )

    # Filter and select best sections and subsections
    top_sections = filter_and_rank_sections(ranked_chunks, max_sections=5)
    best_subsections = select_best_subsections(ranked_chunks, max_subsections=5)

    # Prepare output data according to the exact JSON format required
    output_data = {
        "metadata": {
            "input_documents": [d['filename'] for d in config.get("documents", [])],
            "persona": persona_role,
            "job_to_be_done": job_task,
            "processing_timestamp": datetime.datetime.now().isoformat()
        },
        "extracted_sections": [
            {
                "document": section['document'],
                "section_title": section['section_title'],
                "page_number": section['page_number'],
                "importance_rank": section['importance_rank']
            }
            for section in top_sections
        ],
        "subsection_analysis": [
            {
                "document": subsection['document'],
                "section_title": subsection['section_title'],
                "refined_text": subsection['refined_text'],
                "page_number": subsection['page_number']
            }
            for subsection in best_subsections
        ]
    }

    # Save output
    output_file = output_dir / f"{input_json_path.stem}_analysis.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    processing_time = time.time() - start_time
    logger.info(f"Analysis complete for {input_json_path.name}")
    logger.info(f"Processing time: {processing_time:.2f} seconds")
    logger.info(f"Output saved to: {output_file}")
    logger.info(f"Extracted {len(top_sections)} sections and {len(best_subsections)} subsections")

def main():
    """Main entry point for document collection processing"""
    import sys
    import os
    
    # Get input and output directories from environment or default
    input_dir = Path(os.environ.get('INPUT_DIR', '/app/input'))
    output_dir = Path(os.environ.get('OUTPUT_DIR', '/app/output'))
    
    if len(sys.argv) > 1:
        input_dir = Path(sys.argv[1])
    if len(sys.argv) > 2:
        output_dir = Path(sys.argv[2])
    
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return
    
    # Find all JSON configuration files
    json_files = list(input_dir.glob('*.json'))
    
    if not json_files:
        logger.error("No JSON configuration files found in input directory")
        return
    
    logger.info(f"Found {len(json_files)} configuration files")
    
    # Process each configuration file
    total_start_time = time.time()
    
    for json_file in json_files:
        try:
            logger.info(f"\n{'='*60}")
            process_collection(json_file, input_dir, output_dir)
        except Exception as e:
            logger.error(f"Failed to process {json_file.name}: {str(e)}")
            continue
    
    total_time = time.time() - total_start_time
    logger.info(f"\n{'='*60}")
    logger.info(f"Total processing time: {total_time:.2f} seconds")
    logger.info("All processing complete!")

if __name__ == "__main__":
    main()