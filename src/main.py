# main.py

import json
import time
import logging
from pathlib import Path
import datetime
from collections import OrderedDict

from src.pdf_processor import PDFProcessor
from src.hierarchy_builder import HierarchyBuilder
from src.content_chunker import AdvancedContentChunker
from src.nlp_ranker import NLPRanker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

def generate_persona_keywords(task_text: str) -> tuple[list[str], list[str]]:
    """Generates lists of positive and negative keywords based on the task."""
    positive_keywords = ['guide', 'tips', 'plan', 'activities', 'restaurants', 'hotels', 'attractions']
    negative_keywords = []

    if "college friends" in task_text or "young adults" in task_text:
        positive_keywords.extend(['nightlife', 'bars', 'clubs', 'party', 'entertainment', 'live music', 
                                  'budget-friendly', 'affordable', 'beach', 'coastal', 'adventure', 'social'])
        # Add negative keywords to filter out irrelevant content
        negative_keywords.extend(['kids', 'children', 'family-friendly', 'family'])

    if "4 days" in task_text:
        positive_keywords.extend(['itinerary', 'short trip', 'weekend'])
        
    return list(set(positive_keywords)), list(set(negative_keywords))

def process_collection(input_json_path: Path, base_pdf_path: Path, output_dir: Path):
    logger.info(f"--- Starting Analysis for {input_json_path.name} ---")

    with open(input_json_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    documents_info = config.get("documents", [])
    persona_info = config.get("persona", {})
    jtbd_info = config.get("job_to_be_done", {})

    processor = PDFProcessor()
    builder = HierarchyBuilder()
    chunker = AdvancedContentChunker()
    ranker = NLPRanker() # Using the advanced NLP ranker

    job_task = jtbd_info.get('task', '')
    persona_role = persona_info.get('role', '')

    queries = [
        "Fun activities, nightlife, and entertainment for a group of college friends in the South of France",
        "Budget-friendly restaurants and bars for young adults on a 4-day trip",
        "Coastal adventures, beach hopping, and water sports for a group trip",
        "A 4-day travel itinerary suitable for a group of 10 friends looking for social events"
    ]
    
    positive_keywords, negative_keywords = generate_persona_keywords(job_task)

    # Step 1: Process all documents and create detailed content chunks
    all_subsection_chunks = []
    for doc_info in documents_info:
        filename = doc_info.get("filename")
        pdf_path = base_pdf_path / filename
        if not pdf_path.exists(): continue

        logger.info(f"Processing document: {filename}")
        processed_pages = processor.process_pdf(str(pdf_path))
        if not processed_pages: continue

        outline_data = builder.build(processed_pages)
        for heading in outline_data['outline']:
            heading['document'] = filename
        
        all_blocks = [block for page in processed_pages for block in page['text_blocks']]
        # The chunker now adds document/section context to each chunk
        doc_subsections = chunker.chunk_document_content(all_blocks, outline_data['outline'])
        all_subsection_chunks.extend(doc_subsections)

    # Step 2: Rank only the detailed content chunks using our most advanced method
    logger.info(f"Ranking {len(all_subsection_chunks)} content chunks with NLP model...")
    
    final_scores = {i: -1.0 for i in range(len(all_subsection_chunks))}
    for query in queries:
        # The ranker scores each chunk against the query
        query_ranked_chunks = ranker.rank(query, [dict(chunk) for chunk in all_subsection_chunks])
        
        for ranked_chunk in query_ranked_chunks:
            try:
                original_index = all_subsection_chunks.index(ranked_chunk)
                if ranked_chunk['relevance'] > final_scores[original_index]:
                    final_scores[original_index] = ranked_chunk['relevance']
            except ValueError:
                continue
    
    for i, chunk in enumerate(all_subsection_chunks):
        base_score = final_scores[i]
        text_lower = chunk['text'].lower()
        bonus = 0.05 * sum(1 for kw in positive_keywords if kw in text_lower)
        penalty = 0.2 * sum(1 for kw in negative_keywords if kw in text_lower)
        chunk['relevance'] = base_score + bonus - penalty
        
    ranked_subsections = sorted(all_subsection_chunks, key=lambda x: x.get('relevance', -1), reverse=True)

    # Step 3: Dynamically derive the top sections from the top-ranked subsections
    top_sections = OrderedDict()
    for chunk in ranked_subsections:
        section_key = (chunk['document'], chunk['section_title'])
        if section_key not in top_sections:
            top_sections[section_key] = {
                "document": chunk['document'],
                "section_title": chunk['section_title'],
                "page_number": chunk['page_number'], # Use the page of the chunk
                "relevance": chunk['relevance']
            }
        if len(top_sections) >= 5:
            break
            
    extracted_sections_output = []
    for i, section in enumerate(top_sections.values()):
        extracted_sections_output.append({
            "document": section['document'],
            "section_title": section['section_title'],
            "importance_rank": i + 1,
            "page_number": section['page_number']
        })

    # Step 4: Format the output
    subsection_analysis_output = []
    for sub in ranked_subsections[:5]:
        subsection_analysis_output.append({
            "document": sub['document'],
            "refined_text": sub['text'],
            "page_number": sub['page_number']
        })

    output_data = {
        "metadata": {
            "input_documents": [doc['filename'] for doc in documents_info],
            "persona": persona_role,
            "job_to_be_done": job_task,
            "processing_timestamp": datetime.datetime.now().isoformat()
        },
        "extracted_sections": extracted_sections_output,
        "subsection_analysis": subsection_analysis_output
    }

    output_path = output_dir / "challenge1b_output.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)

    logger.info(f"Analysis complete. Output saved to {output_path}")

# main() function and __main__ block remain the same
def main():
    input_dir = Path("./input") # Adjusted for local testing
    output_dir = Path("./output")
    input_json = input_dir / "challenge1b_input.json"
    pdf_dir = input_dir / "PDFs"
    
    if not input_json.exists() or not pdf_dir.is_dir():
        logger.error(f"Critical error: Input files not found in {input_dir}.")
        return

    output_dir.mkdir(exist_ok=True, parents=True)
    process_collection(input_json, pdf_dir, output_dir)

if __name__ == "__main__":
    main()