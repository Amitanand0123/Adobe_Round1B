Round 1B: Persona-Driven Document Intelligence

## Overview
Our solution implements a semantic-driven document analysis system that intelligently extracts and prioritizes content based on persona requirements and job-to-be-done tasks.

## Core Components

### 1. Document Processing Pipeline
- **PDF Processing**: Uses pdfplumber to extract structured text with positional information
- **Hierarchy Building**: Identifies document structure (headings, sections) using font size, positioning, and formatting cues
- **Content Chunking**: Groups related text blocks into meaningful semantic units while preserving context

### 2. Semantic Understanding
- **Sentence Transformers**: Employs `all-MiniLM-L6-v2` model for generating high-quality text embeddings
- **Multi-Query Matching**: Creates relevant search queries from persona and task descriptions
- **Cosine Similarity**: Calculates semantic similarity between content chunks and user requirements

### 3. Intelligent Ranking System
Our ranking combines multiple signals:
- **Semantic Similarity**: Primary relevance score using transformer embeddings
- **Keyword Matching**: Bonus scoring for domain-specific terms extracted from persona/task
- **Section Title Relevance**: Higher weight for content in relevant sections
- **Context Awareness**: Considers document structure and section hierarchy

### 4. Keyword Extraction Strategy
We implement domain-aware keyword extraction:
- **Role-based Keywords**: Different keyword sets for researchers, students, analysts
- **Task-specific Terms**: Extract action-oriented keywords from job descriptions
- **Dynamic Adaptation**: System adapts to various domains (academic, business, technical)

### 5. Output Generation
- **Top Sections**: Identifies 5 most relevant document sections with importance ranking
- **Subsection Analysis**: Provides detailed analysis of top content chunks
- **Metadata Tracking**: Maintains complete audit trail of processing

## Technical Optimizations
- **Efficient Processing**: Batch encoding for improved performance
- **Memory Management**: Streaming processing for large document collections
- **Model Caching**: Pre-loads models in Docker container for faster execution
- **Error Handling**: Robust error handling for various PDF formats and edge cases

## Scalability Features
- **Generic Design**: Works across diverse document types and domains
- **Configurable Ranking**: Easily adjustable weights for different use cases
- **Modular Architecture**: Clean separation of concerns for maintainability

This approach ensures high relevance scores by combining semantic understanding with domain expertise, making it effective across the diverse test cases specified in the challenge.