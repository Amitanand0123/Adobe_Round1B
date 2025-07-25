#### Methodology for Intelligent Document Analysis and Ranking

This system implements a sophisticated, multi-stage pipeline designed to intelligently analyze a collection of PDF documents, understand their structure, and extract the most relevant content tailored to a specific user persona and task. The approach moves beyond simple keyword matching by integrating structural analysis, content chunking, and a hybrid semantic ranking model.

**Phase 1: Document Ingestion and Structural Reconstruction**

The process begins with the `PDFProcessor`, which ingests raw PDF files. It uses the `pdfplumber` library to extract detailed information about every text block, including its content, font size, style, and precise location on the page. This granular data is crucial for the next step, where the `HierarchyBuilder` reconstructs the document's logical outline. By analyzing features like font size, boldness, and centrality, the builder identifies potential headings. It then employs a KMeans clustering algorithm on the font sizes of these candidates to programmatically classify them into distinct levels (e.g., H1, H2, H3), effectively rebuilding the document's table of contents even when one is not explicitly available.

**Phase 2: Intelligent Content Chunking**

With the document's structure established, the `AdvancedContentChunker` takes over. It intelligently associates all text blocks with their corresponding sections from the generated outline. Its primary role is to group related text into meaningful, coherent "chunks." This process is not arbitrary; it uses heuristics to identify logical breaks in content, such as large vertical gaps, changes in font style, or the start of bulleted lists. Furthermore, it filters out low-value content by assessing "content density" and ensuring chunks meet a minimum length, thus eliminating generic phrases and focusing on substantive information.

**Phase 3: Multi-faceted Semantic Ranking**

The core of the system is the `SemanticRanker`. Instead of relying on a single metric, it calculates a comprehensive relevance score from multiple weighted factors. The primary component is semantic similarity, calculated using a TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer and cosine similarity. This measures the contextual relevance between user queries and the content chunks.

This semantic score is then blended with several other critical signals:
*   **Persona Alignment:** The score is boosted if the chunk contains terminology specific to the user's role (e.g., "methodology" for a researcher, "revenue" for a business analyst).
*   **Content Specificity:** Content with actionable details—such as lists, tables, figures, or numerical data—is prioritized.
*   **Section Importance:** Pre-defined rules assign higher value to sections like "Conclusion" or "Results" over "Introduction."
*   **Keyword Matching:** Direct matches for keywords from the persona and task descriptions contribute to the score.

By combining these elements, the ranker produces a holistic score that reflects not just what a chunk is about, but also how useful and appropriate it is for the user's specific need. The final ranked list of chunks is then used to generate the distilled, actionable output.

***

### `Dockerfile` and Execution Instructions

Here is the provided `Dockerfile` and the instructions to build the image and run the analysis.

#### Dockerfile

```dockerfile
# Dockerfile - Lightweight Version

# Use a slim Python image as the base
FROM --platform=linux/amd64 python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for pdf2image and pdfplumber
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all your source code into the container
COPY . .

# Create the output directory
RUN mkdir -p /app/output

# Set the command to run the application when the container starts
CMD ["python", "main.py"]
```

#### Execution Instructions

To run the document analysis pipeline using Docker, follow these steps.

**1. Prerequisites**
*   Ensure you have Docker installed and running on your system.

**2. Directory Setup**
Create a local directory structure for your input and output files. The application reads from an `input` directory and writes to an `output` directory.

```bash
mkdir -p my_project/input
mkdir -p my_project/output
cd my_project
```

**3. Place Your Files**
*   **Source Code:** Place `main.py`, `semantic_ranker.py`, `pdf_processor.py`, `hierarchy_builder.py`, `content_chunker.py`, `Dockerfile`, and `requirements.txt` directly inside the `my_project` directory.
*   **Input Data:** Place your PDF documents (e.g., `document1.pdf`) and the JSON configuration file (e.g., `config.json`) inside the `my_project/input` directory. The JSON file must reference the PDF filenames correctly.

Your final directory structure should look like this:
```
my_project/
├── Dockerfile
├── main.py
├── semantic_ranker.py
├── pdf_processor.py
├── hierarchy_builder.py
├── content_chunker.py
├── requirements.txt
├── input/
│   ├── config.json
│   └── document1.pdf
└── output/
```

**4. Build the Docker Image**
From within the `my_project` directory, run the following command to build the Docker image. We will tag it `doc-analyzer`.

```bash
docker build -t doc-analyzer .
```

**5. Run the Analysis**
Execute the following command to run the container. This command mounts your local `input` directory to `/app/input` in the container and your local `output` directory to `/app/output`. The application will start automatically.

```bash
docker run --rm \
  -v "$(pwd)/input:/app/input" \
  -v "$(pwd)/output:/app/output" \
  doc-analyzer
```
*Note for Windows users:* You may need to replace `$(pwd)` with the full path to your project directory, like `C:\path\to\my_project`.

**6. Check the Results**
Once the script finishes execution, the analysis results will be saved as a JSON file (e.g., `config_analysis.json`) in your local `my_project/output` directory.