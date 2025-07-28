```markdown
### Execution Instructions

To build the Docker image and run the document analysis pipeline, please follow the steps below.

**1. Prerequisites**

*   Ensure Docker is installed and running on your system.

**2. Directory Setup**

Create a local directory structure for the input and output files. The application is designed to read from an `input` directory and write the analysis results to an `output` directory.

```bash
# Create the project directory and subdirectories
mkdir -p my_project/input
mkdir -p my_project/output
cd my_project
```

**3. Place Your Files**

*   **Source Code:** Place `main.py`, `semantic_ranker.py`, `pdf_processor.py`, `hierarchy_builder.py`, `content_chunker.py`, `Dockerfile`, and `requirements.txt` directly inside the `my_project` directory.
*   **Input Data:** Place the PDF documents and the JSON configuration file (e.g., `config.json`) inside the `my_project/input` directory. The JSON file must reference the PDF filenames correctly.

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

From within the `my_project` directory, execute the following command to build the Docker image. The image will be tagged as `doc-analyzer`.

```bash
docker build -t doc-analyzer .```

**5. Run the Analysis**

Execute the following command to run the container. This command mounts your local `input` directory to `/app/input` and your `output` directory to `/app/output` inside the container. The application will start automatically and process all JSON configurations found in the input directory.

```bash
docker run --rm \
  -v "$(pwd)/input:/app/input" \
  -v "$(pwd)/output:/app/output" \
  doc-analyzer
```
*(Note for Windows users: You may need to replace `$(pwd)` with the full path to your project directory, such as `C:\path\to\my_project`)*

**6. Check the Results**

Once the script finishes execution, the analysis results will be saved as a JSON file (e.g., `config_analysis.json`) in your local `my_project/output` directory.
```