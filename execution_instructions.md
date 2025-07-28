# **Execution Instructions**

This document provides the necessary steps to build the Docker image and run the solution for both **Round 1A (Understand Your Document)** and **Round 1B (Persona-Driven Document Intelligence)**. The entire application is containerized in a single image to ensure a consistent and reproducible environment.

## **Prerequisites**

*   [Docker](https://www.docker.com/products/docker-desktop/) must be installed and running on your system.

## **Directory and File Setup**

Before running the container, prepare a local directory. The application is designed to read from an `input` subdirectory and write all analysis results to an `output` subdirectory.

1.  Create a root project folder (e.g., `my_project`).
2.  Inside `my_project`, create two subdirectories: `input` and `output`.
3.  Place all your Python source code (`main.py`, `hierarchy_builder.py`, etc.), the `Dockerfile`, and `requirements.txt` directly inside the `my_project` directory.

```bash
# Create the project directory and subdirectories
mkdir -p my_project/input
mkdir -p my_project/output
cd my_project
```

The final directory structure before execution should look like this:

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
└── output/
```

## **Step 1: Build the Docker Image**

Navigate to the project's root directory (`my_project`) in your terminal. Run the following command to build the single Docker image that serves both parts of the challenge. The image will be tagged as `doc-analyzer`.

```bash
docker build -t doc-analyzer .
```

This command builds the image, packages all dependencies, and prepares it for execution.

## **Step 2: Running the Solution**

The same `docker run` command is used for both Round 1A and Round 1B. The script inside the container (`main.py`) is designed to automatically detect the type of task by reading the JSON files in the `/app/input` directory.

### **Part 1A: Understand Your Document**

For Round 1A, the task is to extract the structure (title and headings) from individual PDF files.

**1. Prepare Input Files:**
*   Place the PDF files you want to process (e.g., `document1.pdf`, `document2.pdf`) inside the `input` directory.

**2. Execute the Container:**
*   From the `my_project` directory, run the following command:

```bash
docker run --rm \
  -v "$(pwd)/input:/app/input" \
  -v "$(pwd)/output:/app/output" \
  doc-analyzer
```

**3. Verify Output:**
*   The container will process each PDF and generate a corresponding JSON file in the `output` directory. For an input file named `document1.pdf`, the output will be `document1.json`, containing the extracted title and hierarchical outline.

### **Part 1B: Persona-Driven Document Intelligence**

For Round 1B, the task is to analyze a collection of documents based on a persona and a specific job-to-be-done, as defined in a configuration file.

**1. Prepare Input Files:**
*   Place the collection of PDF documents (e.g., `report1.pdf`, `study.pdf`) inside the `input` directory.
*   Also in the `input` directory, create a JSON configuration file (e.g., `config.json`) that lists the documents to be analyzed and defines the `persona` and `job_to_be_done`.

**2. Execute the Container:**
*   From the `my_project` directory, run the same command as before:

```bash
docker run --rm \
  -v "$(pwd)/input:/app/input" \
  -v "$(pwd)/output:/app/output" \
  doc-analyzer
```

**3. Verify Output:**
*   The container will process the instructions from `config.json`, analyze the specified PDFs, and generate a single, consolidated analysis file in the `output` directory named `config_analysis.json`. This file will contain the extracted sections and subsections most relevant to the persona and task.

***
**Note on Operating Systems**: The `$(pwd)` syntax is standard for macOS and Linux terminals. For **Windows Command Prompt**, replace `$(pwd)` with `%cd%`. For **Windows PowerShell**, `$(pwd)` should work as is.