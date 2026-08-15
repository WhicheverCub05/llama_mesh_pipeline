# LLM Pipeline

This project was originally made to automate the production of 3D models by running a model which outputs the text of an OBJ file describing vertices, faces and edges. The 3D generation model is [LLaMA-Mesh](https://research.nvidia.com/labs/toronto-ai/LLaMA-Mesh/), a finetuned LLM for 3D generation — see the [paper](https://arxiv.org/html/2411.09595v1), the [finetuned model weights](https://huggingface.co/Zhengyi/LLaMA-Mesh), and the [demo](https://huggingface.co/spaces/Zhengyi/LLaMA-Mesh). I hosted model weights using llama.cpp.

A Python-based batch LLM processing pipeline that chains multiple stages together. Each stage wraps input text with custom prompts, generates outputs via Ollama or vLLM, and saves results as CSV files. Outputs from one stage can feed into the next, enabling multi-stage processing pipelines.

##  Overview

The LLM Pipeline is designed for multi-stage text processing. It supports both Ollama and vLLM backends through a unified interface. You can define a sequence of operations (e.g., Summarization $\rightarrow$ Translation $\to$ Formatting) via a single JSON configuration file.

### Core Workflow
1.  **Input**: Load text from files or the output of a previous stage.
2.  **Wrap**: Inject text into templates using a customizable `insert_key`.
3.  **Generate**: Send the wrapped prompts to Ollama or vLLM for inference.
4.  **Filter**: Clean model-specific artifacts (e.g., stripping prefixes from Gemma 3).
5.  **Output**: Save structured results (index, prompt, raw response, filtered response) to CSV.
6.  **Chain**: Use the filtered outputs as the input for the subsequent stage.

## Pipeline Functions

The following pipeline functions are fully configurable via `pipeline_config.json`:

| Function | Config Keys | Description |
| :--- | :--- | :--- |
| Input loading | `input.type` (`file`), `input.path`, `input.delimiter` | Load raw input items from a text file. |
| Prompt wrapping | `prompt.template`, `prompt.insert_key` | Inject each input into a template at the `insert_key` placeholder. |
| Wrapped prompt saving | `output.wrapped_file` | Save the wrapped prompts to a text file for inspection. |
| Intermediate output | `output.intermediate_file` | Save this stage's filtered responses as the next stage's input. |
| LLM generation | `model`, `model_params` (`temperature`, `top_p`, `num_predict`, `parallel`, `max_workers`, `stream`, `backend`) | Send wrapped prompts to Ollama or vLLM and generate responses. |
| Chained input | `input.type` (`previous_csv`), `input.delimiter` | Feed the previous stage's CSV output into the next stage. |

##  Project Structure

### Package Organization
```
src/
  llm_pipeline/
    __init__.py
    cli/
      __init__.py
      main.py          # Unified CLI entry point
    core/
      __init__.py
      client.py        # Ollama API client with retry logic
      output.py        # CSV writing functionality
      pipeline.py      # Multi-stage pipeline orchestration
    utils/
      __init__.py
      csv_utils.py     # CSV manipulation utilities
      obj_utils.py     # OBJ file generation utilities
      text_filters.py  # Model-specific output filters
      visual.py        # 3D OBJ visualization
      wrap_prompts.py  # Prompt wrapping utilities
    config/
      __init__.py
      schema.py        # JSON schema validation
```

### Core Components
| Module | Purpose |
| :--- | :--- |
| `cli.main` | Unified CLI and single-stage pipeline execution logic |
| `core.pipeline` | Orchestrates multi-stage pipelines defined in JSON configs |
| `core.llm_client` | Backend-agnostic LLM client (supports Ollama and vLLM) |
| `core.output` | Manages CSV reading and writing for pipeline stages |
| `utils.wrap_prompts` | Utilities for loading inputs and applying prompt templates |
| `utils.csv_utils` | Utilities for CSV manipulation (extract columns to text) |
| `utils.obj_utils` | Utilities for converting CSV data to 3D OBJ files |
| `utils.visual` | 3D OBJ file visualization and rendering |
| `utils.text_filters` | Contains model-specific cleaning logic (e.g., for Gemma, Llama) |
| `config.schema` | Validates the structure of pipeline JSON configurations |

### Directory Layout (Typical)
*   `prompts/`: Contains input text files and templates.
*   `prompts/wrapped/`: Stores the text after templates have been applied.
*   `prompts/results/`: Stores the final CSV outputs from each stage.
*   `.env`: Stores environment variables like `OLLAMA_HOST`.

## Configuration

Pipelines are controlled via JSON configuration files. A stage configuration includes:
*   `input`: Type (`file` or `previous_csv`) and path.
*   `prompt`: The `template` and the `insert_key` (e.g., `<>`).
*   `model`: The Ollama model to use (e.g., `gemma3:1b`).
*   `output`: Paths for the wrapped text file and the resulting CSV.

## Usage

### Prerequisites
*   [Ollama](https://ollama.com/) installed and running, OR
*   [vLLM](https://docs.vllm.ai/) server running
*   Python 3.10+ installed.

### Running a Multi-Stage Pipeline (Ollama)
```bash
python main.py pipeline --config ollama_pipeline_config.json
# Resume from checkpoint
python main.py pipeline --config ollama_pipeline_config.json --resume
```

### Running a Multi-Stage Pipeline (vLLM)
```bash
python main.py pipeline --config vllm_pipeline_config.json
```

### Running a Single Stage (Ollama)
```bash
python main.py run -i prompts.txt -o results.csv -m llama3.2:1b
```

### Running a Single Stage (vLLM)
```bash
python main.py run -i prompts.txt -o results.csv -m meta-llama/Llama-3.1-8B --backend vllm --host http://localhost:8000
```

### Converting CSV Columns to Text Files
```bash
python main.py csv_to_txt --input_csv results.csv --output_txt extracted.txt --column filtered_response
```

### Converting CSV Columns to OBJ Files
```bash
python main.py csv_to_obj --csv_file results.csv --obj_folder objs --column_index 3
```

### Visualizing OBJ Files
```bash
python main.py visual --input_dir objs --output_dir images
```

### Standalone Prompt Wrapping
If you only need to prepare a prompt file without running inference:
```bash
python wrap_prompts.py run -i input.txt -o wrapped.txt -p "Template: <>" -k "<>"
```

## Environment Variables

The following variables can be set in a `.env` file:

| Variable | Description | Default | vLLM Value |
|----------|-------------|---------|------------|
| `LLM_HOST` | API endpoint URL | `http://localhost:11434` | `http://localhost:8000` |
| `MODEL` | Default model name | `llama3.2:1b` | Any loaded vLLM model |
| `CONTEXT_WINDOW` | Token limit for context | `4096` | `max_model_len` |

> **Note**: For Ollama, you can also use `OLLAMA_HOST` (legacy support).

## Configuration Examples

See the `examples/` directory for complete pipeline configurations:
- `ollama_pipeline_config.json` - Example using Ollama backend
- `vllm_pipeline_config.json` - Example using vLLM backend

### Sample Generated Outputs

The `examples/generated/` directory contains a couple of sample pipeline outputs (OBJ files and their renders) so you can see what the pipeline produces:
- `30rio_final_13.obj` / `30rio_final_13_img.jpg` - Outdoor object generated by LLaMA-Mesh
- `10pc_final_2.obj` / `10pc_final_2_img.jpg` - PC component generated by LLaMA-Mesh

> **Note**: For most objects except furniture, LLaMA-Mesh was not able to generate a useful output. We are now experimenting with hosting [Hunyuan3D-2](https://huggingface.co/tencent/Hunyuan3D-2) instead.

## Checkpointing
The pipeline features built-in checkpointing. It saves progress to `.pipeline_checkpoint.json` after every successful generation. If a process is interrupted, you can resume exactly where you left off:
```bash
python main.py pipeline --config pipeline_config.json --resume
```
