# Crisis as Catalyst: Evaluating Ethical Consistency and Cooperation in LLMs under High-Stakes Scenarios

## Abstract

As large language models (LLMs) increasingly integrate into critical decision-making systems, understanding their behavioral consistency and alignment with human values in high-stakes scenarios becomes paramount. This research systematically investigates LLM behavioral patterns across multiple dimensions—cooperative intent, resource distribution, and moral reasoning—under simulated emergency conditions.

The study employs two core experimental paradigms: (1) **Unit Assignment** scenarios that examine inter-organizational cooperation and resource allocation strategies during crisis response, and (2) **Trolley Problem** variants that assess moral reasoning consistency across ethical dilemmas. Our multi-model approach evaluates behavioral patterns across different LLM architectures and training paradigms.

## Research Objectives

- **Behavioral Consistency Analysis**: Evaluate the stability of LLM decision-making patterns across repeated trials and varying contexts
- **Cooperative Intent Assessment**: Measure LLMs' propensity for collaborative versus competitive strategies in resource allocation scenarios
- **Moral Reasoning Evaluation**: Analyze consistency in ethical decision-making across classical moral dilemma variations
- **Cross-Model Comparison**: Compare behavioral patterns across different LLM architectures and training approaches

## Experimental Design

can't show the details here.

## Project Structure

```
Crisis as Catalyst/
├── api.py                     # Unified API configuration and model management
├── data/                      # Raw experimental data
│   ├── assignunit/           # Unit assignment experiment data
│   └── trolley/              # Trolley problem experiment data
├── result/                   # Analysis outputs and visualizations
│   ├── assignunit/           # Unit assignment results
│   └── trolley/              # Trolley problem results
├── src/                      # Core experimental modules
│   ├── assignunit/           # Unit assignment experiment
│   │   ├── assignunit.py     # Main experiment script
│   │   ├── assignunit_simp.py # Simplified version
│   │   └── analyze/          # Analysis scripts
│   ├── trolley/              # Trolley problem experiment
│   │   └── trolley.py        # Main experiment script
│   ├── visualize_assignunit.py # Unit assignment visualization
│   └── visualize_trolley.py    # Trolley problem visualization
└── readme.md                 # This file
```

## Installation and Setup

### Prerequisites

- Python 3.8 or higher
- API access to at least one supported LLM provider

### Environment Configuration

1. **Create environment file**: Create a `.env` file in the project root directory:

```bash
# SiliconFlow API
SiliconFlow_API_KEY=your_siliconflow_key

# DeepInfra API  
DeepInfra_API_KEY=your_deepinfra_key

# Azure OpenAI API
AZURE_API_KEY=your_azure_key
AZURE_API_BASE=your_azure_endpoint
AZURE_API_VERSION=your_api_version
AZURE_DEPLOYMENT_GPT4O=your_gpt4o_deployment
AZURE_DEPLOYMENT_GPT4O_MINI=your_gpt4o_mini_deployment

# Google API (if using)
Google_API_KEY=your_google_key
```

2. **Install dependencies**:

```bash
pip install openai langchain python-dotenv matplotlib pandas numpy seaborn
```

## Usage Instructions

### Running Experiments

Execute experiments from the project root directory:

#### Unit Assignment Experiment

```bash
python src/assignunit/assignunit.py --model <model_name> --num <test_count> --output <output_path>
```

**Parameters**:
- `--model`: Model identifier (e.g., 'gpt-4o', 'deepseek-v2.5', 'qwen-32b')
- `--num`: Number of test iterations per model (default: 10)
- `--output`: Output CSV file path (default: 'assignunit/raw_responses_assignunit.csv')

**Example**:
```bash
python src/assignunit/assignunit.py --model gpt-4o --num 20 --output data/assignunit/gpt4o_responses.csv
```

#### Trolley Problem Experiment

```bash
python src/trolley/trolley.py --model <model_name> --num <response_count> --output <output_path>
```

**Parameters**:
- `--model`: Model identifier
- `--num`: Number of response sets to generate (default: 1)
- `--output`: Output CSV file path (default: 'raw_responses_trolley.csv')

**Example**:
```bash
python src/trolley/trolley.py --model deepseek-v2.5 --num 50 --output data/trolley/deepseek_responses.csv
```

### Data Analysis and Visualization

Run them directly.

## Data Output Structure

### Unit Assignment Data

**CSV Structure**: `[model, scenario, Day1, Day8-Day18, test_id]`
- Each row represents one complete test iteration
- Day columns contain unit IDs (1-6: Blue Team, 7-12: Red Team)
- Scenarios A-D represent different familiarity/ECFI conditions

**Motivation Data**: JSON file containing qualitative decision rationales

### Trolley Problem Data

**CSV Structure**: `[model, Q1-Q13]`
- Each row represents one complete questionnaire response
- Columns Q1-Q13 contain binary choices (A/B) for each moral dilemma

## Contact and Support

For technical issues, experimental design questions, or collaboration inquiries, please contact the research team.
