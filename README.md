
# Synapse AI

Synapse AI is an autonomous software development platform that leverages AI agents to automate the entire software development lifecycle - from requirements analysis to deployment configuration.

## Overview

Synapse AI orchestrates a team of specialized AI agents that collaborate to build software projects based on natural language requirements. Each agent has specific expertise and responsibilities within the development process.

## Features

- **Multi-Agent Collaboration**: Utilizes 10 specialized agents including Requirements Analyst, Software Architect, Developer, Tester, and more
- **Event-Driven Architecture**: Agents communicate and coordinate through an event bus
- **End-to-End Development**: Handles the entire software lifecycle from requirements to deployment
- **Knowledge Graph**: Maintains a graph database of project artifacts and relationships
- **Automatic Documentation**: Generates comprehensive documentation for all aspects of the project

## Agents

- **Requirements Analyst**: Analyzes requirements and creates specifications
- **Software Architect**: Designs the overall system architecture
- **Tech Lead**: Creates project structure and development standards
- **Developer**: Implements code components
- **Tester**: Writes and runs tests for components
- **Code Reviewer**: Performs code quality reviews
- **UI Designer**: Creates UI prototypes
- **Technical Writer**: Creates project documentation
- **Project Manager**: Creates project plans and task lists
- **DevOps Engineer**: Creates deployment configurations

## Getting Started

### Prerequisites

- Python 3.11 or higher
- OpenAI API key

### Installation

1. Set up your environment variables:
   - Create a `.env` file with:
   ```
   OPENAI_API_KEY=your_openai_api_key
   PROJECT_ROOT=./output
   ```

2. Install required packages:
   ```
   pip install openai python-dotenv
   ```

### Usage

Run Synapse AI with a project requirement:

```bash
python synapse.py "create a web-based task management application with user authorization"
```

Optional: Specify an output directory:

```bash
python synapse.py "create a web API for a blog" "./my_project"
```

## Output

Synapse generates a complete project structure with:

- Requirements specification
- Architecture design
- Project structure
- Component implementations
- Tests
- Code reviews
- Documentation
- Deployment configuration

## Configuration

Configuration options in `.env`:

- `OPENAI_API_KEY`: Your OpenAI API key
- `PROJECT_ROOT`: Output directory (default: `./output`)
- `LLM_MODEL`: LLM model to use (default: `gpt-4`)
- `DEBUG`: Enable debug logging (default: `false`)
- `MAX_PARALLEL_TASKS`: Maximum parallel tasks (default: `3`)

## License

This project is available under the MIT License.

## Acknowledgements

- OpenAI for providing the LLM capabilities
- The Python community for the excellent libraries and tools
