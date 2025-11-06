# llm-business-pain-point-finder

This project is a really simple use case that demonstrates how to use Large Language Models (LLMs) to identify business opportunities and propose AI solutions through prompt chaining. The workflow consists of three main steps:

1. **Business Area Selection**: The model is prompted to pick a promising business area for Agentic AI opportunities.
2. **Pain Point Identification**: The model is asked to identify a significant pain point within the selected business area.
3. **AI Solution Proposal**: The model is prompted to propose an Agentic AI solution for the identified pain point.

The project uses OpenAI's API and can be configured with your own API key via a `.env` file. See `src/main.py` for the implementation details.
