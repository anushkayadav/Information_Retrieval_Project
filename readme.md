# Personalized Text Generation Project

This project focuses on enhancing personalized text generation using Large Language Models (LLMs) and stylometric analysis. It includes notebooks for two datasets and additional scripts for specific functionalities like using the Contriever model.

## Project Structure

- `News_Headline_Generation.ipynb`: Notebook for the News Headline Generation dataset.
- `Scholarly_Title_Generation.ipynb`: Notebook for the Scholarly Title Generation dataset.
- `contriever.py`: Python script for obtaining results from the Contriever model.

## Getting Started

### Prerequisites

- Google Colab or a Jupyter Notebook environment.
- Access to a machine with a GPU (for running summarizer models efficiently).

### Installation

1. **Clone the Repository**: Clone this repository to your local machine or open it in Google Colab.

    ```bash
    git clone [https://github.com/anushkayadav/Information_Retrieval_Project.git]
    ```

2. **Open the Notebooks**: The notebooks can be opened in Google Colab or any Jupyter Notebook environment. All necessary installations and requirements are included within the notebooks.

### Generating the PALM API Key

To use the PALM API in this project, you will need to generate an API key. Follow these steps:

1. Visit the PALM API website and log in or create an account.
2. Navigate to the API section and request an API key.
3. Once you have the API key, insert it into the designated section in the notebooks.

### Running the Contriever Script

The `contriever.py` script is used to get results from the Contriever model. These results are then used in the notebooks for further processing. Run the script separately before starting with the notebooks.

```bash
python contriever.py
