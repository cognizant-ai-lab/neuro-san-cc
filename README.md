# Neuro SAN Climate Change
Neuro SAN applied to Climate Change.

Answers questions about the Conference of the Parties (COP), the Kyoto Protocol (CMP) and the Paris Agreement (CMA) using the United Nations Framework Convention on Climate Change (UNFCCC) report documents. For instance, [ FCCC/PA/CMA/2024/17/Add.1](https://unfccc.int/documents/644937) from [CMA 6](https://unfccc.int/event/cma-6).

For more details about Neuro SAN, please check the [Neuro SAN Studio](https://github.com/cognizant-ai-lab/neuro-san-studio) repository.

## Getting started

### Installation

Clone the repo:

```bash
git clone https://github.com/cognizant-ai-lab/neuro-san-cc
```

Go to dir:

```bash
cd neuro-san-cc
```

Ensure you have a supported version of python (e.g. 3.12 or 3.13):

```bash
python --version
```

Create a dedicated Python virtual environment:

```bash
python -m venv venv
```

Source it:

* For Windows:

  ```cmd
  .\venv\Scripts\activate.bat && set PYTHONPATH=%CD%
  ```

* For Mac:

  ```bash
  source venv/bin/activate && export PYTHONPATH=`pwd`
  ```

Install the requirements:

```bash
pip install -r requirements.txt
```

**IMPORTANT**: By default the server relies on OpenAI's `gpt-4o` model. Set the OpenAI API key, and add it to your shell
configuration so it's available in future sessions.

You can get your OpenAI API key from <https://platform.openai.com/signup>. After signing up, create a new API key in the
API keys section in your profile.

**NOTE**: Replace `XXX` with your actual OpenAI API key.  
**NOTE**: This is OS dependent.

* For macOS and Linux:

  ```bash
  export OPENAI_API_KEY="XXX" && echo 'export OPENAI_API_KEY="XXX"' >> ~/.zshrc
  ```

<!-- pyml disable commands-show-output -->
* For Windows:
    * On Command Prompt:

    ```cmd
    set OPENAI_API_KEY=XXX
    ```

    * On PowerShell:

    ```powershell
    $env:OPENAI_API_KEY="XXX"
    ```

<!-- pyml enable commands-show-output -->

Other providers such as Anthropic, AzureOpenAI, Ollama and more are supported too but will require proper setup.
Look at the `.env.example` file to set up environment variables for specific use-cases.

For testing the API keys, please refer to this [documentation](./docs/api_key.md)

---

### Run

Neuro SAN Studio provides a user-friendly environment to interact with agent networks.

1. Start the server and client with a single command, from the project root directory:

    ```bash
    python -m run
    ```

2. Navigate to [http://localhost:4173/](http://localhost:4173/) to access the UI.
3. (Optional) Check the logs:
   * For the server logs: `logs/server.log`
   * For the client logs: `logs/nsflow.log`
   * For the agents logs: `logs/thinking_dir/*`

Use the `--help` option to see the various config options for the `run` command:

```bash
python -m run --help
```

### Using the agent networks

Select the `paris_agreement` network for instance, and ask it questions like:

> What are the main differences between the Sharm el-Sheikh decision and the Baku decision on Article 6.2?

If it has a hard time finding the Baku documentation, help it a bit:

> Baku is CMA.6

And you should get a comprehensive answer.