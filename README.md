# neuro-san-cc
Neuro SAN applied to Climate Change

For more details about Neuro SAN, please check the [Neuro SAN Studio](https://github.com/cognizant-ai-lab/neuro-san-studio) repository.

## Getting started

### Installation

Clone the repo:

```bash
git clone https://github.com/cognizant-ai-lab/neuro-san-cc
```

Go to dir:

```bash
cd neuro-san-studio
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

There are multiple ways in which we can now use the neuro-san server with a client:

<!-- pyml disable-next-line line-length -->
#### Option 1: Using [`nsflow`](https://github.com/cognizant-ai-lab/nsflow) as a developer-oriented web client

If you want to use neuro-san with a FastAPI-based developer-oriented client, follow these steps:

* Start the server and client with a single command, from project root:

  ```bash
  python -m run
  ```

* As a default
    * Frontend will be available at: `http://127.0.0.1:4173`
    * The client and server logs will be saved to `logs/nsflow.log` and `logs/server.log` respectively.

* To see the various config options for this app, on terminal

  ```bash
  python -m run --help
  ```
