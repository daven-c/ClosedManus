# ClosedManus

An AI-powered web automation assistant built with FastAPI, Playwright, and Google Gemini.

## Setup and Installation

1. Install Python 3.9+ if not already installed
2. Clone this repository
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4. Set up Playwright:

    ```bash
    python setup_playwright.py
    ```

    Alternatively, install manually:

    ```bash
    playwright install --with-deps
    ```

5. Create a `.env` file with your Google API key:
    ```
    GOOGLE_API_KEY=your_google_api_key_here
    ```

## Running the Application

Start the web server:

```bash
python main.py
```

Then open your browser to http://localhost:8000

## Troubleshooting

If you encounter a `NotImplementedError` when starting the application, it's usually because Playwright's browser binaries are not properly installed. Run:

```bash
python setup_playwright.py
```

On Windows, you may need to run the terminal as Administrator if the installation fails due to permission issues.
