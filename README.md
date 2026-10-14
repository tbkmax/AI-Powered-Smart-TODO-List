# AI-Powered Smart TODO List

## 1. Problem & Market Need
Many people struggle with task management, not because they don't know *what* to do, but because they don't know *how* to prioritize or break down overwhelming tasks. Traditional TODO lists are passive; they just store text. 

**The Solution:** An AI-powered TODO list that acts as an active productivity coach. By analyzing the user's current task list, the AI provides contextual advice, prioritization strategies, and encouragement, turning a static list into an interactive planning tool.

## 2. Application Design & Interaction Flow
- **UI Layout:** A single-page web application. The center/left focuses on the core TODO list (input, view, delete). The right side features a dedicated chat box.
- **Interaction Flow:** 
  1. User adds tasks to the TODO list.
  2. User opens the chat box and asks for advice (e.g., "What should I tackle first?" or "I'm overwhelmed, help me plan").
  3. The backend packages the current TODO list context and sends it to the local LLM.
  4. The AI responds with actionable, personalized advice directly in the chat interface.

## 3. Prompt Engineering Techniques
To ensure the AI provides high-quality, relevant, and consistent advice, two specific prompt engineering patterns were implemented in `main.py`:

1. **Persona Pattern:** 
   - *Implementation:* The system prompt explicitly defines the AI's role: *"You are an expert productivity coach and task management assistant."*
   - *Why it's used:* This sets the behavioral boundaries and tone. It prevents the AI from acting like a generic search engine and forces it to adopt an encouraging, strategic, and task-oriented mindset.
2. **Few-Shot Prompting:** 
   - *Implementation:* The prompt includes concrete examples of user inputs and ideal AI responses before asking the AI to process the actual request.
   - *Why it's used:* LLMs perform significantly better when given examples of the expected output format and reasoning style. It shows the model exactly how to balance empathy with actionable task breakdowns.

## 4. Setup and Installation

### Prerequisites
- Python 3.12
- **Ollama** installed and running on your machine.

### Step 1: Pull the AI Model
This project strictly requires the `GPT-OSS 20B ` model. Open your terminal and run:
```bash
ollama run gpt-oss-safeguard:20b
```

### Step 2: Start server

```bash
cp .env.example .env
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

```bash
uvicorn main:app --reload
```

Go to: `http://127.0.0.1:8000 `

