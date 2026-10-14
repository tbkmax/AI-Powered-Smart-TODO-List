# main.py
import os
import json
import sqlite3
import uuid
import httpx
from contextlib import asynccontextmanager, contextmanager
from typing import Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:14b")
TODO_DB_PATH = os.getenv("TODO_DB_PATH", "todos.db")

@contextmanager
def get_db_connection():
    connection = sqlite3.connect(TODO_DB_PATH)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
    finally:
        connection.close()

def init_database():
    with get_db_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS todos (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                done INTEGER NOT NULL DEFAULT 0,
                due_at TEXT
            )
            """
        )
        connection.commit()

init_database()

def todo_from_row(row):
    return {
        "id": row["id"],
        "text": row["text"],
        "done": bool(row["done"]),
        "due_at": row["due_at"],
    }

def list_todos():
    with get_db_connection() as connection:
        rows = connection.execute(
            "SELECT id, text, done, due_at FROM todos ORDER BY done ASC, due_at IS NULL ASC, due_at ASC"
        ).fetchall()
    return [todo_from_row(row) for row in rows]

def create_todo_record(text: str, due_at: Optional[str] = None):
    text = text.strip()
    if not text:
        raise ValueError("Todo text cannot be empty")
    todo = {"id": str(uuid.uuid4()), "text": text, "done": False, "due_at": due_at}
    with get_db_connection() as connection:
        connection.execute(
            "INSERT INTO todos (id, text, done, due_at) VALUES (?, ?, ?, ?)",
            (todo["id"], todo["text"], 0, todo["due_at"]),
        )
        connection.commit()
    return todo

def update_todo_record(todo_id: str, text: Optional[str] = None,
                       done: Optional[bool] = None, due_at: Optional[str] = None,
                       due_at_provided: bool = False):
    with get_db_connection() as connection:
        row = connection.execute(
            "SELECT id, text, done, due_at FROM todos WHERE id = ?", (todo_id,)
        ).fetchone()
        if row is None:
            return None
        current = todo_from_row(row)
        if text is not None:
            text = text.strip()
            if not text:
                raise ValueError("Todo text cannot be empty")
            current["text"] = text
        if done is not None:
            current["done"] = done
        if due_at is not None or due_at_provided:
            current["due_at"] = due_at or None
        connection.execute(
            "UPDATE todos SET text = ?, done = ?, due_at = ? WHERE id = ?",
            (current["text"], int(current["done"]), current["due_at"], todo_id),
        )
        connection.commit()
    return current

def delete_todo_record(todo_id: str):
    with get_db_connection() as connection:
        cursor = connection.execute("DELETE FROM todos WHERE id = ?", (todo_id,))
        connection.commit()
    return cursor.rowcount > 0

def execute_agent_tool(name, arguments):
    try:
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        if name == "list_todos":
            return {"todos": list_todos()}
        if name == "create_todo":
            return create_todo_record(arguments.get("text", ""), arguments.get("due_at"))
        if name == "update_todo":
            todo = update_todo_record(
                arguments.get("id", ""),
                arguments.get("text"),
                arguments.get("done"),
                arguments.get("due_at"),
                "due_at" in arguments,
            )
            return todo or {"error": "Todo not found"}
        if name == "delete_todo":
            deleted = delete_todo_record(arguments.get("id", ""))
            return {"deleted": deleted, "id": arguments.get("id", "")}
        return {"error": f"Unknown tool: {name}"}
    except (TypeError, ValueError, sqlite3.Error) as error:
        return {"error": str(error)}

AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_todos",
            "description": "List all current TODO items and their completion and due-time status.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_todo",
            "description": "Create a new TODO item. Use an ISO-8601 local datetime for due_at when one was provided.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The task text."},
                    "due_at": {"type": ["string", "null"], "description": "Optional due date and time."},
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_todo",
            "description": "Modify an existing TODO item by its exact id. Only supplied fields are changed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "The TODO id."},
                    "text": {"type": "string", "description": "Optional replacement task text."},
                    "done": {"type": "boolean", "description": "Optional completion state."},
                    "due_at": {"type": ["string", "null"], "description": "Optional replacement due date and time."},
                },
                "required": ["id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_todo",
            "description": "Delete an existing TODO item by its exact id.",
            "parameters": {
                "type": "object",
                "properties": {"id": {"type": "string", "description": "The TODO id."}},
                "required": ["id"],
            },
        },
    },
]

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup and shutdown events. 
    Enforces the assumption that Ollama is running and qwen3:14b is available.
    """
    init_database()
    print("Checking Ollama connection and model availability...")
    try:
        async with httpx.AsyncClient() as client:
            # 1. Check if Ollama is reachable
            res = await client.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5.0)
            res.raise_for_status()
            
            # 2. Check if the specific model is pulled/available
            models = res.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            
            # Ollama sometimes appends ':latest', so we check if our target is a substring
            if not any(OLLAMA_MODEL in name for name in model_names):
                raise RuntimeError(
                    f"Model '{OLLAMA_MODEL}' not found in Ollama. "
                    f"Please run 'ollama pull {OLLAMA_MODEL}' or 'ollama run {OLLAMA_MODEL}' first."
                )
                
        print(f"Successfully connected to Ollama. Model '{OLLAMA_MODEL}' is ready.")
    except httpx.ConnectError:
        raise RuntimeError(
            f"Cannot connect to Ollama at {OLLAMA_BASE_URL}. "
            "Please ensure Ollama is installed and running."
        )
    except Exception as e:
        # Re-raise RuntimeError, wrap others
        if isinstance(e, RuntimeError):
            raise e
        raise RuntimeError(f"Failed to verify Ollama setup: {str(e)}")

    yield

app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory="templates")

# --- Pydantic Models ---
class TodoCreate(BaseModel):
    text: str
    due_at: Optional[str] = None

class TodoUpdate(BaseModel):
    text: Optional[str] = None
    done: Optional[bool] = None
    due_at: Optional[str] = None

class ChatRequest(BaseModel):
    message: str

# --- Routes ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/todos")
async def get_todos():
    return list_todos()

@app.post("/api/todos")
async def create_todo(todo: TodoCreate):
    try:
        return create_todo_record(todo.text, todo.due_at)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))

@app.patch("/api/todos/{todo_id}")
async def update_todo(todo_id: str, update: TodoUpdate):
    try:
        todo = update_todo_record(todo_id, update.text, update.done, update.due_at)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    if todo is None:
        raise HTTPException(status_code=404, detail="Todo not found")
    return todo

@app.delete("/api/todos/{todo_id}")
async def delete_todo(todo_id: str):
    if not delete_todo_record(todo_id):
        raise HTTPException(status_code=404, detail="Todo not found")
    return {"status": "deleted", "id": todo_id}

@app.post("/api/chat")
async def chat_with_agent(chat_req: ChatRequest):
    todos_context = json.dumps(list_todos())

    # --- PROMPT ENGINEERING ---
    # Pattern 1: Persona
    # Pattern 2: Few-Shot
    system_prompt = f"""You are an expert productivity coach and task management assistant (Persona). 
Your goal is to help the user prioritize, break down, and manage their tasks effectively. Be encouraging, concise, and actionable.

Here are examples of how you should respond (Few-Shot):
User TODOs: ["Buy groceries", "Finish report", "Call mom"]
User Message: "What should I do first?"
Assistant: "I recommend starting with 'Finish report' if it's time-sensitive, as deep work is best done early. After that, 'Call mom' can be a nice mental break before heading out to 'Buy groceries'."

User TODOs: ["Learn Python", "Build app"]
User Message: "I feel overwhelmed."
Assistant: "It's completely normal to feel overwhelmed! Let's break 'Learn Python' into smaller steps, like 'Complete Chapter 1 today'. Small wins build momentum!"

Current User TODOs (use the tools to make changes when requested): {todos_context}
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": chat_req.message}
    ]

    try:
        async with httpx.AsyncClient() as client:
            todo_changes = []
            # A bulk request such as "delete all" may need one list call,
            # several delete calls, and one final response.
            for _ in range(20):
                response = await client.post(
                    f"{OLLAMA_BASE_URL}/api/chat",
                    json={
                        "model": OLLAMA_MODEL,
                        "messages": messages,
                        "tools": AGENT_TOOLS,
                        "stream": False,
                    },
                    timeout=60.0,
                )
                response.raise_for_status()
                message = response.json()["message"]
                tool_calls = message.get("tool_calls", [])
                if not tool_calls:
                    return {
                        "reply": message.get("content", ""),
                        "todo_changes": todo_changes,
                    }

                messages.append(message)
                for tool_call in tool_calls:
                    function = tool_call.get("function", {})
                    result = execute_agent_tool(
                        function.get("name", ""),
                        function.get("arguments", {}),
                    )
                    if function.get("name", "") in {
                        "create_todo", "update_todo", "delete_todo"
                    }:
                        todo_changes.append({
                            "action": function.get("name", ""),
                            "todo": result,
                        })
                    messages.append({
                        "role": "tool",
                        "content": json.dumps(result),
                    })
            raise HTTPException(
                status_code=500,
                detail="AI agent exceeded the maximum number of TODO operations",
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error communicating with AI: {str(e)}")