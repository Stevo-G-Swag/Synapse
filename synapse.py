import asyncio
import inspect
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Generic, List, Optional, Set, Tuple, Type, TypeVar, Union, cast

from openai import AsyncOpenAI

from dotenv import load_dotenv

# =============================================================================
# CONFIGURATION
# =============================================================================

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("synapse")


# Global configuration
class Config:
    project_root: str = os.getenv("PROJECT_ROOT", "./output")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4")
    llm_api_key: str = os.getenv("OPENAI_API_KEY", "")
    llm_base_url: str = os.getenv("OPENAI_API_BASE",
                                  "https://api.openai.com/v1")
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    max_parallel_tasks: int = int(os.getenv("MAX_PARALLEL_TASKS", "3"))
    telemetry_enabled: bool = os.getenv("TELEMETRY_ENABLED",
                                        "true").lower() == "true"

    @classmethod
    def ensure_directories(cls):
        """Ensure all required directories exist."""
        directories = [
            cls.project_root,
            f"{cls.project_root}/src",
            f"{cls.project_root}/docs",
            f"{cls.project_root}/tests",
            f"{cls.project_root}/artifacts",
            f"{cls.project_root}/design",
            f"{cls.project_root}/prototypes",
            f"{cls.project_root}/logs",
        ]
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)


# =============================================================================
# CORE COMPONENTS
# =============================================================================


class EventType(Enum):
    """Event types for the event bus."""
    REQUIREMENT_ADDED = "requirement_added"
    ARTIFACT_CREATED = "artifact_created"
    CODE_GENERATED = "code_generated"
    TEST_CREATED = "test_created"
    REVIEW_SUBMITTED = "review_submitted"
    ISSUE_FOUND = "issue_found"
    ISSUE_FIXED = "issue_fixed"
    SYSTEM_ERROR = "system_error"
    AGENT_STARTED = "agent_started"
    AGENT_COMPLETED = "agent_completed"
    USER_FEEDBACK = "user_feedback"
    MILESTONE_REACHED = "milestone_reached"
    ARCHITECTURE_UPDATED = "architecture_updated"
    PROTOTYPE_READY = "prototype_ready"
    VERIFICATION_FAILED = "verification_failed"
    VERIFICATION_PASSED = "verification_passed"
    EXECUTION_STARTED = "execution_started"
    EXECUTION_COMPLETED = "execution_completed"


@dataclass
class Event:
    """Event class for the event bus."""
    type: EventType
    data: Dict[str, Any]
    source: str
    timestamp: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))


class EventBus:
    """Event bus for decoupled communication between components."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EventBus, cls).__new__(cls)
            cls._instance._subscribers = {}
            cls._instance._history = []
        return cls._instance

    def subscribe(self, event_type: EventType,
                  callback: Callable[[Event], None]) -> None:
        """Subscribe to an event type."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(callback)

    def publish(self, event: Event) -> None:
        """Publish an event to all subscribers."""
        logger.debug(
            f"Event published: {event.type.value} from {event.source}")
        self._history.append(event)

        if event.type in self._subscribers:
            for callback in self._subscribers[event.type]:
                try:
                    callback(event)
                except Exception as e:
                    logger.error(f"Error in event callback: {e}")
                    if Config.debug:
                        traceback.print_exc()

    def get_history(self,
                    event_type: Optional[EventType] = None) -> List[Event]:
        """Get history of events, optionally filtered by type."""
        if event_type is None:
            return self._history.copy()
        return [event for event in self._history if event.type == event_type]


@dataclass
class Message:
    """Message class for communication between agents."""
    content: str
    sender: str
    receiver: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)

    def __str__(self) -> str:
        receiver = self.receiver or "broadcast"
        return f"[{self.sender} → {receiver}]: {self.content[:50]}..."


class Memory:
    """Semantic memory for agents with embedding-based retrieval."""

    def __init__(self, capacity: int = 1000):
        self.messages: List[Message] = []
        self.capacity = capacity
        self.events: List[Event] = []
        self.knowledge_base: Dict[str, str] = {}
        self.embedding_cache: Dict[str, List[float]] = {}

    def add_message(self, message: Message) -> None:
        """Add a message to memory."""
        self.messages.append(message)
        if len(self.messages) > self.capacity:
            self.messages.pop(0)

    def add_event(self, event: Event) -> None:
        """Add an event to memory."""
        self.events.append(event)

    def add_knowledge(self, key: str, value: str) -> None:
        """Add knowledge to the knowledge base."""
        self.knowledge_base[key] = value

    def get_recent_messages(self, n: int = 10) -> List[Message]:
        """Get the n most recent messages."""
        return self.messages[-n:] if self.messages else []

    def search_messages(self, query: str, n: int = 5) -> List[Message]:
        """Search for messages semantically similar to the query."""
        # Simplified implementation - in a real system this would use embeddings
        matching_messages = []
        for message in reversed(self.messages):
            if query.lower() in message.content.lower():
                matching_messages.append(message)
                if len(matching_messages) >= n:
                    break
        return matching_messages

    def get_events_by_type(self, event_type: EventType) -> List[Event]:
        """Get events of a specific type."""
        return [event for event in self.events if event.type == event_type]

    def get_knowledge(self, key: str) -> Optional[str]:
        """Get knowledge from the knowledge base."""
        return self.knowledge_base.get(key)

    def summarize(self) -> str:
        """Generate a summary of the agent's memory."""
        if not self.messages:
            return "No memories yet."

        recent_msgs = self.get_recent_messages(5)
        summary = "Recent interactions:\n"
        for msg in recent_msgs:
            summary += f"- {msg.sender}: {msg.content[:100]}...\n"

        if self.knowledge_base:
            summary += "\nKey knowledge:\n"
            for key, value in list(self.knowledge_base.items())[:5]:
                summary += f"- {key}: {value[:100]}...\n"

        return summary


class GraphDatabase:
    """Knowledge graph database for storing project-related information."""

    def __init__(self):
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.edges: List[Tuple[str, str, str, Dict[str, Any]]] = []

    def add_node(self, node_id: str, node_type: str,
                 properties: Dict[str, Any]) -> None:
        """Add a node to the graph."""
        self.nodes[node_id] = {"type": node_type, **properties}

    def add_edge(self,
                 source_id: str,
                 edge_type: str,
                 target_id: str,
                 properties: Dict[str, Any] = None) -> None:
        """Add an edge between two nodes."""
        if properties is None:
            properties = {}
        self.edges.append((source_id, edge_type, target_id, properties))

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get a node by ID."""
        return self.nodes.get(node_id)

    def get_nodes_by_type(self,
                          node_type: str) -> List[Tuple[str, Dict[str, Any]]]:
        """Get all nodes of a specific type."""
        return [(node_id, node_data)
                for node_id, node_data in self.nodes.items()
                if node_data.get("type") == node_type]

    def get_connected_nodes(self,
                            node_id: str,
                            edge_type: Optional[str] = None
                            ) -> List[Tuple[str, str, Dict[str, Any]]]:
        """Get nodes connected to the given node, optionally filtered by edge type."""
        connected = []
        for source, e_type, target, props in self.edges:
            if source == node_id and (edge_type is None
                                      or e_type == edge_type):
                connected.append((target, e_type, props))
            elif target == node_id and (edge_type is None
                                        or e_type == edge_type):
                connected.append((source, e_type, props))
        return connected

    def query(self, query_type: str,
              params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute a query on the graph database."""
        # Implementation of various query types
        if query_type == "path":
            return self._find_path(params.get("start"), params.get("end"))
        elif query_type == "neighbors":
            return self._find_neighbors(params.get("node_id"))
        elif query_type == "subgraph":
            return self._extract_subgraph(params.get("node_ids", []))
        return []

    def _find_path(self, start_id: str, end_id: str) -> List[Dict[str, Any]]:
        """Find a path between two nodes using BFS."""
        if start_id not in self.nodes or end_id not in self.nodes:
            return []

        visited = {start_id}
        queue = [(start_id, [])]

        while queue:
            current, path = queue.pop(0)

            if current == end_id:
                return path + [self.nodes[current]]

            for neighbor, edge_type, _ in self.get_connected_nodes(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [self.nodes[current]]))

        return []

    def _find_neighbors(self, node_id: str) -> List[Dict[str, Any]]:
        """Find all neighbors of a node."""
        if node_id not in self.nodes:
            return []

        neighbors = []
        for neighbor, edge_type, props in self.get_connected_nodes(node_id):
            if neighbor in self.nodes:
                neighbor_data = self.nodes[neighbor].copy()
                neighbor_data["edge_type"] = edge_type
                neighbor_data["edge_properties"] = props
                neighbors.append(neighbor_data)

        return neighbors

    def _extract_subgraph(self, node_ids: List[str]) -> List[Dict[str, Any]]:
        """Extract a subgraph containing the given nodes and their connections."""
        if not node_ids:
            return []

        subgraph = []
        for node_id in node_ids:
            if node_id in self.nodes:
                subgraph.append({
                    "id": node_id,
                    "data": self.nodes[node_id],
                    "type": "node"
                })

        for source, edge_type, target, props in self.edges:
            if source in node_ids and target in node_ids:
                subgraph.append({
                    "source": source,
                    "target": target,
                    "type": "edge",
                    "edge_type": edge_type,
                    "properties": props
                })

        return subgraph

    def visualize(self) -> str:
        """Generate a mermaid diagram of the knowledge graph."""
        mermaid = ["```mermaid", "graph TD"]

        # Add nodes
        for node_id, data in self.nodes.items():
            node_type = data.get("type", "Unknown")
            label = data.get("name", node_id)
            mermaid.append(f'    {node_id}["{label} ({node_type})"]')

        # Add edges
        for source, edge_type, target, _ in self.edges:
            mermaid.append(f"    {source} -->|{edge_type}| {target}")

        mermaid.append("```")
        return "\n".join(mermaid)

    def to_json(self) -> str:
        """Export the graph database to JSON."""
        data = {
            "nodes": self.nodes,
            "edges": [(s, e, t, p) for s, e, t, p in self.edges]
        }
        return json.dumps(data, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> 'GraphDatabase':
        """Import a graph database from JSON."""
        data = json.loads(json_str)
        db = cls()
        db.nodes = data.get("nodes", {})
        db.edges = [(s, e, t, p) for s, e, t, p in data.get("edges", [])]
        return db


class LLMClient:
    """Client for interacting with LLM APIs."""

    def __init__(self):
        self.api_key = Config.llm_api_key
        self.model = Config.llm_model
        self.base_url = Config.llm_base_url
        self.temperature = 0.2
        self.max_tokens = 4000
        self.timeout = 90
        self.cache = {}
        
        # Initialize API client
        self.aclient = AsyncOpenAI(api_key=self.api_key)
        if self.base_url != "https://api.openai.com/v1":
            self.aclient = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    async def aask(self,
                   prompt: str,
                   system_message: str = None,
                   temperature: float = None,
                   max_tokens: int = None,
                   use_cache: bool = True) -> str:
        """Ask the LLM a question and get the response."""
        # Check cache
        cache_key = f"{prompt}_{system_message}_{temperature}_{max_tokens}"
        if use_cache and cache_key in self.cache:
            logger.debug("Using cached LLM response")
            return self.cache[cache_key]

        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens

        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        try:
            response = await self.aclient.chat.completions.create(model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=self.timeout)

            content = response.choices[0].message.content.strip()

            # Cache the response
            if use_cache:
                self.cache[cache_key] = content

            return content
        except Exception as e:
            logger.error(f"Error calling LLM API: {e}")
            if Config.debug:
                traceback.print_exc()
            return f"Error: {str(e)}"

    async def aask_multiple(self,
                            prompts: List[str],
                            system_message: str = None) -> List[str]:
        """Ask multiple questions in parallel and get responses."""
        tasks = [self.aask(prompt, system_message) for prompt in prompts]
        return await asyncio.gather(*tasks)


# =============================================================================
# AGENT FRAMEWORK
# =============================================================================


class Tool(ABC):
    """Base class for tools that agents can use."""

    name: str
    description: str

    def __init__(self):
        self.llm_client = LLMClient()

    @abstractmethod
    async def run(self, *args, **kwargs) -> Dict[str, Any]:
        """Run the tool with the given arguments."""
        pass

    def get_spec(self) -> Dict[str, Any]:
        """Get the tool specification."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self._get_parameters()
        }

    def _get_parameters(self) -> Dict[str, Any]:
        """Get the parameters of the tool from its run method signature."""
        sig = inspect.signature(self.run)
        params = {}
        for name, param in sig.parameters.items():
            if name == "self":
                continue

            param_info = {
                "type": "string",  # Default type
                "required": param.default is inspect.Parameter.empty
            }

            # Try to get type annotation
            if param.annotation != inspect.Parameter.empty:
                if hasattr(
                        param.annotation,
                        "__origin__") and param.annotation.__origin__ is Union:
                    types = [
                        arg.__name__ for arg in param.annotation.__args__
                        if arg != type(None)
                    ]
                    param_info["type"] = " or ".join(types)
                else:
                    param_info["type"] = param.annotation.__name__

            # Add default value if present
            if param.default is not inspect.Parameter.empty:
                param_info["default"] = str(param.default)

            params[name] = param_info

        return params


class CodeExecutionTool(Tool):
    """Tool for executing Python code and returning the result."""

    name = "execute_code"
    description = "Execute Python code and return the result"

    async def run(self, code: str, timeout: int = 10) -> Dict[str, Any]:
        """Execute Python code and return the result."""
        try:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(suffix='.py',
                                             delete=False) as temp:
                temp_name = temp.name
                temp.write(code.encode())

            # Execute the code in a separate process
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                temp_name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE)

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout)

                # Clean up
                os.unlink(temp_name)

                return {
                    "success": process.returncode == 0,
                    "stdout": stdout.decode() if stdout else "",
                    "stderr": stderr.decode() if stderr else "",
                    "return_code": process.returncode
                }
            except asyncio.TimeoutError:
                # Kill the process if it times out
                process.kill()
                # Clean up
                os.unlink(temp_name)
                return {
                    "success": False,
                    "error": f"Execution timed out after {timeout} seconds",
                    "stdout": "",
                    "stderr": "",
                    "return_code": -1
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "stdout": "",
                "stderr": traceback.format_exc(),
                "return_code": -1
            }


class FileTool(Tool):
    """Tool for reading and writing files."""

    name = "file_tool"
    description = "Read from or write to files"

    async def run(self,
                  operation: str,
                  path: str,
                  content: str = None) -> Dict[str, Any]:
        """Perform file operations.

        Args:
            operation: The operation to perform (read/write/append/delete/list)
            path: The file path
            content: The content to write (for write and append operations)
        """
        # Ensure the path is within the project directory
        full_path = Path(Config.project_root) / path

        try:
            if operation.lower() == "read":
                if not full_path.exists():
                    return {
                        "success": False,
                        "error": f"File {path} does not exist"
                    }

                with open(full_path, "r", encoding="utf-8") as f:
                    return {"success": True, "content": f.read()}

            elif operation.lower() == "write":
                if content is None:
                    return {
                        "success": False,
                        "error": "Content is required for write operation"
                    }

                # Ensure the directory exists
                full_path.parent.mkdir(parents=True, exist_ok=True)

                with open(full_path, "w", encoding="utf-8") as f:
                    f.write(content)

                return {"success": True, "path": str(full_path)}

            elif operation.lower() == "append":
                if content is None:
                    return {
                        "success": False,
                        "error": "Content is required for append operation"
                    }

                # Ensure the directory exists
                full_path.parent.mkdir(parents=True, exist_ok=True)

                with open(full_path, "a", encoding="utf-8") as f:
                    f.write(content)

                return {"success": True, "path": str(full_path)}

            elif operation.lower() == "delete":
                if not full_path.exists():
                    return {
                        "success": False,
                        "error": f"File {path} does not exist"
                    }

                full_path.unlink()

                return {"success": True}

            elif operation.lower() == "list":
                if not full_path.exists():
                    return {
                        "success": False,
                        "error": f"Directory {path} does not exist"
                    }

                if not full_path.is_dir():
                    return {
                        "success": False,
                        "error": f"{path} is not a directory"
                    }

                files = [
                    str(f.relative_to(Config.project_root))
                    for f in full_path.glob("*")
                ]

                return {"success": True, "files": files}

            else:
                return {
                    "success": False,
                    "error": f"Unknown operation: {operation}"
                }

        except Exception as e:
            return {"success": False, "error": str(e)}


class SearchTool(Tool):
    """Tool for searching within project files."""

    name = "search_tool"
    description = "Search for content in project files"

    async def run(self,
                  query: str,
                  file_pattern: str = "*",
                  case_sensitive: bool = False) -> Dict[str, Any]:
        """Search for content in project files.

        Args:
            query: The text to search for
            file_pattern: Glob pattern for files to search in
            case_sensitive: Whether the search should be case-sensitive
        """
        results = []
        root_path = Path(Config.project_root)

        try:
            for path in root_path.glob(f"**/{file_pattern}"):
                if path.is_file():
                    try:
                        with open(path, "r", encoding="utf-8") as f:
                            content = f.read()

                            # Perform the search
                            if not case_sensitive:
                                content_lower = content.lower()
                                query_lower = query.lower()
                                if query_lower in content_lower:
                                    lines = content.split("\n")
                                    line_matches = []

                                    for i, line in enumerate(lines):
                                        if query_lower in line.lower():
                                            line_matches.append({
                                                "line_number":
                                                i + 1,
                                                "line":
                                                line.strip(),
                                                "context":
                                                "\n".join(
                                                    lines[max(0, i - 2):min(
                                                        len(lines), i + 3)])
                                            })

                                    if line_matches:
                                        results.append({
                                            "file":
                                            str(path.relative_to(root_path)),
                                            "matches":
                                            line_matches
                                        })
                            else:
                                if query in content:
                                    lines = content.split("\n")
                                    line_matches = []

                                    for i, line in enumerate(lines):
                                        if query in line:
                                            line_matches.append({
                                                "line_number":
                                                i + 1,
                                                "line":
                                                line.strip(),
                                                "context":
                                                "\n".join(
                                                    lines[max(0, i - 2):min(
                                                        len(lines), i + 3)])
                                            })

                                    if line_matches:
                                        results.append({
                                            "file":
                                            str(path.relative_to(root_path)),
                                            "matches":
                                            line_matches
                                        })
                    except Exception as e:
                        # Skip files that can't be read
                        continue

            return {
                "success": True,
                "results": results,
                "match_count": len(results)
            }

        except Exception as e:
            return {"success": False, "error": str(e)}


class WebSearchTool(Tool):
    """Simulated tool for web searching (no actual web access)."""

    name = "web_search"
    description = "Search the web for information"

    async def run(self, query: str, num_results: int = 5) -> Dict[str, Any]:
        """Simulate web search (no actual web access).

        Args:
            query: The search query
            num_results: Number of results to return
        """
        # This is a simulated tool - it doesn't actually search the web
        # In a real implementation, this would call a search API

        # Generate a template response based on the query
        response = await self.llm_client.aask(
            f"""You are simulating a web search API. Generate {num_results} search results for the query: "{query}"

            Format the results as a list of dictionaries with 'title', 'url', and 'snippet' keys. 
            Make the results realistic and diverse, as if they came from a real search engine.

            Return only the JSON without any additional text.
            """,
            temperature=0.7)

        try:
            # Try to parse the response as JSON
            results = json.loads(response)
            return {
                "success": True,
                "results": results if isinstance(results, list) else [],
                "query": query
            }
        except json.JSONDecodeError:
            # If parsing fails, extract results using regex
            try:
                # Look for JSON-like structures in the text
                match = re.search(r'\[\s*\{.*\}\s*\]', response, re.DOTALL)
                if match:
                    results = json.loads(match.group(0))
                    return {
                        "success": True,
                        "results":
                        results if isinstance(results, list) else [],
                        "query": query
                    }
                else:
                    # Create a basic structure with the response
                    return {
                        "success":
                        True,
                        "results": [{
                            "title": "Search results",
                            "url": "N/A",
                            "snippet": response
                        }],
                        "query":
                        query
                    }
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Failed to parse search results: {str(e)}",
                    "raw_response": response
                }


class PrototypeGeneratorTool(Tool):
    """Tool for generating UI prototypes."""

    name = "prototype_generator"
    description = "Generate UI prototypes as HTML/CSS"

    async def run(self,
                  description: str,
                  prototype_type: str = "web") -> Dict[str, Any]:
        """Generate a UI prototype based on a description.

        Args:
            description: Description of the desired UI
            prototype_type: Type of prototype (web/mobile/desktop)
        """
        system_message = """You are a UI prototype generator. Create HTML and CSS code for a prototype based on the description.
        The prototype should be self-contained in a single HTML file with inline CSS. Use modern design principles.
        Make the prototype interactive where appropriate with simple JavaScript. Do not include any external dependencies."""

        prompt = f"""
        Generate a {prototype_type} UI prototype based on this description:

        {description}

        Return the complete HTML file (including CSS and JavaScript) that I can save and open in a browser.
        The prototype should be visually appealing and functional.
        """

        try:
            response = await self.llm_client.aask(prompt, system_message)

            # Extract HTML code
            html_match = re.search(r'```(?:html)?\s*(<!DOCTYPE.*?<\/html>)```',
                                   response, re.DOTALL)
            if html_match:
                html_code = html_match.group(1)
            else:
                # Try a more lenient extraction
                html_match = re.search(r'<!DOCTYPE.*?<\/html>', response,
                                       re.DOTALL)
                if html_match:
                    html_code = html_match.group(0)
                else:
                    return {
                        "success": False,
                        "error": "Could not extract HTML code from response",
                        "raw_response": response
                    }

            # Save the prototype
            prototype_dir = Path(Config.project_root) / "prototypes"
            prototype_dir.mkdir(parents=True, exist_ok=True)

            filename = f"{prototype_type}_prototype_{int(time.time())}.html"
            file_path = prototype_dir / filename

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(html_code)

            return {
                "success": True,
                "file_path": str(file_path),
                "prototype_type": prototype_type,
                "html_code": html_code
            }

        except Exception as e:
            return {"success": False, "error": str(e)}


class TeamTaskManager:
    """Manages task queue and execution for a team of agents."""

    def __init__(self):
        self.task_queue = asyncio.Queue()
        self.results = {}
        self.event_bus = EventBus()
        self.executor = ThreadPoolExecutor(
            max_workers=Config.max_parallel_tasks)
        self.running = False
        self.completed_tasks = 0
        self.total_tasks = 0

    async def add_task(self, agent: 'Agent', task_type: str,
                       context: Dict[str, Any]) -> str:
        """Add a task to the queue."""
        task_id = str(uuid.uuid4())
        await self.task_queue.put((task_id, agent, task_type, context))
        self.total_tasks += 1

        logger.info(
            f"Task added to queue: {task_id} ({task_type}) for {agent.name}")

        self.event_bus.publish(
            Event(type=EventType.EXECUTION_STARTED,
                  source="TaskManager",
                  data={
                      "task_id": task_id,
                      "agent": agent.name,
                      "task_type": task_type
                  }))

        return task_id

    async def get_result(self,
                         task_id: str,
                         timeout: float = None) -> Optional[Dict[str, Any]]:
        """Get the result of a task."""
        start_time = time.time()
        while timeout is None or time.time() - start_time < timeout:
            if task_id in self.results:
                return self.results[task_id]
            await asyncio.sleep(0.1)
        return None

    async def start(self) -> None:
        """Start processing tasks."""
        if self.running:
            return

        self.running = True
        asyncio.create_task(self._process_tasks())

    async def stop(self) -> None:
        """Stop processing tasks."""
        self.running = False

    async def _process_tasks(self) -> None:
        """Process tasks from the queue."""
        while self.running:
            try:
                # Get a task from the queue
                try:
                    task_id, agent, task_type, context = await asyncio.wait_for(
                        self.task_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                # Process the task
                try:
                    result = await agent.execute_task(task_type, context)
                    self.results[task_id] = result
                    self.completed_tasks += 1

                    logger.info(
                        f"Task completed: {task_id} ({task_type}) by {agent.name}"
                    )

                    self.event_bus.publish(
                        Event(type=EventType.EXECUTION_COMPLETED,
                              source="TaskManager",
                              data={
                                  "task_id": task_id,
                                  "agent": agent.name,
                                  "task_type": task_type,
                                  "success": result.get("success", False)
                              }))

                except Exception as e:
                    logger.error(f"Error processing task {task_id}: {e}")
                    if Config.debug:
                        traceback.print_exc()

                    self.results[task_id] = {
                        "success": False,
                        "error": str(e),
                        "traceback": traceback.format_exc()
                    }

                    self.event_bus.publish(
                        Event(type=EventType.SYSTEM_ERROR,
                              source="TaskManager",
                              data={
                                  "task_id": task_id,
                                  "error": str(e)
                              }))

                finally:
                    # Mark the task as done
                    self.task_queue.task_done()

            except Exception as e:
                logger.error(f"Error in task processing loop: {e}")
                if Config.debug:
                    traceback.print_exc()

    def get_status(self) -> Dict[str, Any]:
        """Get the status of the task manager."""
        return {
            "running": self.running,
            "queue_size": self.task_queue.qsize(),
            "completed_tasks": self.completed_tasks,
            "total_tasks": self.total_tasks
        }


class Agent:
    """Base class for intelligent agents."""

    def __init__(self,
                 name: str,
                 expertise: List[str],
                 tools: List[Tool] = None):
        self.name = name
        self.expertise = expertise
        self.tools = tools or []
        self.memory = Memory()
        self.llm_client = LLMClient()
        self.event_bus = EventBus()
        self.graph_db = GraphDatabase()

    async def execute_task(self, task_type: str,
                           context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task based on its type."""
        # Publish agent started event
        self.event_bus.publish(
            Event(type=EventType.AGENT_STARTED,
                  source=self.name,
                  data={"task_type": task_type}))

        # Select the appropriate method based on task type
        if hasattr(self, f"task_{task_type}"):
            method = getattr(self, f"task_{task_type}")
            result = await method(context)
        else:
            result = {
                "success": False,
                "error": f"Unknown task type: {task_type}"
            }

        # Publish agent completed event
        self.event_bus.publish(
            Event(type=EventType.AGENT_COMPLETED,
                  source=self.name,
                  data={
                      "task_type": task_type,
                      "success": result.get("success", False)
                  }))

        return result

    async def use_tool(self, tool_name: str,
                       args: Dict[str, Any]) -> Dict[str, Any]:
        """Use a tool by name with the given arguments."""
        for tool in self.tools:
            if tool.name == tool_name:
                return await tool.run(**args)

        return {"success": False, "error": f"Tool not found: {tool_name}"}

    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get specifications of available tools."""
        return [tool.get_spec() for tool in self.tools]

    async def perceive(self, message: Message) -> None:
        """Perceive a message from the environment."""
        self.memory.add_message(message)

    async def generate_response(self,
                                prompt: str,
                                system_message: str = None) -> str:
        """Generate a response to a prompt using the LLM."""
        return await self.llm_client.aask(prompt, system_message)

    def add_to_memory(self, key: str, value: str) -> None:
        """Add knowledge to the agent's memory."""
        self.memory.add_knowledge(key, value)


class RequirementsAnalyst(Agent):
    """Agent specialized in analyzing requirements and creating specifications."""

    def __init__(self, name: str = "Requirements Analyst"):
        super().__init__(name=name,
                         expertise=[
                             "requirements analysis", "user stories",
                             "specifications"
                         ],
                         tools=[FileTool(),
                                WebSearchTool(),
                                SearchTool()])

    async def task_analyze_requirements(
            self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze requirements and create a specification document."""
        requirement = context.get("requirement", "")
        if not requirement:
            return {"success": False, "error": "No requirement provided"}

        # Add the requirement to memory
        self.add_to_memory("original_requirement", requirement)

        # Search for relevant information if needed
        search_results = await self.use_tool("web_search", {
            "query": requirement,
            "num_results": 3
        })

        # Generate the specification document
        system_message = """You are a senior requirements analyst with expertise in translating user requirements into detailed specifications.
        Analyze the requirement thoroughly and create a comprehensive specification document that includes:

        1. Project Overview
        2. User Stories in the format "As a [user], I want [feature] so that [benefit]"
        3. Functional Requirements (detailed, numbered list)
        4. Non-Functional Requirements (performance, security, usability, etc.)
        5. Constraints and Assumptions
        6. Success Criteria

        Use markdown formatting for better readability."""

        search_context = ""
        if search_results.get("success",
                              False) and search_results.get("results"):
            search_context = "Relevant information:\n" + "\n".join([
                f"- {r.get('title')}: {r.get('snippet')}"
                for r in search_results.get("results", [])
            ])

        prompt = f"""
        # Requirement Analysis Task

        ## Original Requirement
        {requirement}

        {search_context}

        Create a detailed specification document based on this requirement.
        """

        specification = await self.generate_response(prompt, system_message)

        # Save the specification document
        file_result = await self.use_tool(
            "file_tool", {
                "operation": "write",
                "path": "docs/requirements_specification.md",
                "content": specification
            })

        # Extract user stories for the knowledge graph
        user_stories = self._extract_user_stories(specification)
        for i, story in enumerate(user_stories):
            story_id = f"story_{i+1}"
            self.graph_db.add_node(story_id, "user_story", {"text": story})
            self.graph_db.add_edge("project", "has_story", story_id)

        # Update the knowledge graph with the specification
        self.graph_db.add_node(
            "specification", "document", {
                "name": "Requirements Specification",
                "path": "docs/requirements_specification.md",
                "description": "Detailed project requirements"
            })
        self.graph_db.add_edge("project", "has_specification", "specification")

        # Publish an event
        self.event_bus.publish(
            Event(type=EventType.ARTIFACT_CREATED,
                  source=self.name,
                  data={
                      "artifact_type": "specification",
                      "path": file_result.get("path"),
                      "user_stories": user_stories
                  }))

        return {
            "success": file_result.get("success", False),
            "specification": specification,
            "path": file_result.get("path"),
            "user_stories": user_stories
        }

    def _extract_user_stories(self, text: str) -> List[str]:
        """Extract user stories from text."""
        stories = []

        # Look for "As a ... I want ... so that ..." pattern
        pattern = r"As (?:an?|the) (.+?), I want (.+?) so that (.+?)(?:$|\.|\n)"
        matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)

        for match in matches:
            user = match.group(1).strip()
            want = match.group(2).strip()
            benefit = match.group(3).strip()
            story = f"As {user}, I want {want} so that {benefit}"
            stories.append(story)

        return stories


class SoftwareArchitect(Agent):
    """Agent specialized in designing software architecture."""

    def __init__(self, name: str = "Software Architect"):
        super().__init__(
            name=name,
            expertise=["software architecture", "system design", "API design"],
            tools=[FileTool(), SearchTool()])

    async def task_design_architecture(
            self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Design the software architecture based on requirements."""
        # Get the specification
        spec_path = context.get("specification_path",
                                "docs/requirements_specification.md")

        file_result = await self.use_tool("file_tool", {
            "operation": "read",
            "path": spec_path
        })

        if not file_result.get("success", False):
            return {
                "success":
                False,
                "error":
                f"Failed to read specification: {file_result.get('error')}"
            }

        specification = file_result.get("content", "")

        # Generate the architecture document
        system_message = """You are a senior software architect with expertise in designing scalable and maintainable systems.
        Create a comprehensive architecture design document that includes:

        1. Architecture Overview (pattern and approach)
        2. System Components (services, modules, libraries)
        3. Data Models (entities and relationships)
        4. API Design (endpoints, methods, parameters, responses)
        5. Technology Stack (languages, frameworks, databases)
        6. Deployment Architecture (infrastructure, scaling, services)
        7. Security Considerations
        8. Performance Considerations

        Use markdown formatting for better readability. Include diagrams using mermaid syntax where appropriate."""

        prompt = f"""
        # Architecture Design Task

        ## Project Specification
        {specification}

        Create a detailed architecture design document based on these requirements.
        Use mermaid diagrams for visualizing components, their relationships, and data models.
        """

        architecture = await self.generate_response(prompt, system_message)

        # Fix mermaid diagrams by escaping curly braces
        architecture = self._fix_mermaid_diagrams(architecture)

        # Save the architecture document
        file_result = await self.use_tool(
            "file_tool", {
                "operation": "write",
                "path": "design/architecture.md",
                "content": architecture
            })

        # Create component design document
        components = self._extract_components(architecture)
        component_details = await self._generate_component_details(
            components, specification)

        component_file_result = await self.use_tool(
            "file_tool", {
                "operation": "write",
                "path": "design/components.md",
                "content": component_details
            })

        # Update the knowledge graph
        self.graph_db.add_node(
            "architecture", "document", {
                "name": "Architecture Design",
                "path": "design/architecture.md",
                "description": "Software architecture design"
            })
        self.graph_db.add_edge("specification", "informs", "architecture")

        # Add components to the knowledge graph
        for i, component in enumerate(components):
            component_id = f"component_{i+1}"
            self.graph_db.add_node(
                component_id, "component", {
                    "name": component,
                    "description": f"Software component: {component}"
                })
            self.graph_db.add_edge("architecture", "defines", component_id)

        # Publish an event
        self.event_bus.publish(
            Event(type=EventType.ARCHITECTURE_UPDATED,
                  source=self.name,
                  data={
                      "path": file_result.get("path"),
                      "components": components
                  }))

        return {
            "success": file_result.get("success", False),
            "architecture": architecture,
            "path": file_result.get("path"),
            "components": components,
            "components_path": component_file_result.get("path")
        }

    def _fix_mermaid_diagrams(self, text: str) -> str:
        """Fix mermaid diagrams by escaping curly braces."""
        # Find all mermaid code blocks
        mermaid_blocks = re.finditer(r'```mermaid\s*([\s\S]*?)```', text,
                                     re.MULTILINE)

        result = text
        for block in mermaid_blocks:
            original_block = block.group(1)
            # Double any curly braces that are part of class definitions, etc.
            fixed_block = re.sub(r'(\w+)\s*{', r'\1{{', original_block)
            fixed_block = re.sub(r'}', r'}}', fixed_block)

            # Replace in the result
            result = result.replace(original_block, fixed_block)

        return result

    def _extract_components(self, architecture: str) -> List[str]:
        """Extract component names from the architecture document."""
        components = []

        # Look for component headers
        component_headers = re.finditer(
            r'##\s*(Component|Module|Service|Class):\s*(\w+)', architecture,
            re.IGNORECASE)
        for match in component_headers:
            components.append(match.group(2))

        # Look for components in class diagrams
        class_pattern = r'class\s+(\w+)'
        components.extend(re.findall(class_pattern, architecture))

        # Look for components in component diagrams
        component_pattern = r'\[(.+?)\]'
        components.extend(re.findall(component_pattern, architecture))

        # Deduplicate and clean
        unique_components = []
        for component in components:
            component = component.strip()
            if component and component not in unique_components:
                unique_components.append(component)

        return unique_components

    async def _generate_component_details(self, components: List[str],
                                          specification: str) -> str:
        """Generate detailed descriptions for each component."""
        if not components:
            return "No components identified."

        system_message = """You are a software architect detailing individual components of a system.
        For each component, provide:

        1. Purpose and Responsibilities
        2. Interfaces (methods, properties, events)
        3. Dependencies
        4. Key Algorithms or Processes
        5. Data Handled

        Use markdown formatting with clear headers for each component."""

        prompt = f"""
        # Component Design Task

        ## Project Specification
        {specification[:500]}... (truncated)

        ## Components to Detail
        {", ".join(components)}

        Create detailed design documents for each of these components.
        """

        component_details = await self.generate_response(
            prompt, system_message)
        return component_details


class TechLead(Agent):
    """Agent specialized in technical leadership and creating project structure."""

    def __init__(self, name: str = "Tech Lead"):
        super().__init__(name=name,
                         expertise=[
                             "project structure", "technical leadership",
                             "development standards"
                         ],
                         tools=[FileTool(), SearchTool()])

    async def task_create_project_structure(
            self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create the project structure based on architecture."""
        # Get the architecture
        arch_path = context.get("architecture_path", "design/architecture.md")

        file_result = await self.use_tool("file_tool", {
            "operation": "read",
            "path": arch_path
        })

        if not file_result.get("success", False):
            return {
                "success": False,
                "error":
                f"Failed to read architecture: {file_result.get('error')}"
            }

        architecture = file_result.get("content", "")

        # Generate the project structure document
        system_message = """You are a technical lead responsible for defining project structure and development standards.
        Create a detailed project structure document that includes:

        1. Directory Structure (folders and key files)
        2. Code Organization Patterns
        3. Naming Conventions
        4. Configuration Files
        5. Development Environment Setup
        6. Build and Deployment Scripts

        Use markdown formatting for better readability."""

        prompt = f"""
        # Project Structure Task

        ## Architecture Design
        {architecture[:1000]}... (truncated)

        Create a detailed project structure document based on this architecture.
        Include a comprehensive directory structure diagram using ASCII art or markdown.
        Also include template files for configuration, documentation, and other key project files.
        """

        structure = await self.generate_response(prompt, system_message)

        # Save the project structure document
        file_result = await self.use_tool(
            "file_tool", {
                "operation": "write",
                "path": "design/project_structure.md",
                "content": structure
            })

        # Extract and create directory structure
        directories = self._extract_directories(structure)
        created_directories = await self._create_directories(directories)

        # Generate template files
        template_files = await self._generate_template_files(
            structure, architecture)

        # Update the knowledge graph
        self.graph_db.add_node(
            "project_structure", "document", {
                "name": "Project Structure",
                "path": "design/project_structure.md",
                "description": "Project directory structure and organization"
            })
        self.graph_db.add_edge("architecture", "informs", "project_structure")

        # Publish an event
        self.event_bus.publish(
            Event(type=EventType.ARTIFACT_CREATED,
                  source=self.name,
                  data={
                      "artifact_type": "project_structure",
                      "path": file_result.get("path"),
                      "directories": created_directories,
                      "template_files": [f["path"] for f in template_files]
                  }))

        return {
            "success": file_result.get("success", False),
            "structure": structure,
            "path": file_result.get("path"),
            "directories": created_directories,
            "template_files": template_files
        }

    def _extract_directories(self, structure: str) -> List[str]:
        """Extract directory paths from the project structure document."""
        directories = []

        # Look for directory structure in code blocks
        structure_blocks = re.finditer(r'```(?:\w*)?\s*([\s\S]*?)```',
                                       structure, re.MULTILINE)

        for block in structure_blocks:
            block_content = block.group(1)

            # Check if it looks like a directory structure
            if '/' in block_content or '\\' in block_content:
                # Extract directories line by line
                for line in block_content.split('\n'):
                    line = line.strip()

                    # Skip empty lines and lines without directory indicators
                    if not line or (not '/' in line and not '\\' in line):
                        continue

                    # Extract directory path
                    # Remove ASCII art characters and leading/trailing whitespace
                    path = re.sub(r'[│├└─┬┼┤┴┌┐└┘]', '', line).strip()

                    # If path ends with a filename (no trailing slash), get the directory
                    if '.' in path.split('/')[-1] or '.' in path.split(
                            '\\')[-1]:
                        path = '/'.join(
                            path.split('/')[:-1]) or path.split('/')[0]

                    # Clean up the path
                    path = path.strip()
                    if path and path not in directories:
                        directories.append(path)

        return directories

    async def _create_directories(self, directories: List[str]) -> List[str]:
        """Create the specified directories."""
        created = []

        for directory in directories:
            # Skip if it's just a file reference
            if '.' in directory.split('/')[-1]:
                continue

            # Ensure the path is relative to the project root
            if directory.startswith('/'):
                directory = directory[1:]

            # Create the directory
            directory_path = Path(Config.project_root) / directory
            directory_path.mkdir(parents=True, exist_ok=True)

            created.append(directory)

        return created

    async def _generate_template_files(
            self, structure: str, architecture: str) -> List[Dict[str, str]]:
        """Generate template files based on the project structure."""
        template_files = []

        # Common template files to generate
        common_files = [
            {
                "path": "README.md",
                "type": "documentation"
            },
            {
                "path": ".gitignore",
                "type": "configuration"
            },
            {
                "path": "requirements.txt",
                "type": "configuration"
            },
            {
                "path": "setup.py",
                "type": "configuration"
            },
            {
                "path": "src/__init__.py",
                "type": "code"
            },
            {
                "path": "tests/__init__.py",
                "type": "code"
            },
            {
                "path": "docs/api.md",
                "type": "documentation"
            },
            {
                "path": "docs/usage.md",
                "type": "documentation"
            },
        ]

        for file_info in common_files:
            content = await self._generate_file_content(
                file_info["path"], file_info["type"], structure, architecture)

            file_result = await self.use_tool(
                "file_tool", {
                    "operation": "write",
                    "path": file_info["path"],
                    "content": content
                })

            if file_result.get("success", False):
                template_files.append({
                    "path":
                    file_info["path"],
                    "type":
                    file_info["type"],
                    "content_preview":
                    content[:100] + "..." if len(content) > 100 else content
                })

        return template_files

    async def _generate_file_content(self, path: str, file_type: str,
                                     structure: str, architecture: str) -> str:
        """Generate content for a template file."""
        # Generate appropriate content based on file type and path
        if file_type == "documentation" and path == "README.md":
            return await self._generate_readme(structure, architecture)

        elif file_type == "configuration" and path == ".gitignore":
            return self._generate_gitignore()

        elif file_type == "configuration" and path == "requirements.txt":
            return self._generate_requirements(architecture)

        elif file_type == "configuration" and path == "setup.py":
            return self._generate_setup_py(architecture)

        elif file_type == "code" and path.endswith("__init__.py"):
            return "# Auto-generated __init__.py file\n"

        elif file_type == "documentation" and path == "docs/api.md":
            return "# API Documentation\n\n*This is a template file that will be filled with API documentation.*\n"

        elif file_type == "documentation" and path == "docs/usage.md":
            return "# Usage Guide\n\n*This is a template file that will be filled with usage documentation.*\n"

        else:
            return f"# Template file: {path}\n\n*This is an auto-generated template file.*\n"

    async def _generate_readme(self, structure: str, architecture: str) -> str:
        """Generate README.md content."""
        system_message = """You are creating a README.md file for a new software project.
        Create a comprehensive README that includes:

        1. Project Title and Description
        2. Installation Instructions
        3. Usage Examples
        4. Project Structure
        5. Features
        6. Contributing Guidelines
        7. License Information

        Use markdown formatting for better readability."""

        prompt = f"""
        # README Generation Task

        Based on the following project information, create a comprehensive README.md file:

        ## Project Structure
        {structure[:500]}... (truncated)

        ## Architecture
        {architecture[:500]}... (truncated)

        Create a detailed README.md that would help developers understand and use this project.
        """

        return await self.generate_response(prompt, system_message)

    def _generate_gitignore(self) -> str:
        """Generate .gitignore content."""
        return """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
ENV/
env/

# IDE files
.idea/
.vscode/
*.swp
*.swo

# OS specific files
.DS_Store
Thumbs.db

# Project specific
logs/
.env
*.log
"""

    def _generate_requirements(self, architecture: str) -> str:
        """Generate requirements.txt content based on architecture."""
        # Extract libraries and frameworks from architecture
        libraries = []
        frameworks_pattern = r'(?:libraries|frameworks|dependencies):\s*(.*?)(?:\n\n|\n#|$)'
        frameworks_match = re.search(frameworks_pattern, architecture,
                                     re.IGNORECASE | re.DOTALL)

        if frameworks_match:
            frameworks_text = frameworks_match.group(1)
            for line in frameworks_text.split('\n'):
                # Look for library names
                libs = re.findall(
                    r'`([a-zA-Z0-9_-]+)`|"([a-zA-Z0-9_-]+)"|\'([a-zA-Z0-9_-]+)\'|(\b[a-zA-Z0-9_-]+\b)',
                    line)
                for lib_match in libs:
                    lib = next((l for l in lib_match if l), None)
                    if lib and lib.lower() not in ('and', 'or', 'the', 'for',
                                                   'with'):
                        libraries.append(lib)

        # Add some common libraries if none were found
        if not libraries:
            libraries = [
                "requests", "flask", "pytest", "sqlalchemy", "pydantic"
            ]

        # Create requirements.txt
        requirements = "# Requirements\n\n"
        for lib in libraries:
            requirements += f"{lib}\n"

        return requirements

    def _generate_setup_py(self, architecture: str) -> str:
        """Generate setup.py content."""
        return """from setuptools import setup, find_packages

setup(
    name="project_name",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "requests",
        "flask",
        "sqlalchemy",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A short description of the project",
    keywords="sample, setuptools, development",
    url="https://github.com/yourusername/project_name",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
    ],
    python_requires=">=3.8",
)
"""


class DeveloperAgent(Agent):
    """Agent specialized in implementing code."""

    def __init__(self, name: str = "Developer"):
        super().__init__(
            name=name,
            expertise=["programming", "implementation", "debugging"],
            tools=[FileTool(), SearchTool(),
                   CodeExecutionTool()])

    async def task_implement_component(
            self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Implement a software component."""
        component_name = context.get("component_name", "")
        if not component_name:
            return {"success": False, "error": "No component name provided"}

        # Get architecture and components
        arch_result = await self.use_tool("file_tool", {
            "operation": "read",
            "path": "design/architecture.md"
        })

        comp_result = await self.use_tool("file_tool", {
            "operation": "read",
            "path": "design/components.md"
        })

        struct_result = await self.use_tool(
            "file_tool", {
                "operation": "read",
                "path": "design/project_structure.md"
            })

        if not all(
                r.get("success", False)
                for r in [arch_result, comp_result, struct_result]):
            return {
                "success": False,
                "error": "Failed to read design documents"
            }

        architecture = arch_result.get("content", "")
        components = comp_result.get("content", "")
        structure = struct_result.get("content", "")

        # Extract component details
        component_details = self._extract_component_details(
            component_name, components)

        # Get existing files if any
        existing_code = await self._get_existing_code(component_name)

        # Generate code for the component
        system_message = f"""You are an expert developer implementing the {component_name} component.
        Write clean, well-documented, and efficient code that follows best practices.
        Include appropriate error handling, logging, and comments.
        The code should be production-ready and follow the architectural guidelines."""

        prompt = f"""
        # Implementation Task: {component_name}

        ## Component Details
        {component_details}

        ## Architecture Context
        {architecture[:500]}... (truncated)

        ## Project Structure
        {structure[:500]}... (truncated)

        ## Existing Code (if any)
        {existing_code}

        Implement the {component_name} component according to these specifications.
        Return complete, production-ready code files with proper imports, error handling, and documentation.
        For each file, include a header specifying the file path relative to the project root.
        """

        implementation = await self.generate_response(prompt, system_message)

        # Extract and save code files
        saved_files = await self._save_code_files(implementation,
                                                  component_name)

        # Try to execute the code to verify it
        verification_results = await self._verify_code(saved_files)

        # Update the knowledge graph
        self.graph_db.add_node(
            f"impl_{component_name}", "implementation", {
                "name": component_name,
                "files": [f["path"] for f in saved_files],
                "status": "implemented"
            })

        # Find the component in the graph and connect it
        components = self.graph_db.get_nodes_by_type("component")
        for comp_id, comp_data in components:
            if comp_data.get("name") == component_name:
                self.graph_db.add_edge(comp_id, "implemented_by",
                                       f"impl_{component_name}")
                break

        # Publish an event
        self.event_bus.publish(
            Event(type=EventType.CODE_GENERATED,
                  source=self.name,
                  data={
                      "component": component_name,
                      "files": [f["path"] for f in saved_files],
                      "verification": verification_results
                  }))

        return {
            "success": len(saved_files) > 0,
            "component": component_name,
            "implementation": implementation,
            "saved_files": saved_files,
            "verification_results": verification_results
        }

    def _extract_component_details(self, component_name: str,
                                   components_doc: str) -> str:
        """Extract details for a specific component from the components document."""
        # First try to find a section with the exact component name
        section_pattern = r'#+\s*' + re.escape(
            component_name) + r'\s*\n(.*?)(?=#+\s*\w+\s*\n|$)'
        section_match = re.search(section_pattern, components_doc, re.DOTALL)

        if section_match:
            return section_match.group(1).strip()

        # If not found, look for sections containing the component name
        contains_pattern = r'#+\s*(.*' + re.escape(
            component_name) + r'.*)\s*\n(.*?)(?=#+\s*\w+\s*\n|$)'
        contains_match = re.search(contains_pattern, components_doc, re.DOTALL)

        if contains_match:
            return contains_match.group(2).strip()

        # If still not found, return a generic message
        return f"No specific details found for {component_name}. Implement based on architecture and best practices."

    async def _get_existing_code(self, component_name: str) -> str:
        """Get existing code for the component if any."""
        # Try common patterns for file locations
        possible_paths = [
            f"src/{component_name.lower()}.py",
            f"src/{component_name.lower()}/{component_name.lower()}.py",
            f"src/{component_name.lower()}/__init__.py",
            f"src/components/{component_name.lower()}.py",
            f"src/modules/{component_name.lower()}.py"
        ]

        existing_code = ""
        for path in possible_paths:
            file_result = await self.use_tool("file_tool", {
                "operation": "read",
                "path": path
            })

            if file_result.get("success", False):
                existing_code += f"File: {path}\n```python\n{file_result.get('content', '')}\n```\n\n"

        if not existing_code:
            existing_code = "No existing code found for this component."

        return existing_code

    async def _save_code_files(self, implementation: str,
                               component_name: str) -> List[Dict[str, str]]:
        """Extract and save code files from the implementation."""
        saved_files = []

        # Extract file blocks
        file_blocks = re.finditer(
            r'(?:File:|##\s*File:)\s*([\w./]+)\s*```(?:\w+)?\s*([\s\S]*?)```',
            implementation, re.MULTILINE)

        for block in file_blocks:
            file_path = block.group(1).strip()
            file_content = block.group(2).strip()

            # Ensure the path is valid
            if not file_path:
                continue

            # If path doesn't start with src/, tests/, etc., add to appropriate directory
            if not any(
                    file_path.startswith(prefix)
                    for prefix in ["src/", "tests/", "docs/"]):
                if "test" in file_path.lower():
                    file_path = f"tests/{file_path}"
                else:
                    file_path = f"src/{file_path}"

            # Save the file
            file_result = await self.use_tool("file_tool", {
                "operation": "write",
                "path": file_path,
                "content": file_content
            })

            if file_result.get("success", False):
                saved_files.append({
                    "path":
                    file_path,
                    "content_preview":
                    file_content[:100] +
                    "..." if len(file_content) > 100 else file_content
                })

        # If no files were found using the pattern, try to create a default file
        if not saved_files:
            # Try to extract Python code blocks
            code_blocks = re.finditer(r'```(?:python)?\s*([\s\S]*?)```',
                                      implementation, re.MULTILINE)

            for i, block in enumerate(code_blocks):
                code_content = block.group(1).strip()

                # Skip empty blocks
                if not code_content:
                    continue

                # Create a default file name
                if i == 0:
                    file_path = f"src/{component_name.lower()}.py"
                else:
                    file_path = f"src/{component_name.lower()}_{i}.py"

                # Save the file
                file_result = await self.use_tool(
                    "file_tool", {
                        "operation": "write",
                        "path": file_path,
                        "content": code_content
                    })

                if file_result.get("success", False):
                    saved_files.append({
                        "path":
                        file_path,
                        "content_preview":
                        code_content[:100] +
                        "..." if len(code_content) > 100 else code_content
                    })

        return saved_files

    async def _verify_code(
            self, saved_files: List[Dict[str, str]]) -> Dict[str, Any]:
        """Verify that the code files are valid Python."""
        verification_results = {
            "success": True,
            "failures": [],
            "syntax_valid": True
        }

        for file_info in saved_files:
            file_path = file_info["path"]

            # Skip non-Python files
            if not file_path.endswith(".py"):
                continue

            # Get file content
            file_result = await self.use_tool("file_tool", {
                "operation": "read",
                "path": file_path
            })

            if file_result.get("success", False):
                code = file_result.get("content", "")

                # Check syntax
                exec_result = await self.use_tool("execute_code", {
                    "code": f"import ast\nast.parse('''{code}''')",
                    "timeout": 5
                })

                if not exec_result.get("success", False):
                    verification_results["syntax_valid"] = False
                    verification_results["success"] = False
                    verification_results["failures"].append({
                        "file":
                        file_path,
                        "error":
                        exec_result.get("stderr", "Unknown syntax error")
                    })

        return verification_results


class TesterAgent(Agent):
    """Agent specialized in writing and running tests."""

    def __init__(self, name: str = "Tester"):
        super().__init__(
            name=name,
            expertise=["testing", "quality assurance", "test automation"],
            tools=[FileTool(), SearchTool(),
                   CodeExecutionTool()])

    async def task_write_tests(self, context: Dict[str,
                                                   Any]) -> Dict[str, Any]:
        """Write tests for a component."""
        component_name = context.get("component_name", "")
        implementation_files = context.get("implementation_files", [])

        if not component_name:
            return {"success": False, "error": "No component name provided"}

        if not implementation_files:
            # Try to find implementation files
            search_result = await self.use_tool("search_tool", {
                "query": component_name,
                "file_pattern": "*.py"
            })

            if search_result.get("success",
                                 False) and search_result.get("results"):
                implementation_files = [
                    r["file"] for r in search_result.get("results")
                ]

        if not implementation_files:
            return {"success": False, "error": "No implementation files found"}

        # Get the code to test
        code_to_test = await self._get_code_to_test(implementation_files)

        # Generate tests
        system_message = f"""You are an expert test engineer writing tests for the {component_name} component.
        Write comprehensive test cases using pytest that cover:

        1. Basic functionality
        2. Edge cases
        3. Error handling
        4. Integration with other components (mocked)

        Tests should be thorough, readable, and maintainable.
        Include setup and teardown code where necessary."""

        prompt = f"""
        # Test Writing Task: {component_name}

        ## Code to Test
        {code_to_test}

        Write comprehensive test cases for this code using pytest.
        For each test file, include a header specifying the file path relative to the project root.
        The tests should verify that the code works correctly and handles edge cases appropriately.
        Include appropriate mocking for external dependencies.
        """

        tests = await self.generate_response(prompt, system_message)

        # Extract and save test files
        saved_files = await self._save_test_files(tests, component_name)

        # Try to run the tests
        test_results = await self._run_tests(saved_files)

        # Update the knowledge graph
        self.graph_db.add_node(
            f"tests_{component_name}", "test_suite", {
                "name": f"{component_name} Tests",
                "files": [f["path"] for f in saved_files],
                "status": "created"
            })

        # Find the implementation in the graph and connect it
        impls = self.graph_db.get_nodes_by_type("implementation")
        for impl_id, impl_data in impls:
            if impl_data.get("name") == component_name:
                self.graph_db.add_edge(impl_id, "tested_by",
                                       f"tests_{component_name}")
                break

        # Publish an event
        self.event_bus.publish(
            Event(type=EventType.TEST_CREATED,
                  source=self.name,
                  data={
                      "component": component_name,
                      "files": [f["path"] for f in saved_files],
                      "test_results": test_results
                  }))

        return {
            "success": len(saved_files) > 0,
            "component": component_name,
            "tests": tests,
            "saved_files": saved_files,
            "test_results": test_results
        }

    async def _get_code_to_test(self, implementation_files: List[str]) -> str:
        """Get the code to test from implementation files."""
        code = ""

        for file_path in implementation_files:
            file_result = await self.use_tool("file_tool", {
                "operation": "read",
                "path": file_path
            })

            if file_result.get("success", False):
                code += f"File: {file_path}\n```python\n{file_result.get('content', '')}\n```\n\n"

        return code

    async def _save_test_files(self, tests: str,
                               component_name: str) -> List[Dict[str, str]]:
        """Extract and save test files."""
        saved_files = []

        # Extract file blocks
        file_blocks = re.finditer(
            r'(?:File:|##\s*File:)\s*([\w./]+)\s*```(?:\w+)?\s*([\s\S]*?)```',
            tests, re.MULTILINE)

        for block in file_blocks:
            file_path = block.group(1).strip()
            file_content = block.group(2).strip()

            # Ensure the path is valid
            if not file_path:
                continue

            # If path doesn't start with tests/, add it
            if not file_path.startswith("tests/"):
                file_path = f"tests/{file_path}"

            # Save the file
            file_result = await self.use_tool("file_tool", {
                "operation": "write",
                "path": file_path,
                "content": file_content
            })

            if file_result.get("success", False):
                saved_files.append({
                    "path":
                    file_path,
                    "content_preview":
                    file_content[:100] +
                    "..." if len(file_content) > 100 else file_content
                })

        # If no files were found using the pattern, try to create a default file
        if not saved_files:
            # Try to extract Python code blocks
            code_blocks = re.finditer(r'```(?:python)?\s*([\s\S]*?)```', tests,
                                      re.MULTILINE)

            for i, block in enumerate(code_blocks):
                code_content = block.group(1).strip()

                # Skip empty blocks
                if not code_content:
                    continue

                # Create a default file name
                file_path = f"tests/test_{component_name.lower()}.py"

                # Save the file
                file_result = await self.use_tool(
                    "file_tool", {
                        "operation": "write",
                        "path": file_path,
                        "content": code_content
                    })

                if file_result.get("success", False):
                    saved_files.append({
                        "path":
                        file_path,
                        "content_preview":
                        code_content[:100] +
                        "..." if len(code_content) > 100 else code_content
                    })

                # Only use the first code block for the default file
                break

        return saved_files

    async def _run_tests(self, saved_files: List[Dict[str,
                                                      str]]) -> Dict[str, Any]:
        """Run the tests and return the results."""
        if not saved_files:
            return {"success": False, "error": "No test files to run"}

        test_results = {"success": True, "ran": 0, "failed": 0, "details": []}

        for file_info in saved_files:
            file_path = file_info["path"]

            # Ensure there's an __init__.py in the tests directory
            init_path = "tests/__init__.py"
            await self.use_tool(
                "file_tool", {
                    "operation": "write",
                    "path": init_path,
                    "content": "# Test package\n"
                })

            # Get the module name
            module_name = file_path.replace("/", ".").replace(".py", "")

            # Try to run the tests
            try:
                # Create a simple script to run the test
                test_script = f"""
import sys
import pytest
import os

# Add project root to path
sys.path.insert(0, os.getcwd())

# Run the test
result = pytest.main(["{file_path}", "-v"])
sys.exit(result)
"""

                exec_result = await self.use_tool("execute_code", {
                    "code": test_script,
                    "timeout": 20
                })

                test_ran = exec_result.get(
                    "success", False) or "collected" in exec_result.get(
                        "stdout", "")
                test_failed = "FAILED" in exec_result.get(
                    "stdout", "") or exec_result.get("return_code", 0) != 0

                if test_ran:
                    test_results["ran"] += 1

                    if test_failed:
                        test_results["failed"] += 1
                        test_results["success"] = False

                    test_results["details"].append({
                        "file":
                        file_path,
                        "success":
                        not test_failed,
                        "stdout":
                        exec_result.get("stdout", ""),
                        "stderr":
                        exec_result.get("stderr", "")
                    })
                else:
                    test_results["details"].append({
                        "file":
                        file_path,
                        "success":
                        False,
                        "error":
                        "Failed to run test",
                        "stdout":
                        exec_result.get("stdout", ""),
                        "stderr":
                        exec_result.get("stderr", "")
                    })
            except Exception as e:
                test_results["details"].append({
                    "file": file_path,
                    "success": False,
                    "error": str(e)
                })

        return test_results


class CodeReviewerAgent(Agent):
    """Agent specialized in code review and quality assurance."""

    def __init__(self, name: str = "Code Reviewer"):
        super().__init__(
            name=name,
            expertise=["code review", "best practices", "clean code"],
            tools=[FileTool(), SearchTool()])

    async def task_review_code(self, context: Dict[str,
                                                   Any]) -> Dict[str, Any]:
        """Review code for a component."""
        component_name = context.get("component_name", "")
        implementation_files = context.get("implementation_files", [])
        test_files = context.get("test_files", [])

        if not component_name:
            return {"success": False, "error": "No component name provided"}

        if not implementation_files:
            # Try to find implementation files
            search_result = await self.use_tool("search_tool", {
                "query": component_name,
                "file_pattern": "*.py"
            })

            if search_result.get("success",
                                 False) and search_result.get("results"):
                implementation_files = [
                    r["file"] for r in search_result.get("results")
                    if not r["file"].startswith("tests/")
                ]

        if not test_files:
            # Try to find test files
            search_result = await self.use_tool("search_tool", {
                "query": component_name,
                "file_pattern": "test_*.py"
            })

            if search_result.get("success",
                                 False) and search_result.get("results"):
                test_files = [r["file"] for r in search_result.get("results")]

        if not implementation_files:
            return {"success": False, "error": "No implementation files found"}

        # Get the code to review
        code_to_review = await self._get_code_to_review(implementation_files)
        tests_to_review = await self._get_code_to_review(test_files)

        # Generate review
        system_message = f"""You are an expert code reviewer evaluating the {component_name} component.
        Provide a thorough code review that includes:

        1. Code Quality (readability, maintainability, complexity)
        2. Potential Bugs and Issues
        3. Security Vulnerabilities
        4. Performance Considerations
        5. Test Coverage and Quality
        6. Adherence to Best Practices

        Be specific with line numbers and code examples when possible.
        Include both positive feedback and areas for improvement.
        Suggest concrete improvements for any issues found."""

        prompt = f"""
        # Code Review Task: {component_name}

        ## Implementation Code
        {code_to_review}

        ## Test Code
        {tests_to_review if tests_to_review else "No test files found."}

        Perform a comprehensive code review of this implementation and its tests.
        Identify any issues, bugs, or areas for improvement.
        Consider code quality, maintainability, security, performance, and test coverage.
        Format your review in markdown with clear sections and include specific code references.
        """

        review = await self.generate_response(prompt, system_message)

        # Save the review
        file_result = await self.use_tool(
            "file_tool", {
                "operation": "write",
                "path": f"reviews/review_{component_name}.md",
                "content": review
            })

        # Extract issues
        issues = self._extract_issues(review)

        # Update the knowledge graph
        self.graph_db.add_node(
            f"review_{component_name}", "review", {
                "name": f"{component_name} Review",
                "path": f"reviews/review_{component_name}.md",
                "issue_count": len(issues)
            })

        # Find the implementation in the graph and connect it
        impls = self.graph_db.get_nodes_by_type("implementation")
        for impl_id, impl_data in impls:
            if impl_data.get("name") == component_name:
                self.graph_db.add_edge(impl_id, "reviewed_by",
                                       f"review_{component_name}")
                break

        # Publish an event
        self.event_bus.publish(
            Event(type=EventType.REVIEW_SUBMITTED,
                  source=self.name,
                  data={
                      "component": component_name,
                      "path": file_result.get("path"),
                      "issues": issues
                  }))

        return {
            "success": file_result.get("success", False),
            "component": component_name,
            "review": review,
            "path": file_result.get("path"),
            "issues": issues
        }

    async def _get_code_to_review(self, file_paths: List[str]) -> str:
        """Get the code to review from file paths."""
        code = ""

        for file_path in file_paths:
            file_result = await self.use_tool("file_tool", {
                "operation": "read",
                "path": file_path
            })

            if file_result.get("success", False):
                code += f"File: {file_path}\n```python\n{file_result.get('content', '')}\n```\n\n"

        return code

    def _extract_issues(self, review: str) -> List[Dict[str, Any]]:
        """Extract issues from the review."""
        issues = []

        # Look for issue patterns in the review
        issue_patterns = [
            r'(?:Issue|Problem|Bug|Concern|Error)(?:\s*\d+)?:\s*(.*?)(?=\n\n|\n#|\n\*\*|\n-|\n\d+\.|\n\n|$)',
            r'\*\*(?:Issue|Problem|Bug|Concern|Error)(?:\s*\d+)?:\*\*\s*(.*?)(?=\n\n|\n#|\n\*\*|\n-|\n\d+\.|\n\n|$)',
            r'\n\d+\.\s*(.*?)(?=\n\n|\n#|\n\*\*|\n-|\n\d+\.|\n\n|$)'
        ]

        for pattern in issue_patterns:
            for match in re.finditer(pattern, review, re.MULTILINE):
                issue_text = match.group(1).strip()
                if issue_text and len(
                        issue_text) > 10:  # Filter out too short matches
                    issues.append({
                        "description":
                        issue_text,
                        "severity":
                        self._determine_severity(issue_text)
                    })

        return issues

    def _determine_severity(self, issue_text: str) -> str:
        """Determine the severity of an issue based on its description."""
        # Keywords indicating high severity
        high_severity = [
            "critical", "severe", "security", "vulnerability", "crash",
            "memory leak", "data loss"
        ]

        # Keywords indicating medium severity
        medium_severity = [
            "performance", "inefficient", "maintainability", "refactor",
            "complex", "confusing"
        ]

        # Check if any high severity keywords are in the issue text
        if any(keyword in issue_text.lower() for keyword in high_severity):
            return "high"

        # Check if any medium severity keywords are in the issue text
        if any(keyword in issue_text.lower() for keyword in medium_severity):
            return "medium"

        # Default to low severity
        return "low"


class UIDesignerAgent(Agent):
    """Agent specialized in UI design and prototyping."""

    def __init__(self, name: str = "UI Designer"):
        super().__init__(name=name,
                         expertise=["UI design", "UX", "prototyping"],
                         tools=[FileTool(),
                                PrototypeGeneratorTool()])

    async def task_create_ui_prototype(
            self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create a UI prototype for the application."""
        # Get requirements and architecture
        spec_result = await self.use_tool(
            "file_tool", {
                "operation": "read",
                "path": "docs/requirements_specification.md"
            })

        arch_result = await self.use_tool("file_tool", {
            "operation": "read",
            "path": "design/architecture.md"
        })

        if not spec_result.get("success", False) or not arch_result.get(
                "success", False):
            return {
                "success": False,
                "error":
                "Failed to read specification or architecture documents"
            }

        specification = spec_result.get("content", "")
        architecture = arch_result.get("content", "")

        # Extract UI/UX requirements
        ui_requirements = self._extract_ui_requirements(specification)

        # Generate UI design document
        system_message = """You are a UI/UX designer creating a design document for a software application.
        Create a comprehensive UI design document that includes:

        1. Design Principles and Guidelines
        2. Color Palette and Typography
        3. Layout and Navigation
        4. Key Screens and Components
        5. User Flows
        6. Accessibility Considerations

        Use markdown formatting for better readability."""

        prompt = f"""
        # UI Design Task

        ## Project Requirements
        {specification[:1000]}... (truncated)

        ## Architecture Context
        {architecture[:500]}... (truncated)

        ## UI Requirements
        {ui_requirements}

        Create a detailed UI/UX design document for this application.
        Include specific design elements, color schemes, layouts, and interaction patterns.
        """

        design_doc = await self.generate_response(prompt, system_message)

        # Save the design document
        file_result = await self.use_tool(
            "file_tool", {
                "operation": "write",
                "path": "design/ui_design.md",
                "content": design_doc
            })

        # Generate prototype
        prototype_result = await self.use_tool("prototype_generator", {
            "description": ui_requirements,
            "prototype_type": "web"
        })

        # Update the knowledge graph
        self.graph_db.add_node(
            "ui_design", "document", {
                "name": "UI Design",
                "path": "design/ui_design.md",
                "description": "UI/UX design specifications"
            })
        self.graph_db.add_edge("specification", "informs", "ui_design")

        if prototype_result.get("success", False):
            self.graph_db.add_node(
                "ui_prototype", "artifact", {
                    "name": "UI Prototype",
                    "path": prototype_result.get("file_path"),
                    "description": "Interactive UI prototype"
                })
            self.graph_db.add_edge("ui_design", "implemented_by",
                                   "ui_prototype")

        # Publish an event
        self.event_bus.publish(
            Event(type=EventType.PROTOTYPE_READY,
                  source=self.name,
                  data={
                      "design_path":
                      file_result.get("path"),
                      "prototype_path":
                      prototype_result.get("file_path")
                      if prototype_result.get("success", False) else None
                  }))

        return {
            "success":
            file_result.get("success", False),
            "design_document":
            design_doc,
            "design_path":
            file_result.get("path"),
            "prototype":
            prototype_result
            if prototype_result.get("success", False) else None
        }

    def _extract_ui_requirements(self, specification: str) -> str:
        """Extract UI/UX requirements from the specification document."""
        ui_sections = []

        # Look for UI/UX related sections
        section_patterns = [
            r'#+\s*UI(?:/| )?UX.*?\n(.*?)(?=#+\s|\Z)',
            r'#+\s*User Interface.*?\n(.*?)(?=#+\s|\Z)',
            r'#+\s*Interface.*?\n(.*?)(?=#+\s|\Z)',
            r'#+\s*Design.*?\n(.*?)(?=#+\s|\Z)'
        ]

        for pattern in section_patterns:
            matches = re.finditer(pattern, specification,
                                  re.DOTALL | re.IGNORECASE)
            for match in matches:
                ui_sections.append(match.group(1).strip())

        # If no specific UI sections found, look for UI-related requirements
        if not ui_sections:
            req_pattern = r'(?:[-*]\s*|^\d+\.\s*)(.*?(?:interface|UI|UX|screen|display|button|input|form|layout|design).*?)(?=\n[-*]|\n\d+\.|\n\n|\Z)'
            matches = re.finditer(req_pattern, specification,
                                  re.MULTILINE | re.IGNORECASE)

            for match in matches:
                ui_sections.append(match.group(1).strip())

        # If still no UI requirements found, create a generic one
        if not ui_sections:
            ui_sections = [
                "The application should have a clean, intuitive, and responsive user interface that follows modern design principles and ensures good user experience."
            ]

        return "\n\n".join(ui_sections)


class TechnicalWriterAgent(Agent):
    """Agent specialized in creating documentation."""

    def __init__(self, name: str = "Technical Writer"):
        super().__init__(
            name=name,
            expertise=["technical writing", "documentation", "tutorials"],
            tools=[FileTool(), SearchTool()])

    async def task_create_documentation(
            self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive documentation for the project."""
        # Collect all relevant project files
        project_files = await self._collect_project_files()

        # Generate documentation
        system_message = """You are a technical writer creating comprehensive documentation for a software project.
        Create clear, concise, and well-structured documentation that includes:

        1. Overview and Introduction
        2. Installation and Setup
        3. Configuration
        4. Usage Examples
        5. API Reference
        6. Troubleshooting
        7. FAQ

        Use markdown formatting for better readability. Include code examples where appropriate."""

        prompt = f"""
        # Documentation Task

        ## Project Files
        {project_files}

        Create comprehensive documentation for this project.
        Include detailed instructions for installation, configuration, and usage.
        Provide clear examples and API documentation for developers.
        The documentation should be accessible to both beginner and advanced users.
        """

        documentation = await self.generate_response(prompt, system_message)

        # Save the main documentation
        main_doc_result = await self.use_tool(
            "file_tool", {
                "operation": "write",
                "path": "docs/index.md",
                "content": documentation
            })

        # Generate additional documentation
        additional_docs = await self._generate_additional_docs(project_files)

        # Update the knowledge graph
        self.graph_db.add_node(
            "documentation", "document", {
                "name": "Project Documentation",
                "path": "docs/index.md",
                "description": "Comprehensive project documentation"
            })

        # Link documentation to other artifacts
        for node_type in ["specification", "architecture", "ui_design"]:
            nodes = self.graph_db.get_nodes_by_type("document")
            for node_id, node_data in nodes:
                if node_data.get("name", "").lower() in node_type:
                    self.graph_db.add_edge(node_id, "documented_in",
                                           "documentation")

        # Publish an event
        self.event_bus.publish(
            Event(type=EventType.ARTIFACT_CREATED,
                  source=self.name,
                  data={
                      "artifact_type": "documentation",
                      "main_path": main_doc_result.get("path"),
                      "additional_docs":
                      [doc["path"] for doc in additional_docs]
                  }))

        return {
            "success": main_doc_result.get("success", False),
            "documentation": documentation,
            "main_path": main_doc_result.get("path"),
            "additional_docs": additional_docs
        }

    async def _collect_project_files(self) -> str:
        """Collect information about project files for documentation."""
        file_summary = ""

        # List main directories
        for directory in ["src", "tests", "design", "docs"]:
            list_result = await self.use_tool("file_tool", {
                "operation": "list",
                "path": directory
            })

            if list_result.get("success", False):
                files = list_result.get("files", [])
                file_summary += f"\n## {directory.capitalize()} Files\n"

                for file_path in files:
                    if file_path.endswith(".py") or file_path.endswith(".md"):
                        file_result = await self.use_tool(
                            "file_tool", {
                                "operation": "read",
                                "path": file_path
                            })

                        if file_result.get("success", False):
                            content = file_result.get("content", "")

                            # Include a preview of the file content
                            preview = content[:200] + "..." if len(
                                content) > 200 else content
                            file_summary += f"\n### {file_path}\n```\n{preview}\n```\n"

        return file_summary

    async def _generate_additional_docs(
            self, project_files: str) -> List[Dict[str, str]]:
        """Generate additional documentation files."""
        additional_docs = []

        # Define additional documentation to generate
        doc_types = [{
            "name": "API Reference",
            "path": "docs/api.md",
            "focus": "Detailed API documentation for developers"
        }, {
            "name": "User Guide",
            "path": "docs/user_guide.md",
            "focus": "Step-by-step guide for end users"
        }, {
            "name":
            "Developer Guide",
            "path":
            "docs/developer_guide.md",
            "focus":
            "Guide for developers contributing to the project"
        }, {
            "name":
            "Installation",
            "path":
            "docs/installation.md",
            "focus":
            "Detailed installation instructions for different platforms"
        }]

        for doc in doc_types:
            system_message = f"""You are a technical writer creating a {doc['name']} document.
            Focus on {doc['focus']} with clear explanations and examples.
            Use markdown formatting for better readability."""

            prompt = f"""
            # Documentation Task: {doc['name']}

            ## Project Files
            {project_files}

            Create a detailed {doc['name']} document for this project.
            Focus on {doc['focus']}.
            Include relevant examples and explanations.
            """

            content = await self.generate_response(prompt, system_message)

            file_result = await self.use_tool("file_tool", {
                "operation": "write",
                "path": doc["path"],
                "content": content
            })

            if file_result.get("success", False):
                additional_docs.append({
                    "name":
                    doc["name"],
                    "path":
                    doc["path"],
                    "preview":
                    content[:200] + "..." if len(content) > 200 else content
                })

        return additional_docs


class ProjectManagerAgent(Agent):
    """Agent specialized in project management and coordination."""

    def __init__(self, name: str = "Project Manager"):
        super().__init__(
            name=name,
            expertise=["project management", "planning", "coordination"],
            tools=[FileTool(), SearchTool()])

    async def task_create_project_plan(
            self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create a comprehensive project plan."""
        # Get requirements and architecture
        spec_result = await self.use_tool(
            "file_tool", {
                "operation": "read",
                "path": "docs/requirements_specification.md"
            })

        arch_result = await self.use_tool("file_tool", {
            "operation": "read",
            "path": "design/architecture.md"
        })

        if not spec_result.get("success", False) or not arch_result.get(
                "success", False):
            return {
                "success": False,
                "error":
                "Failed to read specification or architecture documents"
            }

        specification = spec_result.get("content", "")
        architecture = arch_result.get("content", "")

        # Generate project plan
        system_message = """You are a project manager creating a comprehensive project plan.
        Create a detailed plan that includes:

        1. Project Overview and Objectives
        2. Team Structure and Roles
        3. Milestones and Deliverables
        4. Timeline and Schedule
        5. Task Breakdown (with task dependencies)
        6. Risk Assessment and Mitigation Strategies
        7. Resource Allocation
        8. Success Criteria and Evaluation

        Use markdown formatting with clear headers, tables, and lists."""

        prompt = f"""
        # Project Planning Task

        ## Project Requirements
        {specification[:1000]}... (truncated)

        ## Architecture
        {architecture[:500]}... (truncated)

        Create a comprehensive project plan for implementing this software project.
        Include specific milestones, deliverables, timeline, and task assignments.
        Consider potential risks and include mitigation strategies.
        Use mermaid diagrams for Gantt charts and other visualizations where appropriate.
        """

        plan = await self.generate_response(prompt, system_message)

        # Fix mermaid diagrams by escaping curly braces
        plan = self._fix_mermaid_diagrams(plan)

        # Save the project plan
        file_result = await self.use_tool("file_tool", {
            "operation": "write",
            "path": "docs/project_plan.md",
            "content": plan
        })

        # Extract milestones
        milestones = self._extract_milestones(plan)

        # Generate task list
        tasks = await self._generate_task_list(plan, milestones)

        task_list_result = await self.use_tool("file_tool", {
            "operation": "write",
            "path": "docs/task_list.md",
            "content": tasks
        })

        # Update the knowledge graph
        self.graph_db.add_node(
            "project_plan", "document", {
                "name": "Project Plan",
                "path": "docs/project_plan.md",
                "description": "Comprehensive project plan"
            })
        self.graph_db.add_edge("specification", "informs", "project_plan")

        # Add milestones to the knowledge graph
        for i, milestone in enumerate(milestones):
            milestone_id = f"milestone_{i+1}"
            self.graph_db.add_node(
                milestone_id, "milestone", {
                    "name": milestone["name"],
                    "description": milestone["description"],
                    "deadline": milestone.get("deadline", "TBD")
                })
            self.graph_db.add_edge("project_plan", "defines", milestone_id)

        # Publish an event
        self.event_bus.publish(
            Event(type=EventType.ARTIFACT_CREATED,
                  source=self.name,
                  data={
                      "artifact_type": "project_plan",
                      "path": file_result.get("path"),
                      "milestones": milestones
                  }))

        return {
            "success":
            file_result.get("success", False),
            "plan":
            plan,
            "path":
            file_result.get("path"),
            "milestones":
            milestones,
            "tasks_path":
            task_list_result.get("path") if task_list_result.get(
                "success", False) else None
        }

    def _fix_mermaid_diagrams(self, text: str) -> str:
        """Fix mermaid diagrams by escaping curly braces."""
        # Find all mermaid code blocks
        mermaid_blocks = re.finditer(r'```mermaid\s*([\s\S]*?)```', text,
                                     re.MULTILINE)

        result = text
        for block in mermaid_blocks:
            original_block = block.group(1)
            # Double any curly braces that are part of class definitions, etc.
            fixed_block = re.sub(r'(\w+)\s*{', r'\1{{', original_block)
            fixed_block = re.sub(r'}', r'}}', fixed_block)

            # Replace in the result
            result = result.replace(original_block, fixed_block)

        return result

    def _extract_milestones(self, plan: str) -> List[Dict[str, str]]:
        """Extract milestones from the project plan."""
        milestones = []

        # Look for milestone sections
        milestone_section = re.search(r'#+\s*Milestones.*?\n(.*?)(?=#+\s|\Z)',
                                      plan, re.DOTALL | re.IGNORECASE)

        if milestone_section:
            milestone_text = milestone_section.group(1)

            # Look for individual milestones
            milestone_pattern = r'(?:[-*]\s*|^\d+\.\s*)(?:\*\*)?([^:\n]+)(?:\*\*)?:?\s*(.*?)(?=\n[-*]|\n\d+\.|\n\n|\Z)'
            matches = re.finditer(milestone_pattern, milestone_text,
                                  re.MULTILINE)

            for match in matches:
                name = match.group(1).strip()
                description = match.group(2).strip()

                # Try to extract a deadline
                deadline_match = re.search(
                    r'(?:by|deadline|due):\s*(\d{1,2}(?:st|nd|rd|th)?\s+\w+(?:\s+\d{4})?|\w+\s+\d{1,2}(?:st|nd|rd|th)?(?:\s+\d{4})?|\d{4}-\d{2}-\d{2})',
                    description, re.IGNORECASE)

                deadline = deadline_match.group(1) if deadline_match else "TBD"

                milestones.append({
                    "name": name,
                    "description": description,
                    "deadline": deadline
                })

        # If no milestones found, create generic ones
        if not milestones:
            milestones = [{
                "name": "Project Initiation",
                "description":
                "Setup project structure and initial documentation",
                "deadline": "Week 1"
            }, {
                "name": "Design Completion",
                "description": "Complete architecture and detailed design",
                "deadline": "Week 2"
            }, {
                "name": "Implementation",
                "description": "Implement core features and functionality",
                "deadline": "Week 4"
            }, {
                "name": "Testing",
                "description": "Complete testing and bug fixing",
                "deadline": "Week 5"
            }, {
                "name": "Deployment",
                "description": "Prepare for deployment and release",
                "deadline": "Week 6"
            }]

        return milestones

    async def _generate_task_list(self, plan: str,
                                  milestones: List[Dict[str, str]]) -> str:
        """Generate a detailed task list based on the project plan and milestones."""
        system_message = """You are a project manager creating a detailed task list.
        Break down the project into specific, actionable tasks with:

        1. Task ID
        2. Task Name
        3. Description
        4. Estimated Duration
        5. Dependencies (other task IDs)
        6. Assigned To (role)
        7. Status (Not Started)

        Use markdown formatting with tables for better readability."""

        milestone_str = "\n".join([
            f"- {m['name']}: {m['description']} (Deadline: {m['deadline']})"
            for m in milestones
        ])

        prompt = f"""
        # Task List Generation

        ## Project Plan Summary
        {plan[:500]}... (truncated)

        ## Milestones
        {milestone_str}

        Create a detailed task list for this project breaking down each milestone into specific tasks.
        Use a structured format with task ID, name, description, duration, dependencies, assignment, and status.
        Ensure all important aspects of the project are covered with appropriate tasks.
        """

        tasks = await self.generate_response(prompt, system_message)
        return tasks


class DevOpsAgent(Agent):
    """Agent specialized in DevOps and deployment."""

    def __init__(self, name: str = "DevOps Engineer"):
        super().__init__(
            name=name,
            expertise=["DevOps", "CI/CD", "deployment", "infrastructure"],
            tools=[FileTool(), SearchTool()])

    async def task_create_deployment_config(
            self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create deployment configuration for the project."""
        # Get architecture and project structure
        arch_result = await self.use_tool("file_tool", {
            "operation": "read",
            "path": "design/architecture.md"
        })

        struct_result = await self.use_tool(
            "file_tool", {
                "operation": "read",
                "path": "design/project_structure.md"
            })

        if not arch_result.get("success", False) or not struct_result.get(
                "success", False):
            return {
                "success":
                False,
                "error":
                "Failed to read architecture or project structure documents"
            }

        architecture = arch_result.get("content", "")
        structure = struct_result.get("content", "")

        # Generate deployment document
        system_message = """You are a DevOps engineer creating deployment configurations.
        Create comprehensive deployment documentation that includes:

        1. Deployment Architecture
        2. Environment Configuration
        3. CI/CD Pipeline Setup
        4. Infrastructure as Code
        5. Monitoring and Logging
        6. Scaling and High Availability
        7. Security Considerations
        8. Backup and Disaster Recovery

        Include actual configuration files and scripts where appropriate."""

        prompt = f"""
        # Deployment Configuration Task

        ## Architecture
        {architecture[:1000]}... (truncated)

        ## Project Structure
        {structure[:500]}... (truncated)

        Create comprehensive deployment configuration and documentation for this project.
        Include configuration files for:
        - Docker (Dockerfile and docker-compose.yml)
        - CI/CD pipeline (.github/workflows/ci.yml or equivalent)
        - Infrastructure as Code (if applicable)
        - Environment configuration

        Provide detailed explanations for each configuration component.
        """

        deployment_doc = await self.generate_response(prompt, system_message)

        # Save the deployment document
        doc_result = await self.use_tool(
            "file_tool", {
                "operation": "write",
                "path": "docs/deployment.md",
                "content": deployment_doc
            })

        # Extract and save configuration files
        config_files = await self._extract_config_files(deployment_doc)

        # Update the knowledge graph
        self.graph_db.add_node(
            "deployment", "document", {
                "name": "Deployment Configuration",
                "path": "docs/deployment.md",
                "description": "Deployment and CI/CD configuration"
            })
        self.graph_db.add_edge("architecture", "informs", "deployment")

        # Add configuration files to the knowledge graph
        for i, config in enumerate(config_files):
            config_id = f"config_{i+1}"
            self.graph_db.add_node(
                config_id, "configuration", {
                    "name": config["name"],
                    "path": config["path"],
                    "description": config["description"]
                })
            self.graph_db.add_edge("deployment", "includes", config_id)

        # Publish an event
        self.event_bus.publish(
            Event(type=EventType.ARTIFACT_CREATED,
                  source=self.name,
                  data={
                      "artifact_type": "deployment_config",
                      "path": doc_result.get("path"),
                      "config_files": [f["path"] for f in config_files]
                  }))

        return {
            "success": doc_result.get("success", False),
            "document": deployment_doc,
            "path": doc_result.get("path"),
            "config_files": config_files
        }

    async def _extract_config_files(self,
                                    document: str) -> List[Dict[str, str]]:
        """Extract and save configuration files from the document."""
        config_files = []

        # Extract file blocks with ```
        file_blocks = re.finditer(
            r'(?:File:|##\s*File:)\s*([\w./]+)\s*```(?:\w+)?\s*([\s\S]*?)```',
            document, re.MULTILINE)

        for block in file_blocks:
            file_path = block.group(1).strip()
            file_content = block.group(2).strip()

            # Ensure the path is valid
            if not file_path:
                continue

            # Save the file
            file_result = await self.use_tool("file_tool", {
                "operation": "write",
                "path": file_path,
                "content": file_content
            })

            if file_result.get("success", False):
                # Determine file type and description
                file_name = Path(file_path).name
                file_type, description = self._get_file_type_and_description(
                    file_path)

                config_files.append({
                    "name":
                    file_name,
                    "path":
                    file_path,
                    "type":
                    file_type,
                    "description":
                    description,
                    "content_preview":
                    file_content[:100] +
                    "..." if len(file_content) > 100 else file_content
                })

        return config_files

    def _get_file_type_and_description(self,
                                       file_path: str) -> Tuple[str, str]:
        """Determine the file type and description based on the file path."""
        file_name = Path(file_path).name.lower()

        if file_name == "dockerfile":
            return "docker", "Docker container configuration"

        elif file_name == "docker-compose.yml" or file_name == "docker-compose.yaml":
            return "docker-compose", "Docker Compose services configuration"

        elif file_name.endswith(".yml") or file_name.endswith(".yaml"):
            if "workflow" in file_path or "github" in file_path:
                return "ci-cd", "CI/CD pipeline configuration"
            elif "kubernetes" in file_path or "k8s" in file_path:
                return "kubernetes", "Kubernetes configuration"
            else:
                return "yaml", "YAML configuration file"

        elif file_name.endswith(".tf"):
            return "terraform", "Terraform infrastructure configuration"

        elif file_name.endswith(".json"):
            return "json", "JSON configuration file"

        elif file_name.endswith(".env"):
            return "environment", "Environment variables configuration"

        elif file_name.startswith("."):
            return "config", "Configuration file"

        else:
            return "other", "Miscellaneous configuration file"


# =============================================================================
# TEAM COORDINATION
# =============================================================================


class ProjectCoordinator:
    """Central coordinator for the entire software development process."""

    def __init__(self):
        self.event_bus = EventBus()
        self.graph_db = GraphDatabase()
        self.task_manager = TeamTaskManager()
        self.agents = {}
        self.project_state = {
            "name":
            "",
            "description":
            "",
            "status":
            "initialized",
            "current_phase":
            "requirements",
            "phases": [
                "requirements", "architecture", "design", "implementation",
                "testing", "review", "documentation", "deployment"
            ]
        }

    def initialize_knowledge_graph(self) -> None:
        """Initialize the knowledge graph with the project node."""
        self.graph_db.add_node(
            "project", "project", {
                "name": self.project_state["name"],
                "description": self.project_state["description"],
                "status": self.project_state["status"]
            })

    def register_agent(self, agent: Agent) -> None:
        """Register an agent with the coordinator."""
        self.agents[agent.name] = agent
        agent.graph_db = self.graph_db  # Share the same graph DB

    def set_project_details(self, name: str, description: str) -> None:
        """Set project details."""
        self.project_state["name"] = name
        self.project_state["description"] = description

        # Update the project node in the graph
        self.graph_db.add_node(
            "project", "project", {
                "name": name,
                "description": description,
                "status": self.project_state["status"]
            })

    def setup_event_handlers(self) -> None:
        """Set up event handlers for coordination."""
        # Handler for artifact creation events
        self.event_bus.subscribe(EventType.ARTIFACT_CREATED,
                                 self._handle_artifact_created)

        # Handler for architecture updates
        self.event_bus.subscribe(EventType.ARCHITECTURE_UPDATED,
                                 self._handle_architecture_updated)

        # Handler for code generation
        self.event_bus.subscribe(EventType.CODE_GENERATED,
                                 self._handle_code_generated)

        # Handler for test creation
        self.event_bus.subscribe(EventType.TEST_CREATED,
                                 self._handle_test_created)

        # Handler for review submission
        self.event_bus.subscribe(EventType.REVIEW_SUBMITTED,
                                 self._handle_review_submitted)

        # Handler for error events
        self.event_bus.subscribe(EventType.SYSTEM_ERROR,
                                 self._handle_system_error)

    async def start(self) -> None:
        """Start the project coordinator."""
        # Start the task manager
        await self.task_manager.start()

        # Initialize the knowledge graph
        self.initialize_knowledge_graph()

        # Set up event handlers
        self.setup_event_handlers()

        logger.info(
            f"Project coordinator started for project: {self.project_state['name']}"
        )

    async def execute_project(self, requirement: str) -> Dict[str, Any]:
        """Execute the entire project based on a requirement."""
        # Set project details
        project_name = re.search(r'(\w+(?:\s+\w+){0,3})', requirement)
        project_name = project_name.group(
            1) + " Project" if project_name else "New Project"
        self.set_project_details(project_name, requirement)

        # Start with requirements analysis
        requirements_task_id = await self.task_manager.add_task(
            self.agents["Requirements Analyst"], "analyze_requirements",
            {"requirement": requirement})

        # Wait for requirements to be analyzed
        requirements_result = await self.task_manager.get_result(
            requirements_task_id)

        if not requirements_result or not requirements_result.get(
                "success", False):
            return {
                "success": False,
                "error": "Requirements analysis failed",
                "result": requirements_result
            }

        # Update project state
        self.project_state["current_phase"] = "architecture"

        # Continue with architecture design
        architecture_task_id = await self.task_manager.add_task(
            self.agents["Software Architect"], "design_architecture",
            {"specification_path": "docs/requirements_specification.md"})

        # Wait for architecture to be designed
        architecture_result = await self.task_manager.get_result(
            architecture_task_id)

        if not architecture_result or not architecture_result.get(
                "success", False):
            return {
                "success": False,
                "error": "Architecture design failed",
                "result": architecture_result
            }

        # Update project state
        self.project_state["current_phase"] = "design"

        # Continue with project structure
        structure_task_id = await self.task_manager.add_task(
            self.agents["Tech Lead"], "create_project_structure",
            {"architecture_path": "design/architecture.md"})

        # In parallel, create project plan
        plan_task_id = await self.task_manager.add_task(
            self.agents["Project Manager"], "create_project_plan", {})

        # In parallel, create UI prototype
        ui_task_id = await self.task_manager.add_task(
            self.agents["UI Designer"], "create_ui_prototype", {})

        # Wait for design tasks to complete
        structure_result = await self.task_manager.get_result(structure_task_id
                                                              )
        plan_result = await self.task_manager.get_result(plan_task_id)
        ui_result = await self.task_manager.get_result(ui_task_id)

        if not structure_result or not structure_result.get("success", False):
            return {
                "success": False,
                "error": "Project structure creation failed",
                "result": structure_result
            }

        # Update project state
        self.project_state["current_phase"] = "implementation"

        # Get components to implement
        components = architecture_result.get("components", [])

        # Implement each component
        implementation_results = []
        for component in components:
            impl_task_id = await self.task_manager.add_task(
                self.agents["Developer"], "implement_component",
                {"component_name": component})

            impl_result = await self.task_manager.get_result(impl_task_id)
            implementation_results.append(impl_result)

            # If implementation was successful, schedule tests
            if impl_result and impl_result.get("success", False):
                test_task_id = await self.task_manager.add_task(
                    self.agents["Tester"], "write_tests", {
                        "component_name":
                        component,
                        "implementation_files": [
                            f["path"]
                            for f in impl_result.get("saved_files", [])
                        ]
                    })

        # Update project state
        self.project_state["current_phase"] = "testing"

        # Wait for all tests to be written
        await asyncio.sleep(2)  # Give time for pending tasks to complete

        # Update project state
        self.project_state["current_phase"] = "review"

        # Review each component
        review_results = []
        for component in components:
            review_task_id = await self.task_manager.add_task(
                self.agents["Code Reviewer"], "review_code",
                {"component_name": component})

            review_result = await self.task_manager.get_result(review_task_id)
            review_results.append(review_result)

        # Update project state
        self.project_state["current_phase"] = "documentation"

        # Create documentation
        docs_task_id = await self.task_manager.add_task(
            self.agents["Technical Writer"], "create_documentation", {})

        # Wait for documentation to be created
        docs_result = await self.task_manager.get_result(docs_task_id)

        # Update project state
        self.project_state["current_phase"] = "deployment"

        # Create deployment configuration
        deploy_task_id = await self.task_manager.add_task(
            self.agents["DevOps Engineer"], "create_deployment_config", {})

        # Wait for deployment configuration to be created
        deploy_result = await self.task_manager.get_result(deploy_task_id)

        # Update project state
        self.project_state["status"] = "completed"

        # Generate a summary of the project
        summary = self._generate_project_summary()

        # Save the summary
        await self.use_tool("file_tool", {
            "operation": "write",
            "path": "project_summary.md",
            "content": summary
        })

        # Stop the task manager
        await self.task_manager.stop()

        return {
            "success": True,
            "project_name": self.project_state["name"],
            "state": self.project_state,
            "summary": summary
        }

    async def use_tool(self, tool_name: str,
                       args: Dict[str, Any]) -> Dict[str, Any]:
        """Use a tool by name."""
        # Find an agent that has the tool
        for agent in self.agents.values():
            for tool in agent.tools:
                if tool.name == tool_name:
                    return await tool.run(**args)

        return {"success": False, "error": f"Tool not found: {tool_name}"}

    def _generate_project_summary(self) -> str:
        """Generate a summary of the project."""
        # Get project details
        project_node = self.graph_db.get_node("project")

        # Get artifacts
        documents = self.graph_db.get_nodes_by_type("document")
        components = self.graph_db.get_nodes_by_type("component")
        implementations = self.graph_db.get_nodes_by_type("implementation")
        test_suites = self.graph_db.get_nodes_by_type("test_suite")
        reviews = self.graph_db.get_nodes_by_type("review")
        configurations = self.graph_db.get_nodes_by_type("configuration")

        # Build summary markdown
        summary = [
            f"# {self.project_state['name']} - Project Summary", "",
            f"## Project Description", self.project_state["description"], "",
            "## Project Status", f"Status: {self.project_state['status']}",
            f"Current Phase: {self.project_state['current_phase']}", "",
            "## Documents", ""
        ]

        for doc_id, doc_data in documents:
            summary.append(f"- {doc_data.get('name')}: {doc_data.get('path')}")

        summary.extend(["", "## Components", ""])

        for comp_id, comp_data in components:
            summary.append(f"- {comp_data.get('name')}")

        summary.extend(["", "## Implementations", ""])

        for impl_id, impl_data in implementations:
            summary.append(
                f"- {impl_data.get('name')}: {', '.join(impl_data.get('files', []))}"
            )

        summary.extend(["", "## Test Suites", ""])

        for test_id, test_data in test_suites:
            summary.append(
                f"- {test_data.get('name')}: {', '.join(test_data.get('files', []))}"
            )

        summary.extend(["", "## Code Reviews", ""])

        for review_id, review_data in reviews:
            summary.append(
                f"- {review_data.get('name')}: {review_data.get('path')} ({review_data.get('issue_count', 0)} issues)"
            )

        summary.extend(["", "## Deployment Configurations", ""])

        for config_id, config_data in configurations:
            summary.append(
                f"- {config_data.get('name')}: {config_data.get('path')}")

        summary.extend([
            "", "## Project Knowledge Graph", "",
            self.graph_db.visualize(), "", "## Generated Files", ""
        ])

        # List all files in the output directory
        try:
            root_path = Path(Config.project_root)
            for path in sorted(root_path.glob("**/*")):
                if path.is_file():
                    rel_path = path.relative_to(root_path)
                    summary.append(f"- {rel_path}")
        except Exception as e:
            summary.append(f"Error listing files: {str(e)}")

        return "\n".join(summary)

    def _handle_artifact_created(self, event: Event) -> None:
        """Handle artifact creation events."""
        artifact_type = event.data.get("artifact_type")

        if artifact_type == "specification":
            logger.info(
                f"Requirements specification created: {event.data.get('path')}"
            )
            self.project_state["current_phase"] = "architecture"

        elif artifact_type == "project_structure":
            logger.info(
                f"Project structure created with {len(event.data.get('directories', []))} directories"
            )
            self.project_state["current_phase"] = "implementation"

        elif artifact_type == "documentation":
            logger.info(
                f"Documentation created: {event.data.get('main_path')}")
            self.project_state["current_phase"] = "deployment"

        elif artifact_type == "deployment_config":
            logger.info(
                f"Deployment configuration created: {event.data.get('path')}")
            self.project_state["status"] = "completed"

    def _handle_architecture_updated(self, event: Event) -> None:
        """Handle architecture update events."""
        logger.info(
            f"Architecture updated with {len(event.data.get('components', []))} components"
        )
        self.project_state["current_phase"] = "design"

    def _handle_code_generated(self, event: Event) -> None:
        """Handle code generation events."""
        component = event.data.get("component")
        files = event.data.get("files", [])
        verification = event.data.get("verification", {})

        logger.info(
            f"Code generated for component {component} with {len(files)} files"
        )

        if not verification.get("syntax_valid", True):
            logger.warning(
                f"Syntax errors in component {component}: {verification.get('failures')}"
            )

    def _handle_test_created(self, event: Event) -> None:
        """Handle test creation events."""
        component = event.data.get("component")
        files = event.data.get("files", [])
        test_results = event.data.get("test_results", {})

        logger.info(
            f"Tests created for component {component} with {len(files)} files")

        if not test_results.get("success", True):
            logger.warning(
                f"Test failures for component {component}: {test_results.get('failed', 0)} failed tests"
            )

    def _handle_review_submitted(self, event: Event) -> None:
        """Handle review submission events."""
        component = event.data.get("component")
        issues = event.data.get("issues", [])

        logger.info(
            f"Review submitted for component {component} with {len(issues)} issues"
        )

        # If there are high severity issues, log them
        high_severity_issues = [
            issue for issue in issues if issue.get("severity") == "high"
        ]
        if high_severity_issues:
            logger.warning(
                f"High severity issues found in {component}: {len(high_severity_issues)} issues"
            )

    def _handle_system_error(self, event: Event) -> None:
        """Handle system error events."""
        error = event.data.get("error")
        task_id = event.data.get("task_id")

        logger.error(f"System error in task {task_id}: {error}")


# =============================================================================
# MAIN APPLICATION
# =============================================================================


async def run_synapse(requirement: str, output_dir: str = None) -> None:
    """Run the Synapse AI software development platform."""
    start_time = time.time()

    # Set output directory if provided
    if output_dir:
        Config.project_root = output_dir

    # Ensure output directories exist
    Config.ensure_directories()

    logger.info(f"Starting Synapse AI Software Development Platform")
    logger.info(f"Requirement: {requirement}")
    logger.info(f"Output directory: {Config.project_root}")

    # Create the project coordinator
    coordinator = ProjectCoordinator()

    # Create all agents
    agents = [
        RequirementsAnalyst(),
        SoftwareArchitect(),
        TechLead(),
        ProjectManagerAgent(),
        DeveloperAgent(),
        TesterAgent(),
        CodeReviewerAgent(),
        UIDesignerAgent(),
        TechnicalWriterAgent(),
        DevOpsAgent()
    ]

    # Register agents with the coordinator
    for agent in agents:
        coordinator.register_agent(agent)

    # Start the coordinator
    await coordinator.start()

    # Execute the project
    result = await coordinator.execute_project(requirement)

    # Calculate execution time
    execution_time = time.time() - start_time

    # Log the result
    if result.get("success", False):
        logger.info(
            f"Project completed successfully: {result.get('project_name')}")
        logger.info(f"Execution time: {execution_time:.2f} seconds")
        logger.info(f"Output directory: {Config.project_root}")
    else:
        logger.error(f"Project failed: {result.get('error')}")

    # Return the result
    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python synapse.py 'project requirement'")
        sys.exit(1)

    requirement = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    asyncio.run(run_synapse(requirement, output_dir))
