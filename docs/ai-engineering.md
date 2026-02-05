# AI Engineering

## Overview

AI Engineering combines software engineering with AI/ML to build production AI systems. This guide covers key concepts and practical implementations.

## Model Context Protocol (MCP)

### What is MCP?
Model Context Protocol is a standard for connecting AI models to external data sources and tools. It enables:
- Consistent integration patterns across different AI providers
- Secure access to external tools and data
- Standardized context management
- Tool/function discovery and invocation

### Core Concepts
- **Context Servers**: Provide access to data sources
- **Tools**: Functions that AI can invoke
- **Resources**: Data that AI can access
- **Prompts**: Reusable prompt templates

### Implementation Example
```python
# Example MCP server implementation
from mcp import Server, Tool

server = Server("my-app")

@server.tool()
def search_database(query: str) -> dict:
    """Search the database for relevant information"""
    # Implementation
    return {"results": [...]}

@server.tool()
def get_user_profile(user_id: str) -> dict:
    """Get user profile information"""
    # Implementation
    return {"user": {...}}
```

### Key Skills to Develop
- Design MCP-compliant tool interfaces
- Implement secure context providers
- Handle tool invocation errors gracefully
- Optimize context window usage
- Version and document MCP endpoints

## Retrieval-Augmented Generation (RAG)

### What is RAG?
RAG combines information retrieval with language generation. Instead of relying solely on the model's training data, RAG:
1. Retrieves relevant documents from a knowledge base
2. Augments the prompt with retrieved information
3. Generates a response using both the query and retrieved context

### Architecture Components

#### 1. Document Processing
```python
# Chunking strategy
def chunk_document(doc: str, chunk_size: int = 512, overlap: int = 50):
    """Split document into overlapping chunks"""
    chunks = []
    start = 0
    while start < len(doc):
        end = start + chunk_size
        chunks.append(doc[start:end])
        start = end - overlap
    return chunks
```

#### 2. Embedding Generation
```python
# Generate embeddings for semantic search
from openai import OpenAI

def generate_embeddings(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for text chunks"""
    client = OpenAI()
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [item.embedding for item in response.data]
```

#### 3. Vector Storage
```python
# Store and retrieve embeddings
from pinecone import Pinecone

pc = Pinecone(api_key="...")
index = pc.Index("knowledge-base")

def store_embeddings(chunks: list[str], embeddings: list[list[float]]):
    """Store chunks with embeddings in vector DB"""
    vectors = [
        {"id": str(i), "values": emb, "metadata": {"text": chunk}}
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
    ]
    index.upsert(vectors)

def retrieve_relevant(query_embedding: list[float], top_k: int = 5):
    """Retrieve most relevant chunks"""
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    return [match.metadata["text"] for match in results.matches]
```

#### 4. RAG Pipeline
```python
def rag_query(query: str) -> str:
    """Complete RAG pipeline"""
    # 1. Generate query embedding
    query_embedding = generate_embeddings([query])[0]
    
    # 2. Retrieve relevant chunks
    relevant_chunks = retrieve_relevant(query_embedding, top_k=5)
    
    # 3. Construct augmented prompt
    context = "\n\n".join(relevant_chunks)
    prompt = f"""Context:
{context}

Question: {query}

Answer based on the context above:"""
    
    # 4. Generate response
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content
```

### Advanced RAG Techniques
- **Hybrid Search**: Combine semantic and keyword search
- **Re-ranking**: Use a separate model to re-rank retrieved results
- **Query Expansion**: Generate multiple variations of the query
- **Multi-hop Retrieval**: Iteratively retrieve and reason over information
- **HyDE (Hypothetical Document Embeddings)**: Generate hypothetical answers first

### Key Skills to Develop
- Design effective chunking strategies
- Choose appropriate embedding models
- Optimize retrieval accuracy and speed
- Handle missing or contradictory information
- Monitor and improve RAG quality metrics

## Function Calling

### What is Function Calling?
Function calling allows AI models to invoke external functions/APIs to:
- Access real-time data
- Perform computations
- Execute actions
- Integrate with external systems

### Implementation with OpenAI
```python
from openai import OpenAI

client = OpenAI()

# Define functions
functions = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"]
                }
            },
            "required": ["location"]
        }
    }
]

# Make API call with function calling
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "What's the weather in Boston?"}],
    tools=[{"type": "function", "function": f} for f in functions],
    tool_choice="auto"
)

# Handle function call
if response.choices[0].message.tool_calls:
    tool_call = response.choices[0].message.tool_calls[0]
    function_name = tool_call.function.name
    arguments = json.loads(tool_call.function.arguments)
    
    # Execute function
    result = get_weather(**arguments)
    
    # Send result back to model
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": "What's the weather in Boston?"},
            response.choices[0].message,
            {
                "role": "tool",
                "content": json.dumps(result),
                "tool_call_id": tool_call.id
            }
        ]
    )
```

### Best Practices
- **Clear Descriptions**: Write detailed function descriptions
- **Type Safety**: Validate function arguments
- **Error Handling**: Handle function execution failures gracefully
- **Security**: Validate and sanitize all function inputs
- **Rate Limiting**: Protect external APIs from abuse
- **Idempotency**: Design functions to be safely retryable

## Evals (Evaluations)

### What are Evals?
Evals are systematic evaluations of AI system performance. They help:
- Measure quality and reliability
- Detect regressions
- Compare different approaches
- Guide improvements

### Types of Evals

#### 1. Deterministic Evals
```python
def test_exact_match():
    """Test for exact expected output"""
    result = ai_function("What is 2+2?")
    assert "4" in result

def test_format_compliance():
    """Test output format"""
    result = ai_function("Generate JSON with name and age")
    data = json.loads(result)
    assert "name" in data and "age" in data
```

#### 2. Model-Graded Evals
```python
def grade_response(question: str, answer: str, expected: str) -> float:
    """Use an AI model to grade responses"""
    grading_prompt = f"""Grade this answer on a scale of 0-1.

Question: {question}
Expected: {expected}
Actual: {answer}

Score (0-1):"""
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": grading_prompt}]
    )
    
    score_text = response.choices[0].message.content
    return float(score_text.strip())
```

#### 3. Human Evals
```python
def collect_human_feedback():
    """Collect human ratings for AI outputs"""
    # Present outputs to human raters
    # Collect ratings (thumbs up/down, 1-5 stars, etc.)
    # Aggregate and analyze results
    pass
```

### Eval Frameworks
- **OpenAI Evals**: Framework for evaluating OpenAI models
- **Promptfoo**: Automated LLM testing framework
- **Langsmith**: Eval and monitoring for LangChain apps
- **Custom Frameworks**: Build your own for specific needs

### Key Metrics
- **Accuracy**: Percentage of correct responses
- **Latency**: Response time (p50, p95, p99)
- **Cost**: Tokens used per request
- **User Satisfaction**: Thumbs up/down ratings
- **Task Success Rate**: Percentage of completed tasks

### Building an Eval Suite
```python
class EvalSuite:
    def __init__(self):
        self.test_cases = []
    
    def add_test(self, name: str, input: str, validator: callable):
        """Add a test case"""
        self.test_cases.append({
            "name": name,
            "input": input,
            "validator": validator
        })
    
    def run(self) -> dict:
        """Run all tests and return results"""
        results = {"passed": 0, "failed": 0, "errors": []}
        
        for test in self.test_cases:
            try:
                output = ai_function(test["input"])
                if test["validator"](output):
                    results["passed"] += 1
                else:
                    results["failed"] += 1
                    results["errors"].append({
                        "test": test["name"],
                        "output": output
                    })
            except Exception as e:
                results["failed"] += 1
                results["errors"].append({
                    "test": test["name"],
                    "error": str(e)
                })
        
        return results
```

## Agent Systems

### What are AI Agents?
AI agents are autonomous systems that:
- Perceive their environment
- Make decisions
- Take actions to achieve goals
- Learn from feedback

### Agent Architecture

#### 1. ReAct Pattern (Reasoning + Acting)
```python
def react_agent(goal: str, max_iterations: int = 10):
    """ReAct agent loop"""
    context = f"Goal: {goal}\n\n"
    
    for i in range(max_iterations):
        # Reasoning step
        prompt = f"""{context}
Thought: What should I do next?
Action: [choose action]
Action Input: [action parameters]"""
        
        response = llm(prompt)
        context += response + "\n"
        
        # Parse action
        action, action_input = parse_action(response)
        
        # Acting step
        if action == "FINISH":
            return parse_final_answer(action_input)
        
        observation = execute_action(action, action_input)
        context += f"Observation: {observation}\n"
    
    return "Max iterations reached"
```

#### 2. Multi-Agent Systems
```python
class Agent:
    def __init__(self, role: str, tools: list):
        self.role = role
        self.tools = tools
    
    def act(self, task: str) -> str:
        """Execute a task using available tools"""
        prompt = f"""You are a {self.role}.
Task: {task}
Available tools: {self.tools}

How will you complete this task?"""
        
        return llm(prompt)

# Orchestrator coordinates multiple agents
class MultiAgentSystem:
    def __init__(self, agents: list[Agent]):
        self.agents = agents
    
    def solve(self, problem: str) -> str:
        """Break down problem and delegate to agents"""
        # Decompose problem
        subtasks = decompose_problem(problem)
        
        # Assign subtasks to agents
        results = []
        for subtask in subtasks:
            agent = self.select_agent(subtask)
            result = agent.act(subtask)
            results.append(result)
        
        # Synthesize results
        return synthesize_results(results)
```

### Key Agent Capabilities
- **Tool Use**: Invoke external functions/APIs
- **Memory**: Maintain context across interactions
- **Planning**: Break down complex goals
- **Learning**: Improve from feedback
- **Collaboration**: Work with other agents

## AI-as-Judge

### Concept
Use AI models to evaluate other AI models' outputs. This enables:
- Automated quality assessment
- Scalable evaluation without human labelers
- Consistent grading criteria
- Rapid iteration

### Implementation
```python
def ai_judge(
    task: str,
    response: str,
    criteria: list[str]
) -> dict:
    """Use AI to judge response quality"""
    
    criteria_text = "\n".join([f"- {c}" for c in criteria])
    
    prompt = f"""Evaluate this AI response:

Task: {task}
Response: {response}

Criteria:
{criteria_text}

For each criterion, provide:
1. Score (0-10)
2. Justification

Format as JSON."""
    
    evaluation = llm(prompt)
    return json.loads(evaluation)

# Example usage
criteria = [
    "Accuracy: Is the response factually correct?",
    "Completeness: Does it fully answer the question?",
    "Clarity: Is it easy to understand?",
    "Safety: Does it avoid harmful content?"
]

evaluation = ai_judge(
    task="Explain photosynthesis",
    response=ai_response,
    criteria=criteria
)
```

### Advanced Techniques
- **Multiple Judges**: Use several models and aggregate scores
- **Chain-of-Thought Judging**: Ask judges to explain reasoning
- **Pairwise Comparison**: Compare two responses head-to-head
- **Calibration**: Validate judges against human ratings

## Self-Improving Agents

### Concept
Agents that improve their performance over time through:
- Learning from successful/failed attempts
- Updating their strategies
- Refining their knowledge
- Optimizing their prompts

### Implementation Approaches

#### 1. Reflection-Based Improvement
```python
class SelfImprovingAgent:
    def __init__(self):
        self.memory = []  # Store past attempts
        self.strategies = []  # Learned strategies
    
    def act(self, task: str) -> str:
        """Execute task and learn from result"""
        # Try task with current knowledge
        result = self.attempt_task(task)
        
        # Reflect on result
        reflection = self.reflect(task, result)
        
        # Store experience
        self.memory.append({
            "task": task,
            "result": result,
            "reflection": reflection
        })
        
        # Update strategies
        if reflection["success"]:
            self.learn_from_success(reflection)
        else:
            self.learn_from_failure(reflection)
        
        return result
    
    def reflect(self, task: str, result: str) -> dict:
        """Reflect on task attempt"""
        prompt = f"""Reflect on this task attempt:

Task: {task}
Result: {result}

Was it successful? What could be improved?"""
        
        reflection = llm(prompt)
        return parse_reflection(reflection)
    
    def learn_from_success(self, reflection: dict):
        """Extract and store successful strategies"""
        strategy = extract_strategy(reflection)
        self.strategies.append(strategy)
    
    def learn_from_failure(self, reflection: dict):
        """Learn what to avoid"""
        # Update strategies to avoid failure patterns
        pass
```

#### 2. Prompt Optimization
```python
def optimize_prompt(
    base_prompt: str,
    test_cases: list,
    iterations: int = 5
) -> str:
    """Iteratively improve prompt based on results"""
    
    best_prompt = base_prompt
    best_score = evaluate_prompt(base_prompt, test_cases)
    
    for i in range(iterations):
        # Generate variations
        variations = generate_prompt_variations(best_prompt)
        
        # Test each variation
        for variation in variations:
            score = evaluate_prompt(variation, test_cases)
            if score > best_score:
                best_score = score
                best_prompt = variation
        
        print(f"Iteration {i+1}: Best score = {best_score}")
    
    return best_prompt
```

#### 3. Fine-tuning from Experience
```python
def collect_training_data(agent: SelfImprovingAgent):
    """Extract training data from agent's memory"""
    training_examples = []
    
    for experience in agent.memory:
        if experience["reflection"]["success"]:
            training_examples.append({
                "prompt": experience["task"],
                "completion": experience["result"]
            })
    
    return training_examples

def fine_tune_model(training_data: list):
    """Fine-tune model on collected experiences"""
    # Create training file
    with open("training_data.jsonl", "w") as f:
        for example in training_data:
            f.write(json.dumps(example) + "\n")
    
    # Fine-tune (example with OpenAI)
    client.fine_tuning.jobs.create(
        training_file="training_data.jsonl",
        model="gpt-3.5-turbo"
    )
```

## Resources and Tools

### Frameworks
- **LangChain**: Framework for LLM applications
- **LlamaIndex**: Data framework for LLM applications
- **AutoGen**: Framework for multi-agent conversations
- **CrewAI**: Framework for orchestrating AI agents
- **Semantic Kernel**: Microsoft's SDK for AI orchestration

### Vector Databases
- **Pinecone**: Managed vector database
- **Weaviate**: Open-source vector search engine
- **Qdrant**: Vector similarity search engine
- **Milvus**: Open-source vector database
- **ChromaDB**: Embedding database

### Observability
- **LangSmith**: LangChain monitoring and debugging
- **Weights & Biases**: Experiment tracking
- **Helicone**: LLM observability platform
- **Arize**: ML observability platform

### Practice Projects

1. **Build a RAG System**: Document Q&A system with your own knowledge base
2. **Create an Agent**: Multi-tool agent that can search web, read files, execute code
3. **Implement Function Calling**: Assistant that can book appointments, send emails
4. **Build an Eval Suite**: Comprehensive testing for an AI feature
5. **Multi-Agent System**: Team of specialized agents solving complex tasks
6. **Self-Improving Agent**: Agent that learns from its mistakes
7. **AI Judge System**: Automated evaluation pipeline for AI outputs
