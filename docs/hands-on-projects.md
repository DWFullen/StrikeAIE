# Hands-On Projects

## Overview

Practical projects to build and demonstrate skills across all required areas. Each project integrates multiple concepts from full-stack engineering, AI engineering, security, and non-deterministic systems management.

## Project 1: Secure RAG-Based Documentation Assistant

### Description
Build a production-ready chatbot that answers questions about technical documentation using RAG.

### Skills Demonstrated
- RAG implementation
- Vector database integration
- API design
- Security (authentication, rate limiting, PII filtering)
- Non-deterministic output handling

### Requirements

#### Core Features
- [ ] Document ingestion pipeline
- [ ] Vector database (Pinecone/Weaviate/Qdrant)
- [ ] Semantic search with embeddings
- [ ] LLM integration for response generation
- [ ] REST API with authentication
- [ ] Rate limiting per user
- [ ] PII detection and redaction
- [ ] Response validation and retry logic

#### Technical Stack
- **Backend**: Python (FastAPI) or Node.js (Express)
- **Vector DB**: Pinecone/Weaviate/Qdrant
- **LLM**: OpenAI GPT-4 or Anthropic Claude
- **Database**: PostgreSQL for user data
- **Cache**: Redis for response caching
- **Auth**: JWT tokens

#### Implementation Steps

1. **Document Processing** (Week 1)
```python
# Document chunking
def chunk_documents(docs: List[str], chunk_size: int = 500) -> List[str]:
    """Split documents into chunks"""
    pass

# Generate embeddings
def generate_embeddings(chunks: List[str]) -> List[List[float]]:
    """Generate embeddings for chunks"""
    pass

# Store in vector DB
def index_documents(chunks: List[str], embeddings: List[List[float]]):
    """Store chunks and embeddings in vector DB"""
    pass
```

2. **RAG Pipeline** (Week 2)
```python
# Query processing
def rag_query(query: str, k: int = 5) -> str:
    """RAG query pipeline"""
    # 1. Generate query embedding
    # 2. Search vector DB
    # 3. Construct prompt with context
    # 4. Generate response
    # 5. Validate and filter output
    pass
```

3. **API Layer** (Week 3)
```python
# Secure API endpoint
@app.post("/api/v1/chat")
async def chat(request: ChatRequest, user: User = Depends(auth)):
    # Validate input
    # Check rate limit
    # Process query
    # Filter response
    # Return result
    pass
```

4. **Security & Testing** (Week 4)
- Implement authentication
- Add rate limiting
- PII detection
- Security testing
- Load testing

### Success Criteria
- [ ] Handles 100+ req/s
- [ ] <2s response time (p95)
- [ ] 90%+ answer accuracy
- [ ] Zero PII leaks in responses
- [ ] All endpoints authenticated
- [ ] Rate limiting working
- [ ] Comprehensive test suite

### Extensions
- Streaming responses with SSE
- Multi-language support
- Follow-up question handling
- Source citation
- Feedback collection

---

## Project 2: Multi-Agent Task Automation System

### Description
Build an agent system that breaks down complex tasks and delegates to specialized agents.

### Skills Demonstrated
- Agent architecture (ReAct pattern)
- Multi-agent coordination
- Function calling
- State management
- Error handling

### Requirements

#### Core Features
- [ ] Task decomposition
- [ ] Agent orchestration
- [ ] Tool/function execution
- [ ] Memory management
- [ ] Error recovery

#### Example Agents
1. **Research Agent**: Web search, document analysis
2. **Code Agent**: Code generation, debugging
3. **Data Agent**: Data analysis, visualization
4. **Communication Agent**: Email, notifications

#### Technical Stack
- **Backend**: Python (LangChain/AutoGen)
- **LLM**: GPT-4 with function calling
- **Database**: PostgreSQL for state
- **Queue**: Redis/Celery for async tasks
- **Tools**: Custom tool implementations

#### Implementation Steps

1. **Agent Framework** (Week 1)
```python
class Agent:
    def __init__(self, name: str, role: str, tools: List[Tool]):
        self.name = name
        self.role = role
        self.tools = tools
        self.memory = []
    
    def act(self, task: str) -> str:
        """Execute task using ReAct pattern"""
        pass

class MultiAgentSystem:
    def __init__(self, agents: List[Agent]):
        self.agents = agents
    
    def solve(self, problem: str) -> str:
        """Orchestrate agents to solve problem"""
        pass
```

2. **Tool Integration** (Week 2)
```python
# Define tools
tools = [
    Tool(name="web_search", func=web_search),
    Tool(name="code_executor", func=execute_code),
    Tool(name="send_email", func=send_email),
]

# Implement function calling
def execute_tool(tool_name: str, args: dict) -> str:
    """Execute tool with validation"""
    pass
```

3. **Task Orchestration** (Week 3)
```python
# Break down complex task
def decompose_task(task: str) -> List[str]:
    """Decompose task into subtasks"""
    pass

# Route to appropriate agent
def route_subtask(subtask: str) -> Agent:
    """Select best agent for subtask"""
    pass
```

4. **Testing & Refinement** (Week 4)
- Test with diverse tasks
- Measure success rates
- Optimize agent selection
- Error handling

### Success Criteria
- [ ] Completes 80%+ of test tasks
- [ ] Correct agent selection
- [ ] Proper error recovery
- [ ] State persistence
- [ ] Audit logging

### Extensions
- Self-improving agents (learn from feedback)
- Human-in-the-loop approvals
- Parallel agent execution
- Agent communication protocol

---

## Project 3: AI Model Evaluation Platform

### Description
Build a platform for evaluating AI models with automated testing, benchmarking, and AI-as-judge.

### Skills Demonstrated
- Eval framework design
- AI-as-judge implementation
- Test automation
- Metrics tracking
- Data visualization

### Requirements

#### Core Features
- [ ] Test case management
- [ ] Automated eval execution
- [ ] Multiple eval types (deterministic, AI-graded, human)
- [ ] Metrics dashboard
- [ ] Regression detection

#### Technical Stack
- **Backend**: Python (FastAPI)
- **Frontend**: React + Chart.js
- **Database**: PostgreSQL
- **Queue**: Celery for async evals
- **Judge Model**: GPT-4

#### Implementation Steps

1. **Eval Framework** (Week 1)
```python
class EvalSuite:
    def __init__(self):
        self.test_cases = []
    
    def add_test(self, test: TestCase):
        """Add test case"""
        pass
    
    def run(self, model: str) -> EvalResults:
        """Run all tests against model"""
        pass

# Define test types
class DeterministicTest:
    """Exact match or format validation"""
    pass

class AIGradedTest:
    """AI judge scores response"""
    pass

class HumanEvalTest:
    """Human reviewer scores"""
    pass
```

2. **AI Judge** (Week 2)
```python
def ai_judge(
    question: str,
    answer: str,
    criteria: List[str]
) -> dict:
    """Grade answer using AI"""
    prompt = f"""Grade this answer:
    
Question: {question}
Answer: {answer}

Criteria: {criteria}

Provide scores and justification."""
    
    result = llm(prompt)
    return parse_scores(result)
```

3. **Dashboard** (Week 3)
```typescript
// React dashboard
function EvalDashboard() {
  // Display metrics
  // Show test results
  // Compare models
  // Trend analysis
}
```

4. **CI Integration** (Week 4)
```yaml
# GitHub Actions workflow
name: Model Evaluation

on: [pull_request]

jobs:
  eval:
    runs-on: ubuntu-latest
    steps:
      - name: Run Evals
        run: python -m evals.run
      
      - name: Check Thresholds
        run: python -m evals.check --min-score 0.85
```

### Success Criteria
- [ ] 100+ test cases
- [ ] Multiple eval types
- [ ] Automated CI checks
- [ ] Trend tracking
- [ ] <5 min eval runtime

### Extensions
- A/B testing framework
- Cost tracking
- Latency monitoring
- Custom eval metrics

---

## Project 4: Secure AI API Gateway

### Description
Build an API gateway for AI services with authentication, rate limiting, cost control, and monitoring.

### Skills Demonstrated
- API gateway design
- Security implementation
- Rate limiting
- Cost management
- Observability

### Requirements

#### Core Features
- [ ] Authentication (JWT, API keys)
- [ ] Rate limiting (per-user, per-endpoint)
- [ ] Cost tracking and limits
- [ ] Request/response logging
- [ ] Circuit breaker for LLM failures
- [ ] Response caching
- [ ] PII filtering

#### Technical Stack
- **Gateway**: Kong/Nginx or custom (FastAPI/Express)
- **Auth**: Auth0 or custom JWT
- **Cache**: Redis
- **Monitoring**: Prometheus + Grafana
- **Database**: PostgreSQL

#### Implementation Steps

1. **Gateway Core** (Week 1)
```python
class AIGateway:
    def __init__(self):
        self.rate_limiter = RateLimiter()
        self.cost_tracker = CostTracker()
        self.cache = RedisCache()
    
    async def proxy_request(self, request: Request) -> Response:
        # Authenticate
        # Check rate limit
        # Check cost limit
        # Check cache
        # Forward to LLM
        # Filter response
        # Cache and return
        pass
```

2. **Security Layer** (Week 2)
```python
# Authentication
def verify_token(token: str) -> User:
    """Verify JWT token"""
    pass

# Rate limiting
class RateLimiter:
    def check(self, user_id: str, endpoint: str) -> bool:
        """Check if request allowed"""
        pass

# PII filtering
def filter_pii(text: str) -> str:
    """Remove PII from text"""
    pass
```

3. **Monitoring** (Week 3)
```python
# Prometheus metrics
request_counter = Counter('requests_total')
request_duration = Histogram('request_duration_seconds')
cost_gauge = Gauge('api_cost_dollars')

# Middleware
@app.middleware("http")
async def monitor_request(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    
    request_counter.inc()
    request_duration.observe(duration)
    
    return response
```

4. **Testing & Deployment** (Week 4)
- Load testing
- Security testing
- Deploy to cloud
- Set up monitoring

### Success Criteria
- [ ] 1000+ req/s capacity
- [ ] <50ms gateway overhead
- [ ] Zero authentication bypasses
- [ ] Rate limiting works correctly
- [ ] Cost tracking accurate
- [ ] Comprehensive monitoring

### Extensions
- API versioning
- Request replay for debugging
- Real-time alerts
- Custom middleware plugins

---

## Project 5: Full-Stack AI Chat Application

### Description
Build a complete chat application with streaming responses, real-time updates, and production deployment.

### Skills Demonstrated
- Full-stack development
- Real-time communication (WebSockets/SSE)
- Streaming LLM responses
- Cloud deployment
- CI/CD

### Requirements

#### Core Features
- [ ] User authentication
- [ ] Real-time chat UI
- [ ] Streaming AI responses
- [ ] Chat history
- [ ] Multi-user support
- [ ] File upload
- [ ] Responsive design

#### Technical Stack
- **Frontend**: React + TypeScript
- **Backend**: FastAPI or Node.js
- **Database**: PostgreSQL
- **Real-time**: WebSockets or SSE
- **Deployment**: AWS/GCP/Azure
- **CI/CD**: GitHub Actions

#### Implementation Steps

1. **Frontend** (Week 1-2)
```typescript
// Chat component with streaming
function ChatInterface() {
  const [messages, setMessages] = useState([]);
  
  const streamResponse = async (message: string) => {
    const response = await fetch('/api/chat/stream', {
      method: 'POST',
      body: JSON.stringify({ message }),
      headers: { 'Content-Type': 'application/json' }
    });
    
    const reader = response.body.getReader();
    let currentMessage = '';
    
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      
      const chunk = new TextDecoder().decode(value);
      currentMessage += chunk;
      setMessages(prev => [...prev, { text: currentMessage, streaming: true }]);
    }
  };
  
  return <ChatUI messages={messages} onSend={streamResponse} />;
}
```

2. **Backend Streaming** (Week 2-3)
```python
from fastapi.responses import StreamingResponse

@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """Stream AI response"""
    async def generate():
        response = await llm_stream(request.message)
        async for chunk in response:
            yield chunk
    
    return StreamingResponse(generate(), media_type="text/event-stream")
```

3. **Database & Auth** (Week 3)
```python
# User model
class User(Base):
    id = Column(Integer, primary_key=True)
    email = Column(String, unique=True)
    hashed_password = Column(String)
    conversations = relationship("Conversation")

# Conversation model
class Conversation(Base):
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    messages = relationship("Message")
```

4. **Deployment** (Week 4)
```yaml
# docker-compose.yml
version: '3'
services:
  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
  
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://...
      - OPENAI_API_KEY=${OPENAI_API_KEY}
  
  postgres:
    image: postgres:15
    volumes:
      - pgdata:/var/lib/postgresql/data
```

### Success Criteria
- [ ] Smooth streaming experience
- [ ] <1s first token latency
- [ ] Responsive UI
- [ ] Deployed to cloud
- [ ] CI/CD pipeline
- [ ] Monitoring dashboards

### Extensions
- Voice input/output
- Multi-modal (images, PDFs)
- Conversation sharing
- Export chat history
- Custom themes

---

## Learning Path

### Month 1: Foundations
- **Week 1**: Project 1 (RAG basics)
- **Week 2**: Project 1 (API & Security)
- **Week 3**: Project 3 (Evals - Part 1)
- **Week 4**: Review and practice

### Month 2: Advanced
- **Week 1**: Project 2 (Agents - Part 1)
- **Week 2**: Project 2 (Agents - Part 2)
- **Week 3**: Project 4 (API Gateway)
- **Week 4**: Review and practice

### Month 3: Integration
- **Week 1-2**: Project 5 (Full-stack - Part 1)
- **Week 3**: Project 5 (Full-stack - Part 2)
- **Week 4**: Portfolio and documentation

## Portfolio Presentation

### What to Showcase
1. **GitHub Repositories**: Well-documented code
2. **Live Demos**: Deployed applications
3. **Architecture Diagrams**: System designs
4. **Blog Posts**: Technical write-ups
5. **Metrics**: Performance benchmarks

### Resume Points
- "Built production RAG system handling 1000+ req/s with <2s latency"
- "Implemented multi-agent system with 85% task completion rate"
- "Created eval platform with AI-as-judge for automated testing"
- "Designed secure API gateway with rate limiting and PII protection"
- "Deployed full-stack AI chat app on AWS with CI/CD"

## Additional Mini-Projects

### Quick Weekend Projects
1. **Prompt Injection Tester**: Test prompts for injection vulnerabilities
2. **Token Counter**: Estimate costs for LLM calls
3. **Embedding Visualizer**: Visualize embeddings in 2D/3D
4. **RAG Eval Suite**: Compare different chunking strategies
5. **Cost Monitor**: Track and alert on API costs
6. **Response Time Dashboard**: Monitor LLM latency
7. **PII Detector**: Scan text for sensitive data
8. **Function Calling Demo**: Showcase tool use
9. **Agent Logger**: Visualize agent decision making
10. **Prompt Optimizer**: A/B test prompt variations

## Resources

### Code Repositories for Reference
- LangChain examples
- OpenAI Cookbook
- Pinecone examples
- FastAPI examples
- React + TypeScript starters

### Learning Resources
- OpenAI API documentation
- LangChain documentation
- Vector database tutorials
- Full-stack tutorials
- DevOps best practices

### Communities
- r/MachineLearning
- r/LanguageTechnology
- Discord: LangChain, OpenAI
- Twitter: Follow AI engineers
- GitHub: Star and study popular projects

## Next Steps

1. **Choose a project** that excites you
2. **Set up environment** (tools, accounts, etc.)
3. **Build iteratively** (MVP first, then enhance)
4. **Document as you go** (README, architecture docs)
5. **Share your work** (GitHub, blog, Twitter)
6. **Get feedback** (code reviews, community input)
7. **Iterate and improve** (based on feedback)

Remember: The goal is to **build real things** that demonstrate your skills. Focus on quality over quantity, and make sure each project solves a real problem.
