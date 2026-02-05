# Technical Communication

## Overview

Strong technical communication is essential for explaining complex architecture decisions, collaborating with teams, and documenting systems. This guide covers best practices for technical communication in AI engineering roles.

## Architecture Decision Records (ADRs)

### What are ADRs?

Architecture Decision Records document important architectural decisions and their context. They help:
- Capture the reasoning behind decisions
- Provide context for future maintainers
- Enable informed decision-making
- Create institutional knowledge

### ADR Template

```markdown
# ADR-001: Use Vector Database for RAG System

## Status
Accepted

## Context
We need to build a RAG (Retrieval-Augmented Generation) system for our documentation chatbot. The system must:
- Handle 100K+ document chunks
- Support semantic search with <100ms latency
- Scale to millions of chunks over time
- Integrate with our existing Python backend

## Decision
We will use Pinecone as our vector database for the RAG system.

## Alternatives Considered

### 1. PostgreSQL with pgvector
**Pros:**
- We already use PostgreSQL
- No additional infrastructure
- Lower cost for small scale

**Cons:**
- Poor performance at scale (>100K vectors)
- Limited vector search optimizations
- Requires manual indexing tuning

### 2. Weaviate (Self-hosted)
**Pros:**
- Open source
- Good performance
- Rich query capabilities

**Cons:**
- Requires infrastructure management
- Higher operational overhead
- Scaling complexity

### 3. Pinecone (Managed)
**Pros:**
- Excellent performance at scale
- Fully managed (no ops overhead)
- Built-in monitoring and analytics
- Easy to get started

**Cons:**
- Vendor lock-in
- Higher cost at scale
- Less control over infrastructure

## Consequences

### Positive
- Fast time to market (no infrastructure setup)
- Predictable performance
- Minimal operational overhead
- Can focus on application logic

### Negative
- Monthly cost ($70/month minimum)
- Vendor dependency
- Migration effort if we switch later

### Neutral
- Need to learn Pinecone API
- Data stored outside our infrastructure

## Implementation Notes
- Start with starter tier ($70/month)
- Monitor query latency and cost
- Plan for migration strategy if needed
- Document vector schema and indexing strategy

## References
- [Pinecone documentation](https://docs.pinecone.io)
- [Vector database comparison](internal-link)
- [RAG architecture design](internal-link)

## Date
2024-01-15

## Author
@engineer-name
```

### Key Elements of Good ADRs

1. **Clear Context**: Explain the problem and constraints
2. **Explicit Decision**: State what was decided
3. **Alternatives**: Show what else was considered
4. **Trade-offs**: Acknowledge pros and cons
5. **Consequences**: Document impact of the decision
6. **Status**: Track if accepted, rejected, superseded

## Technical Design Documents

### Design Doc Template

```markdown
# Design: Multi-Agent RAG System

## Overview
Build a multi-agent system that routes user queries to specialized agents, each with their own RAG knowledge base.

## Goals
- Support 1000+ concurrent users
- <2 second response time (p95)
- 95% query routing accuracy
- Support for 5+ specialized domains

## Non-Goals
- Real-time learning from user feedback (future phase)
- Multi-language support (future phase)
- Voice interface (future phase)

## Background
Currently, our chatbot uses a single knowledge base, leading to:
- Poor responses for specialized queries
- Slow retrieval across large corpus
- Difficulty maintaining diverse content

## Proposed Solution

### Architecture
```
User Query
    ↓
Query Router (LLM)
    ↓
    ├─→ Technical Agent (Tech Docs KB)
    ├─→ Sales Agent (Sales KB)
    ├─→ Support Agent (Support KB)
    └─→ General Agent (General KB)
    ↓
Response Aggregator
    ↓
User Response
```

### Components

#### 1. Query Router
- **Technology**: GPT-4 with function calling
- **Input**: User query
- **Output**: Agent selection + confidence score
- **Fallback**: General agent if confidence <0.7

```python
def route_query(query: str) -> dict:
    """Route query to appropriate agent"""
    response = llm.function_call(
        prompt=query,
        functions=[{
            "name": "route_to_agent",
            "parameters": {
                "agent": "technical|sales|support|general",
                "confidence": "float 0-1"
            }
        }]
    )
    
    return response
```

#### 2. Specialized Agents
- Each agent has dedicated vector DB index
- Independent prompt templates
- Domain-specific evaluation metrics

#### 3. Response Aggregator
- Combines responses if multiple agents triggered
- Adds source attribution
- Filters for consistency

### Data Model

```python
class AgentKnowledgeBase:
    agent_id: str
    vector_index: str
    documents: List[Document]
    embedding_model: str
    chunk_size: int
    chunk_overlap: int

class Query:
    id: str
    user_id: str
    text: str
    timestamp: datetime
    metadata: dict

class Response:
    query_id: str
    agent_id: str
    text: str
    sources: List[Source]
    confidence: float
    latency_ms: int
```

### API Design

```python
POST /api/v1/query
Request:
{
    "query": "How do I reset my password?",
    "user_id": "user-123",
    "context": {
        "previous_queries": []
    }
}

Response:
{
    "response": "To reset your password...",
    "agent": "support",
    "confidence": 0.92,
    "sources": [
        {
            "title": "Password Reset Guide",
            "url": "https://..."
        }
    ],
    "latency_ms": 847
}
```

### Deployment
- Containerized services on Kubernetes
- Auto-scaling based on request volume
- Blue-green deployment for zero downtime

### Monitoring
- Query latency (p50, p95, p99)
- Routing accuracy
- Per-agent performance
- Cost per query
- Error rates

### Testing Strategy
1. Unit tests for each component
2. Integration tests for end-to-end flow
3. Load testing (1000 concurrent users)
4. A/B testing for routing accuracy

### Migration Plan
1. **Phase 1** (Week 1-2): Build routing layer
2. **Phase 2** (Week 3-4): Migrate agents one-by-one
3. **Phase 3** (Week 5): A/B test with 10% traffic
4. **Phase 4** (Week 6): Full rollout

### Risks & Mitigations

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Poor routing accuracy | High | Medium | Extensive eval suite, fallback to general |
| Increased latency | High | Low | Aggressive caching, async processing |
| Higher costs | Medium | High | Cost monitoring, usage quotas |
| Complex maintenance | Medium | Medium | Good documentation, monitoring |

## Alternatives Considered

### Single KB with metadata filtering
**Rejected**: Doesn't scale well, poor separation of concerns

### Manual routing rules
**Rejected**: Not flexible, high maintenance

## Open Questions
- How to handle queries spanning multiple domains?
- Should we cache routing decisions?
- How often to retrain routing model?

## Timeline
- Design Review: Jan 20
- Implementation: Jan 22 - Feb 16
- Testing: Feb 19 - Feb 23
- Launch: Feb 26

## Reviewers
- @tech-lead (architecture)
- @ai-lead (AI approach)
- @sre-lead (infrastructure)

## References
- [Multi-agent systems research](link)
- [RAG best practices](link)
```

## Code Documentation

### Docstring Best Practices

```python
def generate_embeddings(
    texts: List[str],
    model: str = "text-embedding-3-small",
    batch_size: int = 100
) -> List[List[float]]:
    """Generate embeddings for a list of texts.
    
    This function batches texts to avoid rate limits and handles
    retries for transient failures. Embeddings are normalized
    to unit length for cosine similarity.
    
    Args:
        texts: List of text strings to embed. Each string should
            be preprocessed (lowercased, trimmed, etc.).
        model: OpenAI embedding model to use. Defaults to the
            cost-effective text-embedding-3-small model.
        batch_size: Number of texts to process per API call.
            OpenAI allows up to 2048 texts per call, but smaller
            batches reduce memory usage and improve error recovery.
    
    Returns:
        List of embedding vectors, one per input text. Each vector
        is a list of floats with length determined by the model
        (1536 for text-embedding-3-small).
    
    Raises:
        ValueError: If texts is empty or contains non-string elements.
        openai.RateLimitError: If rate limit exceeded after retries.
        openai.APIError: If API returns an error after retries.
    
    Example:
        >>> texts = ["Hello world", "AI is amazing"]
        >>> embeddings = generate_embeddings(texts)
        >>> len(embeddings)
        2
        >>> len(embeddings[0])
        1536
    
    Note:
        This function makes external API calls and may take several
        seconds for large text lists. Consider caching results.
    
    See Also:
        - retrieve_similar: Use embeddings for similarity search
        - cache_embeddings: Cache embeddings to avoid recomputation
    """
    if not texts or not all(isinstance(t, str) for t in texts):
        raise ValueError("texts must be a non-empty list of strings")
    
    # Implementation...
```

### Inline Comments

```python
# Good: Explain WHY, not WHAT
# Use exponential backoff because API rate limits reset after 60s
retry_delay = 2 ** attempt

# Bad: States the obvious
# Set retry delay to 2 to the power of attempt
retry_delay = 2 ** attempt

# Good: Explain complex logic
# Normalize embeddings to unit length for cosine similarity.
# This allows us to use dot product instead of cosine calculation,
# which is 3x faster in numpy.
normalized = embedding / np.linalg.norm(embedding)

# Good: Warning about gotchas
# WARNING: Changing chunk_size requires re-indexing entire knowledge base.
# See docs/reindexing.md before modifying.
CHUNK_SIZE = 512
```

## README Files

### Repository README Template

```markdown
# AI Chat Application

Production-ready AI chatbot with RAG, multi-agent routing, and real-time streaming.

## Features

- 🤖 **Multi-Agent Routing**: Specialized agents for different domains
- 📚 **RAG System**: Retrieval-augmented generation with vector search
- ⚡ **Real-time Streaming**: Server-sent events for responsive UX
- 🔒 **Security**: Authentication, rate limiting, PII detection
- 📊 **Monitoring**: Comprehensive metrics and alerting

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- PostgreSQL 15+
- Redis 7+

### Installation

```bash
# Clone repository
git clone https://github.com/org/ai-chat.git
cd ai-chat

# Install dependencies
pip install -r requirements.txt
npm install

# Set up environment
cp .env.example .env
# Edit .env with your API keys

# Run database migrations
alembic upgrade head

# Start services
docker-compose up -d  # Redis, PostgreSQL
python -m uvicorn app.main:app --reload
```

### Usage

```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/query",
    headers={"Authorization": "Bearer YOUR_TOKEN"},
    json={"query": "How do I reset my password?"}
)

print(response.json())
```

## Architecture

See [docs/architecture.md](docs/architecture.md) for detailed architecture.

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │
┌──────▼──────────┐
│   API Gateway   │
└──────┬──────────┘
       │
┌──────▼──────────┐
│ Query Router    │
└──────┬──────────┘
       │
   ┌───┴────┬──────────┬─────────┐
   │        │          │         │
┌──▼──┐ ┌──▼──┐ ┌────▼──┐ ┌────▼──┐
│Tech │ │Sales│ │Support│ │General│
│Agent│ │Agent│ │ Agent │ │ Agent │
└─────┘ └─────┘ └───────┘ └───────┘
```

## Development

### Running Tests

```bash
# Unit tests
pytest tests/unit

# Integration tests
pytest tests/integration

# Load tests
locust -f tests/load/locustfile.py
```

### Code Style

```bash
# Format code
black .
isort .

# Lint
flake8 .
mypy .

# Security scan
bandit -r src/
```

## Deployment

### Docker

```bash
docker build -t ai-chat:latest .
docker run -p 8000:8000 ai-chat:latest
```

### Kubernetes

```bash
kubectl apply -f k8s/
```

See [docs/deployment.md](docs/deployment.md) for details.

## Configuration

| Environment Variable | Description | Required | Default |
|---------------------|-------------|----------|---------|
| `OPENAI_API_KEY` | OpenAI API key | Yes | - |
| `DATABASE_URL` | PostgreSQL connection | Yes | - |
| `REDIS_URL` | Redis connection | No | `redis://localhost` |
| `LOG_LEVEL` | Logging level | No | `INFO` |

## Monitoring

- **Metrics**: Prometheus metrics at `/metrics`
- **Health**: Health check at `/health`
- **Logs**: Structured JSON logs to stdout

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing`)
3. Commit changes (`git commit -am 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing`)
5. Create Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Support

- 📧 Email: support@example.com
- 💬 Slack: [#ai-chat](https://workspace.slack.com)
- 📖 Docs: [docs.example.com](https://docs.example.com)
```

## Pull Request Descriptions

### Good PR Description Template

```markdown
## Summary
Implement multi-agent routing for specialized query handling

## Motivation
Currently, all queries go to a single agent, leading to poor responses for specialized domains. This PR adds intelligent routing to domain-specific agents.

## Changes
- Add QueryRouter class for agent selection
- Implement 4 specialized agents (technical, sales, support, general)
- Add routing confidence scoring
- Update API to return agent metadata
- Add routing accuracy metrics

## Testing
- Unit tests for QueryRouter (95% coverage)
- Integration tests for end-to-end routing
- Manual testing with 50 diverse queries
- Load testing: handles 1000 req/s

## Performance Impact
- Latency: +120ms avg (acceptable per design doc)
- Cost: +$0.02 per query (routing call)
- Accuracy: 92% correct routing (vs 85% baseline)

## Deployment Notes
- Requires new environment variables (see .env.example)
- Database migration needed: `alembic upgrade head`
- Backward compatible: old API still works

## Screenshots
[Screenshot of routing dashboard]

## Checklist
- [x] Tests pass locally
- [x] Code follows style guide
- [x] Documentation updated
- [x] Reviewed by @tech-lead
- [x] Security review completed
- [x] Performance benchmarks run

## Related
- Closes #123
- Related to ADR-005
- Builds on PR #456
```

## Presentation Skills

### Technical Presentation Structure

1. **Executive Summary** (1 slide)
   - What we're building
   - Why it matters
   - Key decision

2. **Problem** (1-2 slides)
   - Current pain points
   - User impact
   - Business impact

3. **Proposed Solution** (2-3 slides)
   - Architecture diagram
   - Key components
   - How it works

4. **Alternatives** (1 slide)
   - What else we considered
   - Why we chose this approach

5. **Trade-offs** (1 slide)
   - Benefits
   - Costs/risks
   - Mitigations

6. **Implementation** (1-2 slides)
   - Timeline
   - Team responsibilities
   - Milestones

7. **Success Metrics** (1 slide)
   - How we'll measure success
   - Target metrics

8. **Q&A** (remaining time)

### Tips for Technical Presentations

- **Start with "why"**: Explain the problem before the solution
- **Use visuals**: Diagrams > bullets > text
- **Tell a story**: Make it memorable
- **Anticipate questions**: Prepare backup slides
- **Practice**: Rehearse out loud
- **Know your audience**: Adjust detail level
- **Time management**: Leave room for questions

## Communication Best Practices

### Written Communication

1. **Be Clear**: Use simple language
2. **Be Concise**: Respect reader's time
3. **Be Specific**: Provide examples
4. **Be Structured**: Use headings, lists, tables
5. **Be Visual**: Add diagrams when helpful

### Verbal Communication

1. **Listen First**: Understand before responding
2. **Check Understanding**: "Does that make sense?"
3. **Provide Context**: Don't assume knowledge
4. **Be Open**: Welcome feedback and questions
5. **Follow Up**: Document decisions

### Async Communication (Slack, Email)

```markdown
# Good async message template

**Context**: We're seeing 5% error rate on /chat endpoint

**Problem**: Users experiencing failed requests during peak hours

**Analysis**:
- Error happens when vector DB latency > 5s
- Affects ~500 users per hour during peak
- Started after yesterday's deploy

**Proposal**: 
- Add timeout handling (2s timeout)
- Implement fallback to cached responses
- Scale up vector DB

**Decision Needed**: 
Which approach to prioritize?

**Timeline**: Need to decide by EOD for tomorrow's fix

**Cc**: @oncall-eng @tech-lead
```

## Resources

- "The Documentation System" - Divio
- "Writing Well" - Julian Shapiro
- "Effective Technical Writing" - Google
- "Architecture Decision Records" - Michael Nygard
- "Staff Engineer" - Will Larson (communication chapter)
