# PART 3 — Agentic AI — The Frontier

## The biggest shift in AI since ChatGPT. LLMs that don't just answer — they act.

[← Part 2](./PART2-GenerativeAI.md) | [Back to Home](./README.md) | [Part 4 →](./PART4-Production.md)

---

> **The mindset shift:** An LLM answers a question. An agent solves a problem. An LLM gives you the recipe. An agent goes to the store, buys ingredients, cooks the meal, and serves it.

---

## What You Will Learn in Part 3

- [1. What Makes an AI Agent](#1-what-makes-an-ai-agent)
- [2. Prompt Engineering for Agents](#2-prompt-engineering-for-agents)
- [3. RAG — Retrieval Augmented Generation](#3-rag--retrieval-augmented-generation)
- [4. Agent Frameworks in 2026](#4-agent-frameworks-in-2026)
- [5. Multi-Agent Systems](#5-multi-agent-systems)
- [6. Agent Safety and Guardrails](#6-agent-safety-and-guardrails)

---

## 1. What Makes an AI Agent

### What is an AI Agent?

> An **AI agent** is an LLM that can perceive inputs, reason about them, decide what actions to take, execute those actions using tools, observe results, and iterate — all to accomplish a goal.

**LLM vs Agent — the critical difference:**

```
LLM (Question-Answer):
  User: "How do I find duplicates in a Python list?"
  LLM:  "Here is code: [shows code snippet]"
  Done. One response.

Agent (Goal-Oriented):
  User: "Find all duplicate values in my CSV file and email me a report"
  Agent:
    Step 1: Read CSV file using file tool
    Step 2: Analyze for duplicates using Python REPL
    Step 3: Generate report using LLM
    Step 4: Send email using email API
    Done. Multi-step autonomous execution.
```

---

### The 5 Components of an Agent

#### Component 1: Memory

```
Short-term memory  = The context window (what the agent currently sees)
                     Limit: 128K to 1M tokens
                     Lost when conversation ends

Long-term memory   = Vector database (Pinecone, ChromaDB, FAISS)
                     Persists across sessions
                     Retrieved semantically: "what is relevant to this query?"
```

#### Component 2: Tools

Tools are Python functions the agent can call. The agent decides when to use them.

```python
from langchain.tools import Tool

tools = [
    Tool(
        name="web_search",
        description="Search the web for current information. Use when you need facts you do not know.",
        func=tavily_search
    ),
    Tool(
        name="python_repl",
        description="Execute Python code. Use for calculations, data analysis, or creating files.",
        func=python_executor
    ),
    Tool(
        name="send_email",
        description="Send an email. Use when user asks to email something.",
        func=email_sender
    )
]
```

**Common tool types:**
- Web search: Tavily, Serper, DuckDuckGo
- Code execution: Python REPL, sandbox environments
- File operations: read/write PDFs, CSVs, Word docs
- Database queries: SQL, vector search
- API calls: weather, stocks, calendar
- Browser control: Playwright, Selenium

#### Component 3: Planning and Reasoning

```
Goal: "Research the top 3 AI companies, compare their valuations, create a spreadsheet"

Agent plan:
  1. Search: "top AI companies by valuation 2026"
  2. Search: "Anthropic valuation 2026"
  3. Search: "OpenAI valuation 2026"
  4. Search: "xAI valuation 2026"
  5. Extract numbers from results
  6. Write Python code to create a spreadsheet
  7. Execute the code
  8. Return file path to user
```

#### Component 4: Reflection and Self-Correction

```python
reflection_prompt = """
You generated this code:
{code}

It produced this output:
{output}

The user asked for: {user_request}

Does the output satisfy the request?
If yes: Return DONE
If no: What went wrong? How would you fix it?
"""
```

#### Component 5: Perception

Agents in 2026 are multimodal — they receive:
- Text (always)
- Images: GPT-4o, Claude 3.5 Sonnet, Gemini
- Documents and PDFs via file tools
- Web pages via browser tools
- Audio for voice agents
- Structured data: CSV, JSON

---

## 2. Prompt Engineering for Agents

### The ReAct Pattern — Reason and Act

**ReAct** (Reasoning + Acting) is the standard prompting pattern. The model alternates between thinking and doing.

```
User: "What is the current population of India vs 2020?"

[Thought] I need current population data. I should search for this.
[Action] web_search("India population 2026")
[Observation] "India population in 2026 is approximately 1.45 billion"

[Thought] Now I need 2020 data for comparison.
[Action] web_search("India population 2020")
[Observation] "India population in 2020 was approximately 1.38 billion"

[Thought] I have both numbers. I can calculate and answer.
[Final Answer] India grew from 1.38B in 2020 to 1.45B in 2026,
an increase of 70 million people (5% growth over 6 years).
```

### Chain-of-Thought for Agents

```python
system_prompt = """You are a helpful research assistant.

When faced with a complex question:
1. Break it into smaller sub-questions
2. Answer each sub-question using tools if needed
3. Combine the answers into a coherent response
4. Check your work before responding

Always think before acting. Show your reasoning."""
```

### Tree-of-Thought for Complex Problems

```
Problem: "Design the best architecture for a customer support AI agent"

Branch 1: RAG-based approach
  Sub-branch 1a: ChromaDB for documents
  Sub-branch 1b: Pinecone for production scale

Branch 2: Fine-tuned model approach
  Sub-branch 2a: Full fine-tuning
  Sub-branch 2b: LoRA adaptation

Branch 3: Hybrid approach
  Sub-branch 3a: RAG + small fine-tuned model

[Evaluate all branches and select the best]
```

---

## 3. RAG — Retrieval Augmented Generation

### What is RAG and Why Does it Exist?

> LLMs have a knowledge cutoff — they only know what was in their training data. RAG solves this by letting them retrieve fresh, specific information at query time.

**The problem RAG solves:**

```
Without RAG:
User: "What were our Q4 2025 sales figures?"
LLM:  "I do not have access to your internal sales data." OR hallucinates a number.

With RAG:
User: "What were our Q4 2025 sales figures?"
System: [retrieves Q4 2025 sales report from company database]
LLM:  "Based on your Q4 2025 report, sales were $42M, up 15% from Q3." (accurate + cited)
```

---

### Standard RAG Pipeline

```
INDEXING PHASE (done once, offline)

1. LOAD DOCUMENTS
   PDFs, Word docs, web pages, databases, CSVs

2. CHUNK DOCUMENTS
   Split large documents into smaller overlapping pieces
   Common: 500 character chunks with 50 character overlap

3. EMBED CHUNKS
   Each chunk -> vector (list of 768-1536 numbers)
   Models: text-embedding-3-small, all-MiniLM-L6-v2

4. STORE IN VECTOR DATABASE
   Chunk text + embedding + metadata -> FAISS, Pinecone, ChromaDB

QUERYING PHASE (done for each user query)

5. EMBED USER QUERY
   "What is the return policy?" -> vector

6. SIMILARITY SEARCH
   Find top-5 chunks most similar to query vector

7. INJECT INTO PROMPT
   prompt = f"""
   Answer based on these documents: {retrieved_chunks}
   Question: {user_query}
   Cite your sources.
   """

8. LLM GENERATES GROUNDED ANSWER
```

---

### Vector Databases

| Database | Type | Best for |
|---|---|---|
| FAISS | Local library | Prototyping, small datasets |
| ChromaDB | Local or cloud | Easy setup, development |
| Pinecone | Fully managed cloud | Production scale |
| Qdrant | Self-hosted or cloud | Open source production |
| Weaviate | Self-hosted or cloud | Enterprise, hybrid search |
| Milvus | Self-hosted | Very large scale |

**When to use which:**
```
Prototype/hackathon  -> ChromaDB (easiest setup)
Production startup   -> Qdrant or Pinecone
Enterprise at scale  -> Weaviate or Milvus
CPU-only/offline     -> FAISS
```

---

### Advanced RAG Techniques (2025-2026)

#### Query Rewriting

```python
rewrite_prompt = """
Original query: "what was that revenue thing from last quarter"
Rewrite as a clear, specific search query:
"""
# Output: "Q3 2025 quarterly revenue figures"
```

#### HyDE — Hypothetical Document Embeddings

Instead of embedding the question, generate a hypothetical answer and embed that.

```
Query: "What causes inflation?"

HyDE: Generate hypothetical answer first:
"Inflation is caused by excess money supply relative to goods,
demand-pull factors, cost-push factors..."

Embed this hypothetical answer -> search for similar real documents

Why it works: The hypothetical answer is in "document space" not "question space"
-- finds much better matches than embedding the question directly
```

#### Re-Ranking with Cross-Encoders

```
Standard bi-encoder:
  Embed query + embed all docs separately -> fast but approximate

Re-ranking cross-encoder:
  Take query + each retrieved doc TOGETHER
  Score each (query, doc) pair for relevance
  Re-rank top 20 results -> keep top 5
  
Cost: Slower
Benefit: Much better precision
```

#### Hybrid Search

```
Semantic search (vector):  Finds contextually similar content
Keyword search (BM25):     Finds exact keyword matches

User asks "CUDA OOM error" ->
  Semantic: finds pages about GPU memory management
  BM25:     finds exact occurrences of "CUDA" and "OOM"
  Hybrid:   combines both -> best results
```

#### Agentic RAG

```python
class AgenticRAG:
    def answer(self, query):
        # Agent decides IF retrieval is needed
        needs_retrieval = self.classify(f"Does '{query}' require specific documents? YES/NO")
        
        if needs_retrieval == "YES":
            search_query = self.rewrite_query(query)
            docs = self.vector_db.search(search_query)
            
            if self.is_sufficient(docs, query):
                return self.generate(query, docs)
            else:
                # Fallback to web search
                web_docs = self.web_search(query)
                return self.generate(query, docs + web_docs)
        else:
            return self.generate(query)
```

#### GraphRAG (Microsoft, 2024)

```
Standard RAG: Find similar text chunks
GraphRAG:     Build a knowledge graph over documents

Documents -> Extract entities and relationships -> Build graph

"Microsoft" -- acquired --> "GitHub"
"GitHub" -- hosts --> "Copilot"
"Copilot" -- uses --> "OpenAI GPT-4"

Query: "What AI companies has Microsoft acquired?"
GraphRAG: Traverses graph, finds all acquisition relationships
Standard RAG: Only finds chunks with "Microsoft" and "acquired" near each other
```

---

## 4. Agent Frameworks in 2026

### LangChain — The Foundation

```python
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# LCEL chain
chain = (
    ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("human", "{input}")
    ])
    | ChatAnthropic(model="claude-3-5-sonnet-20241022")
    | StrOutputParser()
)

response = chain.invoke({"input": "What is RAG?"})
```

### LangGraph — Agents as State Machines

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

# Define the state
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    next_action: str
    final_answer: str

# Define nodes
def router(state: AgentState):
    if needs_search(state["messages"][-1]):
        return {"next_action": "search"}
    return {"next_action": "generate"}

def search_node(state: AgentState):
    results = tavily_search(state["messages"][-1])
    return {"messages": [f"Search results: {results}"]}

def generate_node(state: AgentState):
    answer = llm.invoke(state["messages"])
    return {"final_answer": answer.content}

# Build the graph
workflow = StateGraph(AgentState)
workflow.add_node("router", router)
workflow.add_node("search", search_node)
workflow.add_node("generate", generate_node)
workflow.set_entry_point("router")
workflow.add_conditional_edges("router",
    lambda state: state["next_action"],
    {"search": "search", "generate": "generate"}
)
workflow.add_edge("search", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()
result = app.invoke({"messages": ["What is the latest AI news?"]})
```

**LangGraph killer features:**
- Human-in-the-loop interrupts
- Checkpointing to resume from any point
- Conditional routing between nodes
- Built-in state management

### CrewAI — Role-Based Multi-Agent

```python
from crewai import Agent, Task, Crew, Process

researcher = Agent(
    role="Senior Research Analyst",
    goal="Find accurate, comprehensive information on any topic",
    backstory="You are a veteran researcher who digs deep and cross-references sources.",
    tools=[web_search_tool, wikipedia_tool],
    llm=claude_llm
)

writer = Agent(
    role="Content Writer",
    goal="Write clear, engaging, well-structured articles",
    backstory="You are an award-winning science writer who makes complex topics accessible.",
    llm=claude_llm
)

research_task = Task(
    description="Research the top 5 AI breakthroughs in 2025",
    expected_output="A detailed report with sources",
    agent=researcher
)

writing_task = Task(
    description="Write a 1000-word article based on the research",
    expected_output="A compelling article ready for publication",
    agent=writer,
    context=[research_task]
)

crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    process=Process.sequential
)

result = crew.kickoff()
```

---

## 5. Multi-Agent Systems

### Why Multiple Agents?

```
Single Agent (Jack of all trades):
  "Research, write, review, optimize SEO, and post the article"
  Hard to do all well simultaneously

Multi-Agent (team of specialists):
  Researcher     -> finds facts and sources
  Writer         -> crafts engaging narrative
  Critic         -> fact-checks and reviews
  SEO Optimizer  -> adds keywords
  Publisher      -> posts to CMS
  Each agent excels at its specific role
```

### Supervisor-Worker Pattern (Most Common)

```
Supervisor Agent
  Receives goal, plans, delegates, aggregates
        |
    delegates to
        |
Researcher  |  Writer  |  Reviewer
  Agent     |  Agent   |  Agent
```

### Shared State Between Agents

```python
class TeamState(TypedDict):
    original_goal: str
    research_notes: list[str]    # Researcher writes here
    draft_content: str           # Writer reads/writes here
    review_feedback: list[str]   # Reviewer writes here
    final_output: str

def researcher_agent(state: TeamState):
    results = search(state["original_goal"])
    return {"research_notes": state["research_notes"] + [results]}
```

---

## 6. Agent Safety and Guardrails

### Why Agent Safety Matters

> A bad search result is annoying. A bad agent with access to email, files, and databases can send secrets, delete data, or take irreversible actions. Guardrails are not optional.

### Layer 1 — Input Guardrails

```python
def input_guard(user_query: str) -> dict:
    # Check for PII
    pii_results = presidio_analyzer.analyze(text=user_query, language="en")
    if pii_results:
        return {"safe": False, "reason": "PII detected"}
    
    # Check for harmful content
    harmful_patterns = [
        r"ignore (all )?(previous|above) instructions",  # Prompt injection
        r"(how to|help me) (make|build|create) (a )?(bomb|weapon|virus)"
    ]
    for pattern in harmful_patterns:
        if re.search(pattern, user_query.lower()):
            return {"safe": False, "reason": "Harmful query detected"}
    
    return {"safe": True, "reason": ""}
```

### Layer 2 — Output Guardrails

```python
def output_guard(answer: str, retrieved_context: str) -> dict:
    # Check faithfulness
    faithfulness_score = evaluate_faithfulness(answer, retrieved_context)
    
    if faithfulness_score < 0.6:
        return {
            "approved": True,
            "warning": "Low confidence answer — please verify independently"
        }
    
    # Check for PII leak
    if presidio_analyzer.analyze(text=answer, language="en"):
        answer = redact_pii(answer)
    
    # Self-consistency check
    answer2 = llm.invoke(original_prompt)
    if cosine_similarity(embed(answer), embed(answer2)) < 0.85:
        return {
            "approved": True,
            "warning": "Inconsistent answers detected — low reliability"
        }
    
    return {"approved": True, "warning": ""}
```

### Layer 3 — Fallback Mechanisms

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def call_llm_with_retry(prompt: str) -> str:
    return llm.invoke(prompt)

def get_answer(query: str) -> str:
    try:
        context = vector_db.search(query)
        if context:
            return generate_from_context(query, context)
    except Exception:
        pass
    
    try:
        web_results = tavily_search(query)
        return generate_from_context(query, web_results)
    except Exception:
        pass
    
    return "I could not find reliable information. Please rephrase or consult a specialist."
```

### Layer 4 — Human-in-the-Loop

```python
HIGH_STAKES_ACTIONS = ["send_email", "delete_file", "make_payment", "post_to_social_media"]

def execute_tool(tool_name: str, tool_args: dict) -> str:
    if tool_name in HIGH_STAKES_ACTIONS:
        approval = request_human_approval(action=tool_name, args=tool_args)
        if not approval:
            return f"Action '{tool_name}' was not approved. Aborting."
    return tools[tool_name](**tool_args)
```

### Layer 5 — Observability with LangSmith

```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-key"
os.environ["LANGCHAIN_PROJECT"] = "production-agent"

# Every LangChain/LangGraph call is now automatically traced
# In LangSmith dashboard you see:
#   - Which nodes ran and in what order
#   - Input and output of each node
#   - Token counts per step
#   - Latency per step
#   - Total cost per run
#   - Full error traces
```

### Evaluating Agents with RAGAS

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall

results = evaluate(
    dataset=test_cases,
    metrics=[faithfulness, answer_relevancy, context_recall]
)

# Target scores:
# faithfulness     > 0.85  (answers are grounded in retrieved content)
# answer_relevancy > 0.90  (answers the question that was asked)
# context_recall   > 0.80  (retrieved the right content)
```

---

## Part 3 Summary

You now understand:

- What AI agents are and the 5 components that make them
- ReAct, CoT, and Tree-of-Thought prompt patterns for agents
- The full RAG pipeline: loading, chunking, embedding, storing, retrieving
- Advanced RAG: HyDE, re-ranking, hybrid search, GraphRAG, Agentic RAG
- LangChain, LangGraph, CrewAI, AutoGen — when to use each
- Multi-agent architectures: supervisor, pipeline, parallel
- 5-layer safety system: input guardrails, output guardrails, fallbacks, HITL, observability
- RAGAS evaluation — how to measure agent quality objectively

---

**You know how to build agents. Now let us talk about deploying them.**

## Continue to Part 4: Production AI

[PART4-Production.md](./PART4-Production.md)

*"Building an agent is one thing. Making it reliable, scalable, and cost-efficient in production is another."*

[Part 2](./PART2-GenerativeAI.md) | [Back to Home](./README.md) | [Part 4](./PART4-Production.md)
