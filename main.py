import os
import json
from dotenv import load_dotenv
from crewai import Agent, Crew, Process, Task, LLM
from pydantic import BaseModel, Field
from typing import List

# Load environment variables from .env
load_dotenv()

# --- 1. CONFIGURATION ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    if not os.getenv("SERPER_API_KEY"):
         print("WARNING: SERPER_API_KEY not found. Researcher will be limited.")
    raise ValueError("GEMINI_API_KEY not found in .env file. Please set it.")
    
manager_llm = LLM(
    model="gemini/gemini-2.5-flash",
    verbose=True,
    temperature=0.7, # A bit higher temperature for creative delegation
    google_api_key=GEMINI_API_KEY
)

worker_llm = LLM(
    model="gemini/gemini-2.5-flash",
    verbose=True,
    temperature=0.2, # Lower temperature for factual, consistent work
    google_api_key=GEMINI_API_KEY 
)

# --- 2. ADVANCED CONCEPTS: STRUCTURED OUTPUT (Pydantic Schema) ---

class ProjectPlan(BaseModel):
    """A structured plan for the startup simulation project."""
    target_market: str = Field(description="The specific market segment to target (e.g., 'sustainable coffee consumers in Berlin').")
    key_deliverables: List[str] = Field(description="List of 3-5 main deliverables (e.g., 'Market Analysis Report', 'Brand Identity Concept').")
    budget_estimate_usd: float = Field(description="The estimated cost to complete the project in USD, must be a number.")

# --- 3. DEFINING AGENTS ---

# Import the GoogleSearch tool (requires a SERPER_API_KEY in .env)
try:
    from crewai_tools import SerperDevTool
    search_tool = SerperDevTool()
    print("Search Tool Initialized")
except Exception:
    search_tool = None
    print("WARNING: SerperDevTool not available. Researcher will not be able to search the web.")

# --- Manager Agents (allow_delegation=True) ---

ceo_agent = Agent(
    role="CEO & Project Manager",
    goal="Oversee the entire project, ensuring all deliverables meet the goal. Must first output the final structured ProjectPlan.",
    backstory="You are a seasoned CEO, highly skilled at breaking down complex goals into actionable plans and managing your team to deliver high-quality, structured results.",
    llm=manager_llm,
    tools=[], 
    allow_delegation=True,
    verbose=True
)

cto_agent = Agent(
    role="CTO & Technical Architect",
    goal="Translate the CEO's plan into a technical specification, manage the research and writing execution, and ensure tool usage is optimal.",
    backstory="A strategic CTO who excels at engineering workflows and optimizing resource allocation for efficiency and accuracy. You are the main delegator of worker tasks.",
    llm=manager_llm,
    tools=[],
    allow_delegation=True, # The CTO will delegate to the Researcher/Writer
    verbose=True
)

# --- Worker Agents (allow_delegation=False) ---

researcher_agent = Agent(
    role="Market Research Specialist",
    goal="Conduct thorough, up-to-date research on market trends, competitor analysis, consumer needs, **and sustainable coffee sourcing/supply chain logistics**.",
    backstory="A fast, efficient researcher using modern search tools to find the most accurate and relevant data.",
    llm=worker_llm, 
    tools=[search_tool] if search_tool else [],
    allow_delegation=False,
    verbose=True
)

writer_agent = Agent(
    role="Report Synthesizer and Writer",
    goal="Consolidate all research findings into a clear, professional, and well-structured final report that is executive-ready.",
    backstory="An expert technical writer who transforms complex data into compelling, easy-to-read business reports.",
    llm=worker_llm,
    tools=[],
    allow_delegation=False,
    verbose=True
)


# --- 4. DEFINING TASKS ---

# Task 1: CEO's Planning Task (Must produce structured output)
task_plan = Task(
    description=(
        "Analyze the user request: '{project_request}'. "
        "Your first and most important step is to generate a comprehensive "
        "ProjectPlan using the provided Pydantic schema (ProjectPlan). This plan must be the *entire* output, "
        "formatted as a JSON object, and saved to the 'project_plan.json' file."
    ),
    expected_output=ProjectPlan.schema_json(indent=2),
    agent=ceo_agent,
    context=[],
    output_file="project_plan.json"
)

# Task 2: CTO's Orchestration and Delegation Task
# This task is given to the CTO, who will internally delegate to the Researcher and Writer.
task_orchestrate_and_write = Task(
    description=(
        "You have the ProjectPlan from the CEO available in your context."
        "Your job is to orchestrate the execution. First, delegate a focused **'Market Research & Sourcing Task'** "
        "to the **Researcher Agent**, specifically instructing them to research competitive pricing, digital strategy examples, "
        "AND sustainable European coffee supply chains. "
        "Once the research is complete (from the Researcher's output), you MUST "
        "**synthesize all findings into a complete, executive-ready final report**. "
        "The report must be professional, use markdown for formatting, and directly address all key deliverables in the plan. "
        "**This final report must be the ENTIRE output of this task, saved to 'final_report.md'.**"
    ),
    expected_output="A professionally written, 10-paragraph minimum, markdown report saved to 'final_report.md'.",
    agent=cto_agent,
    context=[task_plan],
    output_file="final_report.md"
)

# Task 3: Writer's Final Report Task
# task_final_report = Task(
#     description=(
#         "Generate a complete, executive-ready final report based on all gathered research and the original ProjectPlan. "
#         "The report must be professional, use markdown for formatting, and directly address all key deliverables in the plan. "
#         "Use the consolidated research findings provided in your context."
#     ),
#     expected_output="A professionally written, 10-paragraph minimum, markdown report saved to 'final_report.md'.",
#     agent=writer_agent,
#     context=[task_orchestrate], # The context is the CTO's final, consolidated research findings
#     output_file="final_report.md"
# )


# --- 5. CREATING AND RUNNING THE HIERARCHICAL CREW ---

PROJECT_REQUEST = (
    "Develop a **full-spectrum market entry and growth strategy** for a new line "
    "of premium, sustainable coffee capsules in **Berlin and Amsterdam**. The strategy "
    "must include: a comprehensive competitor analysis, a detailed pricing model, "
    "a 12-month digital marketing plan (including content themes), and a **sourcing/supply chain recommendation**."
)


crew = Crew(
    agents=[cto_agent, researcher_agent, writer_agent],
    tasks=[task_plan, task_orchestrate_and_write],
    process=Process.hierarchical, 
    manager_agent=ceo_agent,       # KEY: Explicitly sets the top manager
    manager_llm=manager_llm,
    verbose=True  # High verbosity shows all agent thought processes
)

print("Starting the Autonomous Startup Simulator...")
print("------------------------------------------")
print(f"Project: {PROJECT_REQUEST}")
print("------------------------------------------")

try:
    result = crew.kickoff(inputs={"project_request": PROJECT_REQUEST})
    print("\n\n################################################")
    print("## SIMULATION COMPLETE")
    print("################################################")
    print(result)
except Exception as e:
    print(f"\n\n--- SIMULATION ERROR ---\n{e}")
    print("Check your .env file for the GEMINI_API_KEY, and ensure you have either a SERPER_API_KEY or have removed the search tool.")
