from crewai import Agent
from textwrap import dedent
from langchain_community.llms import OpenAI
from langchain_ollama import OllamaLLM

from langchain_openai import ChatOpenAI
from Tools.search_tool import SearchTools
from Tools.calculator_tool import CalculatorTools
"""
Creating Agents Cheat Sheet:
- Think like a boss. Work backwards from the goal and think which employee 
    you need to hire to get the job done.
- Define the Captain of the crew who orient the other agents towards the goal. 
- Define which experts the captain needs to communicate with and delegate tasks to.
    Build a top down structure of the crew.

Goal:
- Create a 7-day travel itinerary with detailed per-day plans,
    including budget, packing suggestions, and safety tips.

Captain/Manager/Boss:
- Expert Travel Agent

Employees/Experts to hire:
- City Selection Expert 
- Local Tour Guide


Notes:
- Agents should be results driven and have a clear goal in mind
- Role is their job title
- Goals should actionable
- Backstory should be their resume
"""

# This is an example of how to define custom agents.
# You can define as many agents as you want.
# You can also define custom tasks in tasks.py
class TravelAgents:
    def __init__(self):
        self.OpenAIGPT35 = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
        self.OpenAIGPT4 = ChatOpenAI(model_name="gpt-4", temperature=0.7)
        self.OllamaLLM = OllamaLLM(model="openhermes")

    def expert_travel_agent(self):
        return Agent(
            role="Expert Travel Agent",
            backstory=dedent(f""" EXpert in travel planning and logistics, decades of experience creating travel iterinaries """),
            goal=dedent(f"""Createa a 7 day travel iterinary, 
                        include budget, 
                        packing suggestions and safety tips"""),
            tools=[SearchTools.search_internet, CalculatorTools.calculate],
            allow_delegation=False,
            verbose=True,
            llm=self.OpenAIGPT4,
        )

    def city_selection_expert(self):
        return Agent(
            role="City Selection expert",
            backstory=dedent(f"""EXpert in selecting the best cities based on weather, seasons and travelers interest"""),
            goal=dedent(f"""select the best cities to visit based on weather, season, price and traveler interest"""),
            tools=[SearchTools.search_internet],
            allow_delegation=False,
            verbose=True,
            llm=self.OpenAIGPT4,
        )
    
    def local_tour_guide(self):
        return Agent(
            role="Local Tour guide",
            backstory=dedent(f"""Ton of experience with cities"""),
            goal=dedent(f"""provide the best insights about the city selectd"""),
            tools=[SearchTools.search_internet],
            allow_delegation=False,
            verbose=True,
            llm=self.OpenAIGPT4,
        )