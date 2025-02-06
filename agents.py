from crewai import Agent
from textwrap import dedent
from langchain_community.llms import OpenAI, Ollama
from langchain_openai import ChatOpenAI



"""
--Work Backwards from the goal
-- define the captain of the crew who orient the other agents towards the goal
--define the experts the captain need to communicate with and delegat tasks to



Goal : 7 day travel itinerary with detailed per-day plans including budget , packing suggestions and safety tips


Captain/Manager/Boss : 
 --EXpert travel agent


 Employees :
 -City selection expert
 -Local tour guide 

Notes : 
-Agents should be result driven and have a clear goal in mind
-Role is their job titles
-Goals should be actionable
-Backstory should be their resume
"""


# This is an example of how to define custom agents.
# You can define as many agents as you want.
# You can also define custom tasks in tasks.py
class CustomAgents:
    def __init__(self):
        self.OpenAIGPT35 = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
        self.OpenAIGPT4 = ChatOpenAI(model_name="gpt-4", temperature=0.7)
        self.Ollama = Ollama(model="openhermes")

    def expert_travel_agent(self):
        return Agent(
            role="Expert Travel Agent",
            backstory=dedent(f""" EXpert in travel planning and logistics, decades of experience creating travel iterinaries """),
            goal=dedent(f"""Createa a 7 day travel iterinary, 
                        include budget, 
                        packing suggestions and safety tips"""),
            # tools=[tool_1, tool_2],
            allow_delegation=False,
            verbose=True,
            llm=self.OpenAIGPT35,
        )

    def city_selection_expert(self):
        return Agent(
            role="City Selection expert",
            backstory=dedent(f"""EXpert in selecting the best cities based on weather, seasons and travelers interest"""),
            goal=dedent(f"""select the best cities to visit based on weather, season, price and traveler interest"""),
            # tools=[tool_1, tool_2],
            allow_delegation=False,
            verbose=True,
            llm=self.OpenAIGPT35,
        )
    
    def local_tour_guide(self):
        return Agent(
            role="Local Tour guide",
            backstory=dedent(f"""Ton of experience with cities"""),
            goal=dedent(f"""provide the best insights about the city selectd"""),
            # tools=[tool_1, tool_2],
            allow_delegation=False,
            verbose=True,
            llm=self.OpenAIGPT35,
        )