import os
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from decouple import config
from agents import TravelAgents
from textwrap import dedent
from tasks import TravelTasks


from dotenv import load_dotenv
load_dotenv()
# Install duckduckgo-search for this example:
# !pip install -U duckduckgo-search

from langchain_community.tools import DuckDuckGoSearchRun

search_tool = DuckDuckGoSearchRun()

os.environ["OPENAI_API_KEY"] = config("OPENAI_API_KEY")
os.environ["OPENAI_ORGANIZATION"] = config("OPENAI_ORGANIZATION_ID")

# This is the main class that you will use to define your custom crew.
    # You can define as many agents and tasks as you want in agents.py and tasks.py




class TripCrew:
    def __init__(self, origin, cities,date_range, interests):
        self.origin = origin
        self.cities = cities
        self.date_range = date_range
        self.interests = interests
        

    def run(self):
        # Define your custom agents and tasks in agents.py and tasks.py
        agents = TravelAgents()
        tasks = TravelTasks()

        # Define your custom agents and tasks here
        travelAgent = agents.expert_travel_agent()
        citySelectionAgent = agents.city_selection_expert()
        localTourGuide = agents.local_tour_guide()

        # Custom tasks include agent name and variables as input
        plan_iterinary = tasks.plan_itinerary(
            travelAgent,
            self.cities,
            self.date_range,
            self.interests
        )

        identify_city = tasks.identify_city(
            citySelectionAgent,
            self.origin,
            self.cities,
            self.date_range,
            self.interests
        )


        gather_city_info = tasks.gather_city_info(localTourGuide,self.cities,self.date_range,self.interests)

        # Define your custom crew here
        crew = Crew(
            agents=[travelAgent, citySelectionAgent,localTourGuide],
            tasks=[plan_iterinary, identify_city, gather_city_info],
            verbose=True,
        )

        result = crew.kickoff()
        return result


# This is the main function that you will use to run your custom crew.
if __name__ == "__main__":
    print("## Welcome to Trip Planner Crew")
    print('-------------------------------')
    origin = input(
        dedent("""
      From where will you be traveling from?
    """))
    cities = input(
        dedent("""
      What are the cities options you are interested in visiting?
    """))
    date_range = input(
        dedent("""
      What is the date range you are interested in traveling?
    """))
    interests = input(
        dedent("""
      What are some of your high level interests and hobbies?
    """))

    trip_crew = TripCrew(origin, cities, date_range, interests)
    result = trip_crew.run()
    print("\n\n########################")
    print("## Here is you Trip Plan")
    print("########################\n")
    print(result)