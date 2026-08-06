from utils.llm import llm_config
from crewai import Agent, Task, Crew, LLM


async def run_intent_classifier_crew_async(incident_description: str, history: list, interaction: str) -> str:
    
    if not history:
        history = ["NA"]

    llm = LLM(
        model = "openai/" + "/app/models/Qwen3-14B-FP8",
        temperature=0.0,
        base_url= "https://qwen3-14b.iservebetter.idfcfirstbank.com/v1",
        api_key=llm_config.token
    )

    intent_agent = Agent(
        role="Intent Classifier",
        goal="Analyze user input and categorize it into the correct intent category",
        backstory=(
            "You are an expert at classifying user intents in a technical support system. "
            "Your job is to analyze the conversation history and current user input to determine "
            "what the user is trying to accomplish. You are precise and always output only the category name."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm,
        temperature=0
    )

    intent_task = Task(
        description=(
            "Analyze the user input and categorize it.\n\n"
            "Interaction History: \n```\n{history}\n```\n\n"
            "User Input: \n```\n{incident_description}\n```\n\n"
            "Latest Interaction: \n```\n{interaction}\n```\n\n"
            "Depending on history you can figure out latest question if it exists. Catagorise intent on basis of `Latest Interaction` if it is NA, then classify on baisis of `User Input` \n\n"
            "**Categories**:\n\n"
            "\t- **closure**: Greeting, thanks, or ending the chat.\n\n"
            "\t- **rebuttal**: User is disagreeing, correcting the system, or insisting that information they previously provided is correct (e.g., 'I already told you', 'That's wrong', 'This is correct.').\n\n"
            "\t- **additional_info**: Providing IDs, account numbers or subsequent question/information asked.\n\n"
            "**Rule**: If the User Input contradicts the Latest Interaction or expresses frustration with the system's request, it MUST be 'rebuttal'.\n\n"
            "Output ONLY the category name."
        ),
        agent=intent_agent,
        expected_output="Only the category name (closure, rebuttal, or additional_info)"
    )

    crew = Crew(
        agents=[intent_agent],
        tasks=[intent_task],
        verbose=True
    )

    result = await crew.akickoff(
        inputs={
            "incident_description": incident_description,
            "history": history,
            "interaction": interaction
        }
    )
    
    return str(result)
