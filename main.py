from dotenv import load_dotenv

load_dotenv()

# Importing to register hub functionality -> hub is used to push/pull prompts
from langchain_classic import hub
from langchain_classic.agents import AgentExecutor
from langchain_classic.agents.react.agent import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
# from langchain_core.output_parsers import PydanticOutputParser
from schemas import AgentOutput


tools = [TavilySearch()]

# The reason we are using gpt-4 and not gpt-5 is that as of June 2024, gpt-5 does not support function calling which is required for the React agent to work properly
llm = ChatOpenAI(model="gpt-4", temperature=0)

# structured output parsing using pydantic models
# Instead of the formatting the output ourselves, we can leverage the pydantic model to define the structure of the output we want from the LLM
# Difference between structured_llm and pydantic output parser is that structured_llm directly integrates with the LLM to produce structured output
structured_llm = llm.with_structured_output(AgentOutput)

react_prompt = hub.pull("hwchase17/react")

# Define the output parser using the Pydantic model -> this will ensure the agent's output adheres to the schema
# output_parser = PydanticOutputParser(pydantic_object=AgentOutput)

# Create a PromptTemplate from the REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS
react_prompt_template = PromptTemplate(template=REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS, input_variables=["input", "tools", "tool_names", "agent_scratchpad", "format_instructions"]).partial(
    format_instructions = ""
)

# Create the React agent with the custom prompt and output parser -> The output parser will validate the output using the schema (pydantic model)
agent = create_react_agent(llm=llm, tools=tools, prompt=react_prompt_template)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Runnable basically turns a function into a runnable component that can be composed into chains,
# here we are extracting the "output" key from the agent executor's output
extract_output = RunnableLambda(
    lambda x: x["output"]
)

# Runnable to parse the output using the Pydantic output parser
# parse_output = RunnableLambda(
#     lambda x: output_parser.parse(x)
# )

# Compose the runnables into a chain
chain = agent_executor | extract_output | structured_llm


def main():
    print("Hello from langchain-course!")
    result = chain.invoke(
        {
            "input": "search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details?",
        }
    )
    print("Type of result:", type(result))
    print(result)


if __name__ == "__main__":
    main()
