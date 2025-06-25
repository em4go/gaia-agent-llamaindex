import os
from dotenv import load_dotenv
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.llms.groq import Groq
from llama_index.tools.arxiv import ArxivToolSpec
from llama_index.tools.wikipedia import WikipediaToolSpec
from llama_index.tools.tavily_research import TavilyToolSpec
from llama_index.tools.code_interpreter import CodeInterpreterToolSpec
from PIL import Image
import pytesseract

load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


def extract_text_from_image(image_path: str) -> str:
    """
    Extract text from an image using OCR library pytesseract (if available).
    Args:
        image_path (str): the path to the image file.
    Returns:
        str: the extracted text from the image, or an error message if OCR fails.
    """
    try:
        image = Image.open(image_path)

        text = pytesseract.image_to_string(image)

        return f"Extracted text from image:\n\n{text}"
    except Exception as e:
        return f"Error extracting text from image: {str(e)}"


def create_agent(llm_model: str = "qwen-qwq-32b"):
    SYSTEM_PROMPT_TEMPLATE = """
    You are a helpful assistant tasked with answering questions using a set of tools. 
    Now, I will ask you a question. Report your thoughts, and finish your answer with the following template: 
    FINAL ANSWER: [YOUR FINAL ANSWER]. 
    YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise. If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise. If you are asked for a comma separated list, Apply the rules above for each element (number or string), ensure there is exactly one space after each comma.
    Your answer should only start with "FINAL ANSWER: ", then follows with the answer. 
    """.strip()
    llm = Groq(model=llm_model)
    arxiv_tools = ArxivToolSpec().to_tool_list()
    wikipedia_tools = WikipediaToolSpec().to_tool_list()
    tavily_tools = TavilyToolSpec(api_key=TAVILY_API_KEY).to_tool_list()
    code_interpreter_tools = CodeInterpreterToolSpec().to_tool_list()

    agent = AgentWorkflow.from_tools_or_functions(
        llm=llm,
        tools_or_functions=[
            *arxiv_tools,
            *wikipedia_tools,
            *tavily_tools,
            *code_interpreter_tools,
            extract_text_from_image,
        ],
        system_prompt=SYSTEM_PROMPT_TEMPLATE,
    )
    return agent


async def main():
    agent = create_agent(llm_model="qwen-qwq-32b")
    question = "What year was Rafa Nadal born?"
    response = await agent.run(user_msg=question)

    # Parse and print final answer
    if isinstance(response, str):
        raw = response
    else:
        raw = str(response)

    if "FINAL ANSWER:" in raw:
        answer = raw.split("FINAL ANSWER:")[-1].strip()
    else:
        answer = raw.strip()

    print(f"\nFinal Answer: {answer}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
