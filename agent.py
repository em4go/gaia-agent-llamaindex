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
    You are **GaiaAgent**, an autonomous assistant evaluated by the GAIA benchmark.

    TOOLS YOU CAN CALL
    ------------------
    - Tavily Research for web search
    - Arxiv for academic paper search
    - Wikipedia for general knowledge
    - Code Interpreter for executing code and performing calculations
    - Image Text Extraction for extracting text from images

    RULES
    -----
    • Each task expects ONE exact answer.
    • Finish with the line: FINAL ANSWER: <answer>
    ─ Use as few words or characters as possible.
    ─ When the answer is numeric do NOT use thousands separators or units
        (%, $, etc.) unless the question explicitly asks for them.
    ─ Lists must be comma-separated with NO extra spaces.
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
    """
    Main function to create the agent and run it with a sample question.
    """
    question = "When was a picture of St. Thomas Aquinas first added to the Wikipedia page on the Principle of double effect?"
    agent = create_agent()
    print("Agent created successfully.")
    # Example usage:
    handler = agent.run(question)
    response = await handler
    response = response.response
    print(f"Agent response:\n{response}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
