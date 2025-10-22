from langchain.agents import AgentExecutor, create_react_agent, Tool
from langchain_openai import ChatOpenAI
from langchain import hub
from dotenv import load_dotenv,find_dotenv

load_dotenv(find_dotenv())

def save_file_tool(content: str):
    with open("story.md", "w", encoding="utf-8") as f:
        f.write(content)
    return "文件 story.md 已保存！"

tools = [
    Tool(
        name="SaveMarkdownFile",
        func=save_file_tool,
        description=(
            "将提供的完整 Markdown 文本内容保存为工作目录下的 story.md 文件。"
            "输入应为要保存的完整内容（字符串）。调用此工具会覆盖已有的 story.md。"
        ),
    )
]
llm = ChatOpenAI(model="gpt-4o-mini")

# Get the prompt to use
prompt = hub.pull("hwchase17/react")
# Create the agent
agent = create_react_agent(llm, tools, prompt)
# Create an agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools)

result = agent_executor.invoke({
    "input": (
        "写一个关于太空旅行的完整故事（使用 Markdown，包含标题与小节）。"
        "生成后，将“完整故事文本”作为工具输入保存为 story.md。"
    )
})
print(result["output"])
print(result)
