import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain.agents import load_tools
from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI
from dotenv import load_dotenv
from langchain.chains.conversation.memory import ConversationalBufferWindowMemory
from dotenv import load_dotenv
from vn_healthcare import vn_healthcare_bot

load_dotenv()

llm = OpenAI()
vn_healthbot = vn_healthcare_bot
tools = load_tools(["google-search", "llm-math"], llm=llm)
tools.append(Tool(name="Vietnamese Healthcare Chatbot",
                  func=vn_healthbot.run,
                  description="must use when you need to answer questions about healthcare in Vietnamese(dont paraphase the output)"))
memory = ConversationalBufferWindowMemory(k=4, memory_key="chat_history")
agent = initialize_agent(tools, llm, agent="conversational-react-description", verbose=True, memory=memory)
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    question: str
    buffer: str

@app.post("/chat")
def chat(json_payload: ChatMessage):
    curr_buffer = json_payload.buffer
    memory.buffer.append(curr_buffer)
    input = json_payload.question
    output = agent.run(input)
    return {"chatMessage": output, "currBuffer" : memory.buffer}
