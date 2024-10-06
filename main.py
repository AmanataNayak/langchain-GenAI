import os
import ssl
from fastapi import FastAPI
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langserve import add_routes
from dotenv import load_dotenv
import aiohttp

# Load environment variables
load_dotenv()
ACCESS_TOKEN = os.getenv('TOKEN')

# Initialize FastAPI app
app = FastAPI(title='LangChain Chatbot', version='1.0')

# Create a custom SSL context that doesn't verify certificates
ssl._create_default_https_context = ssl._create_unverified_context
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE


# Create a custom aiohttp ClientSession with the SSL context
async def get_aiohttp_session():
    return aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ssl_context))


# Initialize the HuggingFace endpoint with the custom session
llm = HuggingFaceEndpoint(
    repo_id='tiiuae/falcon-7b-instruct',
    huggingfacehub_api_token=ACCESS_TOKEN,
    client=get_aiohttp_session
)

# Create a prompt template for chat
prompt = ChatPromptTemplate.from_messages(
    [
        ('system', 'You are a helpful assistant. Please respond to the user queries.'),
        ('user', 'Question: {question}')
    ]
)

# Create a chain
chain = (
        prompt
        | llm
        | StrOutputParser()
)

# Add routes to the FastAPI app
add_routes(
    app,
    chain,
    path="/chat",
)

if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host='0.0.0.0', port=8000)
