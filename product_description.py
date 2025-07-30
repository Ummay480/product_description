import os
from dotenv import load_dotenv
import chainlit as cl
from openai import OpenAI

# Load environment variables
load_dotenv()

# Get Gemini API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Set up Gemini-compatible OpenAI client
client = OpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Optional greeting on app start
@cl.on_chat_start
async def greet():
    await cl.Message(content="👋 Welcome! Enter your product name and I'll write a description for you.").send()

# Message handler
@cl.on_message
async def generate_product_description(message: cl.Message):
    user_input = message.content

    # Chat completion call
    response = client.chat.completions.create(
        model="gemini-1.5-flash",  # You may change to gemini-pro if required
        messages=[
            {"role": "system", "content": "You are a creative assistant that writes compelling e-commerce product descriptions."},
            {"role": "user", "content": f"Generate a product description for: {user_input}"}
        ],
        temperature=0.7,
        max_tokens=300
    )

    # Extract content
    description = response.choices[0].message.content

    # Send back the response
    await cl.Message(content=description).send()
