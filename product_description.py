import os
import json
import asyncio
from uuid import uuid4
from dataclasses import dataclass, field
from typing import List, Any

from dotenv import load_dotenv
from openai import AsyncOpenAI
from bs4 import BeautifulSoup
import aiohttp
import chainlit as cl
from dapr.clients import DaprClient
from pydantic import BaseModel

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = "gemini-2.0-flash"
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/"
STATE_STORE = "statestore"

# Set up OpenAI client for Gemini
client = AsyncOpenAI(api_key=GEMINI_API_KEY, base_url=BASE_URL)


@dataclass
class Agent:
    name: str
    instructions: str
    model: Any
    handoffs: List = field(default_factory=list)


class Runner:
    @staticmethod
    async def run(starting_agent: Agent, input: str, user_id: str = None) -> dict:
        if starting_agent.name == "input_agent":
            return await input_agent(input, user_id)
        elif starting_agent.name == "competitor_analysis_agent":
            return await competitor_analysis_agent(input, user_id)
        elif starting_agent.name == "description_generator_agent":
            return await description_generator_agent(input, user_id)
        return {"final_output": "No suitable agent found."}


async def input_agent(input: str, user_id: str) -> dict:
    try:
        data = json.loads(input)
        product_details = data.get("product_details", {})
        competitor_urls = data.get("competitor_urls", [])
        tone = data.get("tone", "professional")
        structure = data.get("structure", {
            "intro_sentences": 3,
            "features_count": 5,
            "include_benefits": True,
            "include_faqs": True
        })

        # Save user preferences to Dapr state
        async with DaprClient() as dapr:
            await dapr.save_state(
                store_name=STATE_STORE,
                key=f"user_prefs_{user_id}",
                value=json.dumps({"tone": tone, "structure": structure})
            )

        return await Runner.run(
            Agent(name="competitor_analysis_agent", instructions="Analyze competitor URLs", model=client),
            json.dumps({"urls": competitor_urls, "product_details": product_details}),
            user_id=user_id
        )
    except Exception as e:
        return {"final_output": f"Error processing input: {str(e)}"}


async def competitor_analysis_agent(input: str, user_id: str) -> dict:
    try:
        data = json.loads(input)
        urls = data.get("urls", [])
        product_details = data.get("product_details", {})

        competitor_data = []
        async with aiohttp.ClientSession() as session:
            for url in urls:
                try:
                    async with session.get(url) as response:
                        if response.status == 200:
                            html = await response.text()
                            soup = BeautifulSoup(html, 'html.parser')
                            meta = soup.find('meta', {'name': 'description'})
                            content = meta.get('content', '') if meta else ''
                            images = [img.get('src') for img in soup.find_all('img') if img.get('src')]
                            competitor_data.append({
                                "url": url,
                                "description": content,
                                "images": images[:3]
                            })
                except Exception as e:
                    competitor_data.append({"url": url, "error": str(e)})

        return await Runner.run(
            Agent(name="description_generator_agent", instructions="Generate Shopify-ready description", model=client),
            json.dumps({"competitor_data": competitor_data, "product_details": product_details}),
            user_id=user_id
        )
    except Exception as e:
        return {"final_output": f"Error analyzing competitors: {str(e)}"}


async def description_generator_agent(input: str, user_id: str) -> dict:
    try:
        data = json.loads(input)
        competitor_data = data.get("competitor_data", [])
        product_details = data.get("product_details", {})

        # Load user preferences from Dapr
        async with DaprClient() as dapr:
            state = await dapr.get_state(store_name=STATE_STORE, key=f"user_prefs_{user_id}")
            prefs = json.loads(state.data.decode()) if state.data else {
                "tone": "professional",
                "structure": {"intro_sentences": 3, "features_count": 5}
            }

        tone = prefs.get("tone", "professional")
        structure = prefs.get("structure", {})

        prompt = f"""
        Create a Shopify-ready product description with the following structure:
        - Product Name: Bolded, clear, and concise.
        - Introduction: {structure.get('intro_sentences', 3)} sentences highlighting the product's unique value.
        - Key Features: Bullet-point list of {structure.get('features_count', 5)} features, customer-focused.
        - Benefits: A short paragraph emphasizing the product's advantages and differentiation.
        - FAQs: 3-5 frequently asked questions with concise answers addressing common customer concerns.
        Tone: {tone}
        Product Details: {json.dumps(product_details)}
        Competitor Data: {json.dumps(competitor_data)}
        Improve on competitor descriptions by making it more engaging and SEO-friendly.
        Include a section for competitor image URLs if available.
        Output in Markdown format.
        """

        response = await client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": prompt}]
        )

        description = response.choices[0].message.content

        image_urls = [img for c in competitor_data for img in c.get("images", [])]
        markdown_images = "\n".join(image_urls) if image_urls else ""

        output = f"{description}\n\n## Competitor Image URLs\n{markdown_images}" if markdown_images else description
        return {"final_output": output}

    except Exception as e:
        return {"final_output": f"Error generating description: {str(e)}"}


@cl.on_message
async def handle_message(message: cl.Message):
    user_id = str(uuid4())
    result = await Runner.run(
        starting_agent=Agent(name="input_agent", instructions="Process user input", model=client),
        input=message.content,
        user_id=user_id
    )
    await cl.Message(content=result.get("final_output", "Error processing request")).send()
