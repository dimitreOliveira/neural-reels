from google.adk.agents import Agent
from pydantic import BaseModel, Field

MODEL_ID = "gemini-2.5-flash"

SEO_OPTIMIZER_PROMPT = """
# Your role

You are an expert SEO (Search Engine Optimization) specialist for YouTube.
Your task is to generate an engaging video title and a comprehensive video description based on the provided video script.
The title and description should be optimized for discoverability on YouTube, incorporating relevant keywords naturally.

# Your task

Based on the following video script, suggest a concise and catchy video title and a detailed, keyword-rich description.

## Video Script:
{script}

## Output Format:
Provide the output in JSON format, with two keys: "video_title" and "video_description".

# Examples

## Video Script:
This video explains the basics of quantum physics, including wave-particle duality and quantum entanglement. We'll cover the history and key concepts.

## Output:
```json
{{
  "video_title": "Quantum Physics Explained: Wave-Particle Duality & Entanglement Basics",
  "video_description": "Dive into the fascinating world of quantum physics! This video covers the fundamental concepts of quantum mechanics, including wave-particle duality and quantum entanglement. Learn about the history of quantum theory and its core principles. Perfect for beginners and anyone curious about the universe's smallest scales. #QuantumPhysics #QuantumMechanics #WaveParticleDuality #QuantumEntanglement #PhysicsExplained #Science"
}}
```

## Video Script:
Learn how to bake the perfect sourdough bread at home. This tutorial covers starter maintenance, mixing, kneading, proofing, and baking techniques for a crusty loaf.

## Output:
```json
{{
  "video_title": "Bake Perfect Sourdough Bread at Home: Beginner's Guide",
  "video_description": "Master the art of sourdough bread baking with this comprehensive home tutorial! Learn everything from sourdough starter maintenance to mixing, kneading, proofing, and baking techniques. Achieve that perfect crusty loaf every time. Ideal for home bakers and beginners. #SourdoughBread #BakingTutorial #HomemadeBread #SourdoughStarter #BreadBaking #HomeBaking"
}}
```
"""


class SEOOptimizerAgentOutput(BaseModel):
    video_title: str = Field(
        description="A concise and engaging video title optimized for YouTube SEO."
    )
    video_description: str = Field(
        description="A detailed, keyword-rich video description optimized for YouTube SEO."
    )


seo_optimizer_agent = Agent(
    name="SEOOptimizerAgent",
    description="Optimizes video titles and descriptions for YouTube SEO based on the video script.",
    instruction=SEO_OPTIMIZER_PROMPT,
    model=MODEL_ID,
    output_key="seo_optimized_content",
    output_schema=SEOOptimizerAgentOutput,
)
