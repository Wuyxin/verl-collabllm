import re
from openai import OpenAI
from pydantic import BaseModel
from typing import Optional

client = OpenAI()

class TagJudgment(BaseModel):
    reasoning: str
    score: float

def llm_judge_reward_function(data_source, solution_str, ground_truth, extra_info=None):
    """
    Reward function that extracts tags and uses LLM judge for evaluation.
    
    Args:
        data_source: Dataset identifier
        solution_str: Model's generated response with <tag>...</tag>
        ground_truth: Expected correct answer/comment
        extra_info: Additional metadata including original_post, user_persona, bag_of_tokens
    
    Returns:
        dict: Contains "score" and "tag_star" for two-stage processing
    """
    
    try:
        # Extract single tag from <tag>...</tag> format
        tag_match = re.search(r'<tag>(.*?)</tag>', solution_str)
        
        if not tag_match:
            return {"score": 0.0, "tag_star": ""}
        
        tag = tag_match.group(1).strip()
        
        # Get context information from extra_info
        original_post = extra_info.get("original_post", "") if extra_info else ""
        user_persona = extra_info.get("user_persona", "") if extra_info else ""
        bag_of_tokens = extra_info.get("bag_of_tokens", []) if extra_info else []
        
        # LLM judge with structured output
        response = client.responses.parse(
            model="gpt-5",
            input=[
                {
                    "role": "system", 
                    "content": "Rate how well the tag connects the post to the comment given the user persona. Provide reasoning and score (0.0-1.0)."
                },
                {
                    "role": "user",
                    "content": f"""User Persona: {user_persona}
Post: {original_post}
Comment: {ground_truth}
Tag: {tag}
Bag of tokens: {', '.join(bag_of_tokens) if bag_of_tokens else 'None'}"""
                },
            ],
            text_format=TagJudgment,
        )
        
        judgment = response.output_parsed
        
        return {
            "score": max(0.0, min(1.0, judgment.score)),
            "tag_star": tag
        }
        
    except Exception as e:
        print(f"Error in tag reward function: {e}")
        return {"score": 0.0, "tag_star": ""}
