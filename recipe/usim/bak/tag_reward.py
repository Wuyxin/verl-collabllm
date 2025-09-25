import re
import json
from openai import OpenAI
from pydantic import BaseModel

PREDICTIONS_FILE = "/dfs/project/kgrlm/common/llm_twin/outputs"

client = OpenAI()

class TagJudgment(BaseModel):
    reasoning: str
    score: float


TAG_RATING_PROMPT = '''You are a helpful and meticulous evaluator. Your task is to \
evaluate the *accuracy* of an AI model's answer to a target question. \
You will be given the target question, the ground truth answer, and the conversation between the AI and the user.

Provided Information:

<|The Start of Target Question and Ground Truth Answer|>
Target Question: {single_turn_prompt}
Ground Truth Answer: {ground_truth}
<|The End of Target Question and Ground Truth Answer|>

<|The Start of The Conversation|>
{chat_history}
<|The End of The Conversation|>

You should determine whether the model's final response to the target question is \
factually correct and consistent with the provided ground truth.

Rating criteria (binary):
  • 1 = Correct   — the response matches the ground truth.
  • 0 = Incorrect — the response contradicts or misses the ground truth.

Output format (JSON):
{{
    "thought": "<your reasoning here>",
    "accuracy": <0 or 1>
}}

Double check if the JSON object is formatted correctly. Ensure that all fields are present and properly structured. \
Use " or """ to wrap up the thought and use single quotes inside the "thought" field to avoid JSON escape issues.

Your evaluation:
'''



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
    
    
    print(f"Data Source: {data_source}")
    print(f"Model Output: {solution_str}")
    print(f"Ground Truth: {ground_truth}")
    print(f"Extra Info Keys: {list(extra_info.keys()) if extra_info else 'None'}")
    
    try:
        # Extract single tag from <tag>...</tag> format
        tag_match = re.search(r'<tag>(.*?)</tag>', solution_str, re.DOTALL)
        print(f"solution_str {solution_str} | tag_match {tag_match}")
        if not tag_match:
            print(f"No tag found in solution: {solution_str}")
            return {"score": 0.0, "tag_star": ""}
            
        tag = tag_match.group(1).strip(" \n")
        print(f"Extracted Tag: '{tag}'")

        print(f"solution_str {solution_str} | tag_match {tag_match} | tag {tag}")
        import pdb; pdb.set_trace()
        
        # Get context information from extra_info
        original_post = extra_info.get("original_post", "") if extra_info else ""
        user_persona = extra_info.get("user_persona", "") if extra_info else ""
        bag_of_tokens = extra_info.get("bag_of_tokens", []) if extra_info else []
        
        print(f"Original Post: {original_post[:100]}...")
        print(f"User Persona: {user_persona[:100]}...")
        print(f"Bag of Tokens: {bag_of_tokens}")
        
        # Extract response from ground truth for evaluation
        response_match = re.search(r'<response>(.*?)</response>', ground_truth, re.DOTALL)
        actual_comment = response_match.group(1).strip() if response_match else ground_truth
        
        print(f"Actual Comment: {actual_comment[:100]}...")
        
        # LLM judge with structured output
        print(f"Calling LLM Judge...")
        response = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": f"""Rate how well the tag connects the AITA post to the comment given the user persona. Provide reasoning and score (0.0-1.0).
                    User Persona: {user_persona}

AITA Post: {original_post}

Comment: {actual_comment}

Tag: {tag}

Rate how well this tag captures the essence of the user's response to the AITA post, considering their persona."""
                },
            ],
            response_format=TagJudgment,
        )
        
        judgment = response.choices[0].message.parsed
        print(f"LLM Judge Result:")
        print(f"   Reasoning: {judgment.reasoning}")
        print(f"   Score: {judgment.score}")
        
        final_score = max(0.0, min(1.0, judgment.score))
        result = {
            "score": final_score,
            "tag_star": tag
        }
        
        # Load existing predictions
        idx = extra_info.get("index", -1) if extra_info else -1
        try:
            with open(PREDICTIONS_FILE, "r") as f:
                data = json.load(f)
        except:
            data = {}
        
        # Add new prediction
        if str(idx) not in data:
            data[str(idx)] = {"predictions": []}
        data[str(idx)]["predictions"].append({"tag": tag, "score": final_score})
        
        # Save back
        with open(PREDICTIONS_FILE, "w") as f:
            json.dump(data, f)
        
        print(f"Final Result: {result}")
        print(f"=" * 50)
        
        return result
        
    except Exception as e:
        print(f"Error in tag reward function: {e}")
        return {"score": 0.0, "tag_star": ""}
