from recipe.usim.utils import extract_json, parse_messages
import random

WIN_RATE_PROMPT = '''You are a helpful and meticulous judge. \
You will be given two conversations, each one of which is a session of an anonymous user with a real user. \
Your task is to judge which anonymous user is more human-like. 

Provided Information:

<|The Start of Conversation A|>
{conversation_a}
<|The End of Conversation A|>

<|The Start of Conversation B|>
{conversation_b}
<|The End of Conversation B|>

Do not let the order of the conversations affect your judgement as the order is random. 

Output format (JSON):
{{
    "thought": "<your comparison and reasoning on the two conversations>",
    "decision": <"A" or "B", if the anonymous user in conversation A/B is more human-like. In very rare cases where it is extremely hard to distinguish between the two anonymous users, output "N">
}}


Double check if the JSON object is formatted correctly. Ensure that all fields are present and properly structured. Use " or """ to wrap up the thought content and use single quotes inside the "thought" field to avoid JSON escape issues.

Your output:
'''


async def compute_score(data_source, generation, ground_truth, extra_info, **kwargs):
    # Check if litellm is available, fallback to openai if not
    try:
        import litellm

        use_litellm = True
    except ImportError:
        # litellm not found, falling back to openai
        import openai

        use_litellm = False

    gd_first = random.choice([0, 1])


    post = [{"role": "Poster", "content": extra_info.get("post")}]
    

    conversation_gd = parse_messages(post + [{"role": "Anonymous User", "content": ground_truth}])
    conversation_model = parse_messages(post + [{"role": "Anonymous User", "content": generation}])

    prompt = WIN_RATE_PROMPT.format(
        conversation_a=conversation_gd if gd_first else conversation_model, 
        conversation_b=conversation_model if gd_first else conversation_gd, 
    )

    if use_litellm:
        try:
            full_response = (
                (
                    await litellm.acompletion(
                        messages=[{"role": "user", "content": prompt}],
                        **kwargs
                    )
                )
                .choices[0]
                .message.content
            )
        except Exception as e:
            print(f"LiteLLM Error: {e}") 
            return 0.0

    else:
        client = openai.AsyncOpenAI()  # Assumes API key is set in environment
        try:
            full_response = (
                (
                    await client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        **kwargs
                    )
                )
                .choices[0]
                .message.content
            )
        except Exception as e:
            print(f"OpenAI API Error: {e}") 
            return 0.0
        
    full_response = extract_json(full_response)
    print(full_response)

    assert isinstance(full_response, dict), f"Expected a dict, got {type(full_response)}"
    assert {"decision", "thought"}.issubset(full_response.keys()), (
        f"Expected keys not found from {full_response.keys()}"
    )
    decision = full_response["decision"].capitalize().strip()

    if decision == "N":
        score = 0.5
    elif (decision == "B" and gd_first) or (decision == "A" and not gd_first):
        score = 1.
    else:
        score = 0.
        
    return score
