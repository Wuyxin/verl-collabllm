from recipe.usim.utils import extract_json, parse_messages

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


async def compute_score(data_source, messages, ground_truth, extra_info, **kwargs):
    # Check if litellm is available, fallback to openai if not
    try:
        import litellm

        use_litellm = True
    except ImportError:
        # litellm not found, falling back to openai
        import openai

        use_litellm = False

    # TODO: starting from here
    gd_first = random.choice([0, 1])

    chat_history = parse_messages(messages, strip_sys_prompt=True)
    prompt = WIN_RATE_PROMPT.format(
        single_turn_prompt=extra_info["interaction_kwargs"]["single_turn_prompt"],
        ground_truth=ground_truth,
        chat_history=chat_history,
    )

    if use_litellm:
        full_response = (
            (
                await litellm.acompletion(
                    messages=[{"role": "user", "content": prompt}],
                    **kwargs,
                )
            )
            .choices[0]
            .message.content
        )
    else:
        client = openai.AsyncOpenAI()  # Assumes API key is set in environment
        full_response = (
            (
                await client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    **kwargs,
                )
            )
            .choices[0]
            .message.content
        )

    full_response = extract_json(full_response)

    assert isinstance(full_response, dict), f"Expected a dict, got {type(full_response)}"
    assert {"decision", "thought"}.issubset(full_response.keys()), (
        f"Expected keys not found from {full_response.keys()}"
    )

    decision = full_response.pop("decision")
    return float(decision)
