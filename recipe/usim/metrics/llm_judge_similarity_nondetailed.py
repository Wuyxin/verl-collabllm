from recipe.usim.utils import extract_json, parse_messages

SIMILARITY_PROMPT = '''You are a helpful and meticulous judge. \
Given an input context, your job is to judge if a target response aligns with a reference response in terms of the high-level stance and reaction. \
You are not a chat assistant; you are a judge.

Provided Information:

<|The Start of Context|>
{context}
<|The End of Context|>

<|The Start of Target Response|>
{generation}
<|The End of Target Response|>

<|The Start of Reference Response|>
{ground_truth}
<|The End of Reference Response|>

## Notes:
- Use context to identify the argument.
- Target response and reference response are two separate replies to the context. 

## Guidelines:
For each response:
1) What is the goal? What does it try to express? 
2) What is the stance relative to that goal? Stance examples: agree/support, disagree/oppose, neutral/uncertain.
3) What is reaction/tone category and how is its intensity? Reaction/tone examples: positive, negative, mocking, sarcastic, disgusted, empathetic, skeptical, enthusiastic.
Then, compare the alignment between the target response and reference response.
1) Do they emphasize similar stance, reaction/tone, and overall topic? Compare the big-picture sentiments, attitudes, intentions.
2) Ignore length, writing style, grammar, punctuation, emojis, profanity level, links, and formatting.
3) Ignore small factual omissions, unless they flip the stance.

## Scoring criteria:
- 1.0 (Perfect Alignment): Same stance, same reaction/tone with similar intensity, same overall topic/goal. No meaningful divergence.
- 0.75 (Strong Alignment): Stance aligned, reaction/tone mostly similar with minor differences in intensity or emphasis.
- 0.5 (Partial Alignment): Some overlap in stance/reaction, but important differences in tone or focus. Mixed alignment.
- 0.25 (Weak Alignment): Significant divergence in stance or tone. Minimal overlap; core attitude misaligned.
- 0.0 (No Alignment): Opposite stance or tone, or completely different reactions. Overall sentiment and intention not aligned.


## Output format (JSON):
{{
    "thought": "<your reasoning and answers to the questions in the guidelines>",
    "score": <a numerical score between 0-1 following your thought and the scoring criteria>
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

    prompt = SIMILARITY_PROMPT.format(
        context=extra_info.get("post"),
        generation=generation,
        ground_truth=ground_truth
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

    assert isinstance(full_response, dict), f"Expected a dict, got {type(full_response)}"
    assert {"score", "thought"}.issubset(full_response.keys()), (
        f"Expected keys not found from {full_response.keys()}"
    )
    score = full_response['score']

    return float(score)
