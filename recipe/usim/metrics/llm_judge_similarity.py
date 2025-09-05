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

## Edge Cases:
- Sarcasm and mockery count as negative stance toward the target. Look for cues like exaggerated praise, scare quotes, "yeah right", or obvious hyperbole.
- Rhetorical questions and hedging ("I guess", "maybe") can soften intensity.
- If reference response has multiple points, judge by the central attitude/stance.

## Scoring criteria:
- 1.0: Same stance and reaction; tone intensity similar; topic recognizable as the same.
- 0.8: Same stance and reaction; tone intensity differs; topic aligned.
- 0.5: Same stance but reaction category differs or is weak/unclear; topic aligned or loosely aligned.
- 0.3: Related topic but stance or reaction ambiguous/misaligned.
- 0.0: Opposite stance or clearly off-topic

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
        except litellm.exceptions.BadRequestError as e:
            raise RuntimeError(f"LiteLLM BadRequestError: {e}") 

        print(full_response)
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
        except openai.error.OpenAIError as e:
            raise RuntimeError(f"OpenAI API Error: {e}")
            
    print(full_response)
    full_response = extract_json(full_response)

    assert isinstance(full_response, dict), f"Expected a dict, got {type(full_response)}"
    assert {"score", "thought"}.issubset(full_response.keys()), (
        f"Expected keys not found from {full_response.keys()}"
    )
    score = full_response['score']

    return float(score)
