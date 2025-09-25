import re, json, asyncio, time
from typing import Optional
from recipe.usim.utils import extract_json  # your existing JSON extractor


TAG_SCORE_PROMPT = '''You are a helpful and meticulous evaluator. \
Your task is to score how well a Response Signature is expressed in a comment. \
A Response Signature is a short, high-level description of the stance, \
attitude, or overall perspective shown in the comment. \
It is not a detailed summary or direction response, but rather a concise impression of how the response is expressed.

You will be given the post to understand the context, the comment itself, \
and the candidate tag (the Response Signature) that you should evaluate.

Provided Information:
<|The Start of Post|>
{post}
<|The End of Post|>

<|The Start of Comment|>
{comment}
<|The End of Comment|>

<|The Start of Tag|>
{tag}
<|The End of Tag|>

Scoring Criteria:
Score how well the candidate Response Signature matches the stance, attitude, or overall perspective in the comment.

- 1.0 — Excellent match: The tag fully captures the stance/attitude expressed in the comment, with no major gaps or errors.
- 0.7 — Good match: The tag reflects the general stance/attitude, but misses some nuance or is slightly incomplete.
- 0.3 — Weak match: The tag only partially captures the comment, or is too vague/generic to be very useful.
- 0.0 — Mismatch: The tag does not reflect the stance/attitude at all, or is clearly wrong given the comment.


Output format (JSON):
{{
    "thought": "<How good is the signature>",
    "score": <score>
}}

Double check if the JSON object is formatted correctly. Ensure that all fields are present and properly structured. \
Use " or """ to wrap up the thought. You should not use other triple quotes inside the "thought" field. \
Instead you should use single quotes to avoid JSON escape issues.

Your evaluation:
'''

async def compute_score(data_source, generation, ground_truth, extra_info, **kwargs) -> float:
    """
    usim-compatible tag reward:
    - async
    - returns float
    """

    if not isinstance(generation, str):
        return 0.0
    m = re.search(r"<tag>(.*?)</tag>", generation, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return 0.0
    tag = m.group(1).strip()
    if not tag:
        return 0.0

    comment = ""
    if isinstance(ground_truth, str):
        m2 = re.search(r"<response>(.*?)</response>", ground_truth, flags=re.IGNORECASE | re.DOTALL)
        comment = (m2.group(1) if m2 else ground_truth).strip()[:4000]

    extra_info = extra_info or {}
    post = (extra_info.get("post") or extra_info.get("original_post") or "")[:4000]
    persona = (extra_info.get("persona") or extra_info.get("user_persona") or "")[:2000]
    idx = extra_info.get("index")

    prompt = TAG_SCORE_PROMPT.format(persona=persona, post=post, comment=comment, tag=tag)

    content = None
    try:
        import litellm
        try:
            resp = await litellm.acompletion(
                messages=[{"role": "user", "content": prompt}],
                model=kwargs.pop("model", "gpt-4o-mini"),
                temperature=kwargs.pop("temperature", 0),
                max_tokens=kwargs.pop("max_tokens", 128),
                **kwargs,
            )
            content = resp.choices[0].message.content
        except Exception:
            content = None
    except Exception:
        content = None

    if content is None:
        try:
            import openai
            client = openai.AsyncOpenAI()
            resp = await client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=kwargs.pop("model", "gpt-4o-mini"),
                temperature=kwargs.pop("temperature", 0),
                max_tokens=kwargs.pop("max_tokens", 128),
                **kwargs,
            )
            content = resp.choices[0].message.content
        except Exception:
            # LLM failed 
            return 0.0

    try:
        obj = extract_json(content)  
        score = float(obj.get("score", 0.0))
    except Exception:
        score = 0.0

    score = 0.0 if score < 0 else (1.0 if score > 1 else score)


    return score

