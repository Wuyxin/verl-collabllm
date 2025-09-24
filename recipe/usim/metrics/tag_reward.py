import re, json, asyncio, time
from typing import Optional
from recipe.usim.utils import extract_json  # your existing JSON extractor


TAG_SCORE_PROMPT = """You are a precise judge. Score how well a single tag captures the essence of a user's comment on a post, taking the user's persona into account.

Return JSON with fields: "thought" (brief reasoning) and "score" (0.0-1.0).

Context:
User Persona: {persona}
Post: {post}
Comment: {comment}
Tag: {tag}

Rate how well this tag captures the essence of the user's response to the AITA post, considering their persona.
Guidelines:
- Score higher if the tag reflects the comment's main stance/reaction/topic with persona-consistent framing.
- Penalize generic, off-topic, or inaccurate tags.
- Output ONLY a JSON object with keys "thought" and "score".
"""

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

