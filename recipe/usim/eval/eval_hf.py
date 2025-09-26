import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model_and_tokenizer(model_path: str):
    """Load model and tokenizer from a local path or checkpoint."""
    print(f"Loading model from path: {model_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    return model, tokenizer

def generate_response(model, tokenizer, system_prompt: str, user_message: str, max_new_tokens: int = 512, temperature: float = 0.7):
    """Generate response using the model."""
    
    # Create simple prompt
    prompt = f"{system_prompt}\n\nUser: {user_message}\n\nAssistant:"
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode response
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the new generated part
    response = generated_text[len(prompt):].strip()
    
    return response

def main():
    # CONFIGURATION - Edit these as needed
    MODEL_PATH = "/dfs/project/kgrlm/common/llm_twin/rl_merged/outputs_real_LLM_global_step_250"  # Change this to your model path
    SYSTEM_PROMPT = "You are a helpful AI assistant."
    USER_MESSAGE = "Whats 2+7?"
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(MODEL_PATH)
    
    response = generate_response(model, tokenizer, SYSTEM_PROMPT, USER_MESSAGE)
    
    print(f"Assistant: {response}")
    print("-" * 50)
    print("Done!")

if __name__ == "__main__":
    main()