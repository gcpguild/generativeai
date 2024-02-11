import sys
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model_and_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    return model, tokenizer

def generate_text(model, tokenizer, prompt):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    output_ids = model.generate(
        input_ids, 
        max_length=100, 
        temperature=0.7, 
        top_k=50, 
        top_p=0.95, 
        do_sample=True, 
        pad_token_id=tokenizer.eos_token_id
    )
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output_text

def main():
    if sys.stdin.isatty():
        if len(sys.argv) > 1:
            input_str = sys.argv[1]
        else:
            print(json.dumps({"error": "No prompt provided"}))
            sys.exit(1)
    else:
        input_str = sys.stdin.read()
        
    try:
        input_data = json.loads(input_str)
        prompt = input_data['prompt']
    except (json.JSONDecodeError, KeyError):
        print(json.dumps({"error": "Invalid JSON input"}))
        sys.exit(1)

    model_path = r"C:\contracts\modelslist\gpt2"
    model, tokenizer = load_model_and_tokenizer(model_path)
    output_text = generate_text(model, tokenizer, prompt)
    print(json.dumps({"output": output_text}))

if __name__ == "__main__":
    main()
