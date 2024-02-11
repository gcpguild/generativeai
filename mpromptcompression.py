import os
import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

def load_model(model_path):
    gpt2_model = None
    try:
        if gpt2_model is None:
            model_file = os.path.join(model_path, "pytorch_model.bin")
            config_file = os.path.join(model_path, "config.json")
            
            st.write(f"Loading GPT-2 model from {model_path}...")

            with st.spinner("Loading GPT-2 model. Please wait..."):
                # Use from_pretrained with local files
                gpt2_model = GPT2LMHeadModel.from_pretrained(
                    model_path,
                    state_dict=torch.load(model_file, map_location=torch.device('cpu')),  # Specify CPU
                    config=config_file
                )
            st.success("GPT-2 model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading GPT-2 model: {e}")

    return gpt2_model

def compress_prompt(gpt2_model, prompt, max_tokens=100, temperature=0.7):
    try:
        if gpt2_model is not None:
            # Tokenize and encode the prompt
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            input_ids = tokenizer.encode(prompt, return_tensors="pt")

            # Generate compressed prompt
            output = gpt2_model.generate(
                input_ids,
                max_length=len(input_ids[0]) + max_tokens,
                temperature=temperature,
                num_beams=5,
                no_repeat_ngram_size=2,
                top_k=50,
                top_p=0.95,
                early_stopping=True,
            )

            # Decode the generated prompt
            compressed_prompt = tokenizer.decode(output[0], skip_special_tokens=True)

            st.write(f"Original Prompt: {prompt}")
            st.write(f"Compressed Prompt: {compressed_prompt}")

            return compressed_prompt
    except Exception as e:
        st.write(f"Error compressing prompt: {e}")

def main():
    st.title("Prompt Compression App")
    model_path = r"C:\contracts\modelslist\gpt2"  
    # locally downloaded GPT-2 model

    # Button to trigger model loading
    if st.button("Load GPT-2 Model"):
        st.session_state.gpt2_model = load_model(model_path)

    # Check if the model is loaded
    if st.session_state.gpt2_model is not None:
        # User input for the original prompt
        original_prompt = st.text_area("Enter the original prompt:")

        # Button to trigger prompt compression
        if st.button("Compress Prompt"):
            if original_prompt:
                st.subheader("Original Prompt:")
                st.write(original_prompt)

                # Use GPT-2 to compress the prompt
                with st.spinner("Compressing prompt. Please wait..."):
                    try:
                        compress_prompt(st.session_state.gpt2_model, original_prompt)
                    except Exception as e:
                        st.error(f"Error compressing prompt: {e}")
            else:
                st.warning("Please enter a prompt.")

if __name__ == "__main__":
    main()
