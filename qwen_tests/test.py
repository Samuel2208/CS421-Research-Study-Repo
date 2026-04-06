from vllm import LLM, SamplingParams

def main():
    # 1. Initialize the model using the AWQ version ONLY
    llm = LLM(
        model="Qwen/Qwen2.5-7B-Instruct-AWQ", 
        quantization="awq",
        max_model_len=4096
    )

    # 2. Set generation parameters
    sampling_params = SamplingParams(temperature=0.7, top_p=0.8, max_tokens=512)

    # 3. Format prompt and generate
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing in one simple sentence."}
    ]
    
    # Apply Qwen's specific chat template using the correct tokenizer access method
    tokenizer = llm.get_tokenizer()
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # 4. Generate the output
    outputs = llm.generate([prompt], sampling_params)

    for output in outputs:
        print(f"\nGenerated text: {output.outputs[0].text}")

if __name__ == "__main__":
    main()