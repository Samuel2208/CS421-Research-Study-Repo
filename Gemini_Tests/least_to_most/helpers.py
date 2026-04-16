



def run_experiment_parallel(dataset, client, max_workers=5, use_structured=False):
    results = []

    def process_sample(sample):
        if use_structured:
            structured_input = format_structured_dialogue(sample)
            full_prompt = prompt.format(
                structured_input=json.dumps(structured_input, indent=2)
            )
            prompt_type = "least_to_most_structured"
        else:
            dialogue_text = sample["input_text"]
            full_prompt = prompt.format(dialogue_here=dialogue_text)
            prompt_type = "least_to_most"

        raw_pred = query_gemini(client, full_prompt)
        prediction, reasoning = parse_llm_output(raw_pred)

        return {
            "dialogue_id": sample["dialogue_id"],
            "window_size": sample["window_size"],
            "input_text": sample["input_text"],
            "prediction": prediction,
            "label": sample["label"].lower(),
            "prompt_type": prompt_type,
            "model": "gemini",
            "reasoning": reasoning,
            "accuracy": prediction == sample["label"],
            "prompt_word_count": len(full_prompt.split()),
            "prompt_character_count": len(full_prompt),
        }

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_sample, sample) for sample in dataset]

        for i, future in enumerate(as_completed(futures)):
            try:
                results.append(future.result())
            except Exception as e:
                print("Error in thread:", e)

            if i % 10 == 0:
                print(f"Processed {i} samples")

    return results