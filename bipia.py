def evaluate_scenario(
    scenario_name,
    handler,
    template,
    seed=2023,
    do_sample=True,
    batch_size=1,
    attack="text",
):
    if attack == "text":
        attack_path = "BIPIA/benchmark/text_attack_test.json"
    elif attack == "code":
        attack_path = "BIPIA/benchmark/code_attack_test.json"
    else:
        raise ValueError(f"Invalid attack type: {attack}")

    scenario_to_path = {
        "email": "BIPIA/benchmark/email/test.jsonl",
        "code": "BIPIA/benchmark/code/test.jsonl",
        "qa": "BIPIA/benchmark/qa/test.jsonl",
        "abstract": "BIPIA/benchmark/abstract/test.jsonl",
        "table": "BIPIA/benchmark/table/test.jsonl",
    }
    pia_builder = AutoPIABuilder.from_name(scenario_name)(seed)
    pia_samples = pia_builder(
        scenario_to_path[scenario_name],
        attack_path,
        enable_stealth=False,
    )

    # Filter samples for specific attacks (which do no require LLM evaluation)
    df = pia_samples[
        pia_samples.attack_name.str.startswith("Emoji Substitution")
        | pia_samples.attack_name.str.startswith("Language Translation")
        | pia_samples.attack_name.str.contains("Marketing & Advertising-1")
        | pia_samples.attack_name.str.contains("Scams & Fraud-1")
        | pia_samples.attack_name.str.contains(
            "Bringing Down Hosts and Servers (Denial of Service)"
        )
        | pia_samples.attack_name.str.contains("Compromising Computers")
        | pia_samples.attack_name.str.contains(
            "Encrypting Documents and Demanding Ransom (Ransomware)"
        )
        | pia_samples.attack_name.str.contains("Corrupting an Operating System")
        | pia_samples.attack_name.str.contains("Blocking Internet Connection")
        | pia_samples.attack_name.str.contains("Introduce System Fingerprinting")
        | pia_samples.attack_name.str.contains("Screen Scraping")
        | pia_samples.attack_name.str.contains("Keylogging")
        | pia_samples.attack_name.str.contains("Traffic Analysis")
        | pia_samples.attack_name.str.contains("Data Eavesdropping")
    ]

    # Prepare prompts
    system_prompts = []
    user_prompts = []

    for i in range(len(df)):
        system_prompt, user_prompt = pia_builder.construct_aside_prompts(df.iloc[i])
        # Format prompts using the provided template
        system_prompt = format_prompt(system_prompt, template, "system")
        user_prompt = format_prompt(user_prompt, template, "user")
        system_prompts.append(system_prompt)
        user_prompts.append(user_prompt)

    # Generate responses using the model
    responses = call_model_with_batch_support(
        handler,
        system_prompts,
        user_prompts,
        batch_size=batch_size,
        do_sample=do_sample,
    )
    # Prepare output for evaluation
    out = []
    if "question" not in df.columns:
        df["question"] = ""
    for attack_name, task_name, target, message, position, response in zip(
        df["attack_name"],
        df["task_name"],
        df["ideal"],
        df["question"],
        df["position"],
        responses,
    ):
        out.append(
            {
                "attack_name": attack_name,
                "task_name": task_name,
                "response": response,
                "message": message,
                "target": target,
                "position": position,
            }
        )

    # Save responses
    output_path = Path(f"BIPIA/output/{scenario_name}_responses.jsonl")
    output_path.parent.mkdir(exist_ok=True, parents=True)
    with jsonlines.open(output_path, "w") as writer:
        writer.write_all(out)

    # Evaluate responses
    gpt_config_file = "BIPIA/config/my_gpt35.yaml"  # Make sure this exists
    # Define attacks to evaluate
    attacks = np.array(df.attack_name.unique())

    evaluator = BipiaEvalFactory(
        gpt_config=gpt_config_file,
        activate_attacks=attacks,
    )

    asrs = evaluator.add_batch(
        predictions=responses,
        references=df["ideal"],
        attacks=df["attack_name"],
        tasks=df["task_name"],
    )

    avg_asr = np.mean(asrs)
    print(f"The average ASR for {scenario_name} scenario is: {avg_asr}")

    return df, asrs, avg_asr


def evaluate_bipia(
    handler,
    template,
    seeds,
    scenarios=["email", "code", "table", "abstract", "qa"],
    attacks=["text", "code"],
    do_sample=True,
    batch_size=1,
):
    """
    Evaluates model performance on BIPIA dataset across multiple scenarios and seeds.

    Args:
        handler: The model handler instance
        template: The prompt template to use
        seeds: List of seeds for evaluation
        scenarios: List of BIPIA scenarios to evaluate
        attacks: List of attack types to evaluate ('text' or 'code')
        do_sample: Whether to use sampling in generation
        batch_size: Batch size for processing

    Returns:
        dict: Dictionary containing results for each attack type (text and code)
    """
    results = {}

    for attack_type in attacks:
        attack_results = []
        print(f"\nEvaluating BIPIA {attack_type} attacks:")

        # Use only code scenario for code attacks, and other scenarios for text attacks
        if attack_type == "code":
            attack_scenarios = ["code"]
        else:
            attack_scenarios = [s for s in scenarios if s != "code"]

        print(f"Using scenarios {attack_scenarios} for {attack_type} attacks")

        for seed in seeds:
            set_seed(seed)
            seed_asrs = []

            for scenario in attack_scenarios:
                print(
                    f"\nEvaluating {scenario} scenario with seed {seed}, attack: {attack_type}"
                )
                _, asrs, avg_asr = evaluate_scenario(
                    scenario,
                    handler,
                    template,
                    seed,
                    do_sample,
                    batch_size,
                    attack_type,
                )
                seed_asrs.append(avg_asr)
                print(
                    f"Attack: {attack_type}, Scenario: {scenario}, Seed: {seed}, ASR: {avg_asr:.4f}"
                )

            # Calculate average ASR across all scenarios for this seed and attack type
            mean_seed_asr = np.mean(seed_asrs)
            attack_results.append(mean_seed_asr)
            print(
                f"Seed {seed} average ASR across scenarios for {attack_type} attacks: {mean_seed_asr:.4f}"
            )

        # Calculate overall mean and std across all seeds for this attack type
        mean_asr = np.mean(attack_results)
        std_asr = np.std(attack_results)

        print(f"\nOverall BIPIA {attack_type} Results:")
        print(f"Mean ASR: {mean_asr:.4f}, Std ASR: {std_asr:.4f}")

        results[attack_type] = {"mean": mean_asr * 100, "std": std_asr * 100}

    return results