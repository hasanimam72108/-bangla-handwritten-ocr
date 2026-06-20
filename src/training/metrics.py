import torch
from jiwer import cer, wer


def compute_cer(prediction: str, ground_truth: str) -> float:
    return cer(ground_truth, prediction)


def compute_wer(prediction: str, ground_truth: str) -> float:
    return wer(ground_truth, prediction)


def compute_exact_match(prediction: str, ground_truth: str) -> bool:
    return prediction.strip() == ground_truth.strip()


@torch.no_grad()
def evaluate_model(model, dataloader, tokenizer, device):
    model.eval()
    total_cer = 0.0
    total_wer = 0.0
    exact_matches = 0
    total = 0

    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    from tqdm import tqdm
    for batch in tqdm(dataloader, desc="Validating"):
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"]
        texts = batch["texts"]

        with torch.autocast(device_type="cuda" if "cuda" in str(device) else "cpu", dtype=torch.float16):
            generated_ids = model.generate(
                pixel_values,
                max_length=model.generation_config.max_length,
                num_beams=1,
                early_stopping=True,
            )

        for i, gen_ids in enumerate(generated_ids):
            prediction = tokenizer.decode(gen_ids, skip_special_tokens=True)
            ground_truth = texts[i]
            if total < 3:
                print(f"\nSample {total+1}:")
                print(f"  GT: {ground_truth}")
                print(f"  PR: {prediction}")

            total_cer += compute_cer(prediction, ground_truth)
            total_wer += compute_wer(prediction, ground_truth)
            if compute_exact_match(prediction, ground_truth):
                exact_matches += 1
            total += 1

    avg_cer = total_cer / max(total, 1)
    avg_wer = total_wer / max(total, 1)
    accuracy = exact_matches / max(total, 1)

    return {
        "cer": avg_cer,
        "wer": avg_wer,
        "exact_match_accuracy": accuracy,
    }
