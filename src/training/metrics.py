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

    for batch in dataloader:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"]
        texts = batch["texts"]

        generated_ids = model.generate(
            pixel_values,
            max_length=model.config.max_length,
            num_beams=4,
            early_stopping=True,
        )

        for i, gen_ids in enumerate(generated_ids):
            prediction = tokenizer.decode(gen_ids, skip_special_tokens=True)
            ground_truth = texts[i]

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
