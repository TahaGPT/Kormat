import yaml
import torch
from peft import LoraConfig, get_peft_model

from kormat.common.paths import CONFIGS
from kormat.modeling import tiny_qwen


def test_configs_match_shared_run():
    c = yaml.safe_load((CONFIGS / "train.yaml").read_text())
    assert c["base_model"] == "Qwen/Qwen2.5-1.5B-Instruct" and c["lora"]["r"] == 16 and c["lora"]["alpha"] == 32
    assert c["epochs"] == 3 and c["grad_accum"] == 16 and c["max_len"] == 1024 and c["seed"] == 42
    s = yaml.safe_load((CONFIGS / "train_smoke.yaml").read_text())
    assert s["base_model"].endswith("0.5B-Instruct") and s["data"]["subset"] == 16


def test_completion_only_loss_decreases_and_masks_prompt():
    torch.manual_seed(0)
    m = get_peft_model(tiny_qwen(), LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"]))
    ids = torch.randint(0, 1000, (1, 12))
    labels = ids.clone(); labels[:, :8] = -100
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=5e-2)
    l0 = m(input_ids=ids, labels=labels).loss.item()
    for _ in range(30):
        loss = m(input_ids=ids, labels=labels).loss; loss.backward(); opt.step(); opt.zero_grad()
    assert loss.item() < l0 * 0.9
    # changing masked tokens' labels must not change the loss
    l1 = m(input_ids=ids, labels=labels).loss.item()
    labels2 = labels.clone(); labels2[:, :8] = -100
    assert abs(l1 - m(input_ids=ids, labels=labels2).loss.item()) < 1e-6
