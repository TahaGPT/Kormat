"""Tokenizer-only compatibility proof: Qwen2.5-1.5B-Instruct vs Qwen2.5-0.5B-Instruct. No weights loaded."""
import hashlib, json
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
A, B = "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct"
def sha(repo, f):
    return hashlib.sha256(open(hf_hub_download(repo, f), "rb").read()).hexdigest()
ta, tb = AutoTokenizer.from_pretrained(A), AutoTokenizer.from_pretrained(B)
va, vb = ta.get_vocab(), tb.get_vocab()
ma = json.load(open(hf_hub_download(A, "tokenizer.json")))["model"]["merges"]
mb = json.load(open(hf_hub_download(B, "tokenizer.json")))["model"]["merges"]
strings = [
 "Patient is a 63-year-old male with type 2 diabetes mellitus and hypertension.",
 "Exclusion: history of myocardial infarction within the past 6 months.",
 "1. The patient presented with chest pain.\n2. ECG showed ST elevation.",
 "HbA1c 8.2%, eGFR 45 mL/min/1.73 m^2, BMI 31.4 kg/m2.",
 "No evidence of brain metastases on MRI.",
 "Inclusion: histologically confirmed non-small cell lung carcinoma (NSCLC), stage IIIB/IV.",
 "ECOG performance status 0-1; adequate organ function.",
 "Patient denies tobacco use; quit smoking 10 years ago.",
 '{"label": "included", "sentence_ids": [0, 3, 5]}',
 "<think>\n</think>", "Pregnant or breastfeeding women are excluded.",
 "Dose: 5 mg/kg IV q3w x 4 cycles.", "Cr 1.9 mg/dL, K 5.6 mEq/L.",
 "   leading spaces and\ttabs\n\n\nnewlines", "Ünïcödé — naïve café 患者 糖尿病 пациент",
 "The quick brown fox jumps over the lazy dog.", "", "a", "🙂 emoji test 🚑",
 "def f(x):\n    return x ** 2  # square",
] 
base = ["Patient has {} for {} years.", "Criterion {}: {} must be documented.", "Lab {} value {} recorded on day {}."]
words = ["asthma","warfarin","anticoagulant","hepatocellular","creatinine","HIV","metformin","neuropathy","immunotherapy","pembrolizumab","stroke","COPD","dialysis","hemoglobin","tamoxifen","BRCA1","EGFR","prednisone","lupus","sepsis","osteoporosis","insulin","ketoacidosis","thrombocytopenia","atrial fibrillation","cirrhosis","radiotherapy","biopsy","remission","relapse"]
i = 0
while len(strings) < 50:
    strings.append(base[i % 3].format(words[i % 30], words[(i * 7 + 3) % 30], i + 1)); i += 1
mism = [s for s in strings if ta(s)["input_ids"] != tb(s)["input_ids"]]
msgs = [{"role": "system", "content": "You are a clinical trial assistant."}, {"role": "user", "content": "Is the patient eligible?"}]
ca, cb = ta.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True), tb.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
out = {
 "models": [A, B],
 "vocab_equal": va == vb, "vocab_size_tokenizer": [len(ta), len(tb)],
 "config_vocab_size": [151936, 151936],
 "merges_equal": ma == mb, "n_merges": [len(ma), len(mb)],
 "special_tokens_equal": ta.special_tokens_map == tb.special_tokens_map and ta.all_special_ids == tb.all_special_ids,
 "special_tokens_map": ta.special_tokens_map,
 "added_tokens_equal": ta.get_added_vocab() == tb.get_added_vocab(),
 "eos_pad_ids": {"1.5B": [ta.eos_token_id, ta.pad_token_id], "0.5B": [tb.eos_token_id, tb.pad_token_id]},
 "chat_template_equal": ta.chat_template == tb.chat_template, "rendered_chat_equal": ca == cb,
 "tokenizer_json_sha256": [sha(A, "tokenizer.json"), sha(B, "tokenizer.json")],
 "vocab_json_sha256": [sha(A, "vocab.json"), sha(B, "vocab.json")],
 "merges_txt_sha256": [sha(A, "merges.txt"), sha(B, "merges.txt")],
 "n_strings": len(strings), "n_id_mismatches": len(mism), "mismatching_strings": mism,
 "total_tokens_checked": sum(len(ta(s)["input_ids"]) for s in strings),
}
out["compatible"] = out["vocab_equal"] and out["merges_equal"] and out["special_tokens_equal"] and out["chat_template_equal"] and not mism
json.dump(out, open("results/research/tokenizer_compat.json", "w"), indent=2, ensure_ascii=False)
print(json.dumps({k: v for k, v in out.items() if k != "special_tokens_map"}, indent=1, ensure_ascii=False))
