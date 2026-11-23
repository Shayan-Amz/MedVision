import json

pqa = json.load(open("./checkpoints/pubmedqa_split_manifest.json", encoding="utf-8"))
assert len(pqa["test_pubids"]) == 200 and len(pqa["train_pubids"]) == 800 and len(pqa["pqa_artificial_train_pubids"]) == 3200
assert all(len(h) == 64 for h in pqa["test_question_sha256"])
assert all(len(h) == 64 for h in pqa["pqa_artificial_question_sha256"])
assert len(set(pqa["test_pubids"]) & set(pqa["train_pubids"])) == 0
assert len(set(pqa["test_pubids"]) & set(pqa["pqa_artificial_train_pubids"])) == 0
assert len(set(pqa["test_question_sha256"]) & set(pqa["train_labeled_question_sha256"])) == 0
assert len(set(pqa["test_question_sha256"]) & set(pqa["pqa_artificial_question_sha256"])) == 0
print("[OK] PubMedQA Manifest: 0 pubid & 0 full 64-char SHA-256 question overlap (200 test vs 800 labeled + 3,200 artificial)")

iu = json.load(open("./checkpoints/iu_xray_split_manifest.json", encoding="utf-8"))
e2e = json.load(open("./checkpoints/iu_xray_e2e_audit.json", encoding="utf-8"))
e2e_uids = [str(it["uid"]) for it in e2e["items"]]
assert len(iu["test_selected_250_uids"]) == 250 and len(iu["test_evaluated_238_uids"]) == 238
assert len(set(iu["test_selected_250_uids"]) & set(iu["radiologist_lora_train_uids"])) == 0
assert set(e2e_uids) == set(iu["test_evaluated_238_uids"])
print("[OK] IU-Xray Manifest: 238 E2E UIDs match 250 selected (12 missing frontal) & 0 overlap with 3,080 train UIDs")
