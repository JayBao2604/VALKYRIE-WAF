from tools.grammar_sampler_ssrf import refine_if_needed

payload = "<body/+/src/type=img%0aonload=//+/onerror%09=%09alertScript%0dx"
score = 0.7

refined_payload, new_score = refine_if_needed(
    payload,
    score
)

print(f"Before: {payload} (score: {score})")
print(f"After : {refined_payload} (score: {new_score})")