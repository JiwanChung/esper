from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
x = 'the yellow fox'
z = tok(x, return_tensors='pt').to('cuda')
model = AutoModelForCausalLM.from_pretrained(
    "EleutherAI/gpt-j-6B", revision="float16", torch_dtype=torch.float16, low_cpu_mem_usage=True
).to('cuda').eval()
z2 = model.generate(**z, max_length=10)
z3 = tok.decode(z2[0])
print(z3)
