from transformers import GPTJForCausalLM,AutoTokenizer, AutoModelForCausalLM
import torch
device = 'cpu'
#tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
#model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
model = GPTJForCausalLM.from_pretrained(
    "EleutherAI/gpt-j-6B", revision="float16", torch_dtype=torch.float16, low_cpu_mem_usage=True
)
