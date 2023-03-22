import torch
from torch import nn
import torch.nn.functional as F
from typing import Union, List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTJForCausalLM
from transformers.generation_logits_process import (
    LogitsProcessorList,
    NoBadWordsLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
)

from utils.constants import NEGATIVE_INF, HALF_NEGATIVE_INF
from utils.utils import logits_to_entropy, mask_pad
from load import load_weights


class RefPolicy(nn.Module):
    def __init__(self, model_name, temperature, device, model_weight: str = '', model = None):
        super().__init__()

        self.model_name = model_name
        self.device = device

        self.model = model
        if self.model is None:
            hparams = load_weights(self, AutoModelForCausalLM, model_weight, 'model', model_name)

            for param in self.model.parameters():
                param.requires_grad_(False)
            self.model = self.model.to(device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, pad_token="<|endoftext|>")
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        self.temperature = temperature

    def get_processor(self, no_repeat_ngram_size: int = 3):
        logits_processor = LogitsProcessorList()
        if no_repeat_ngram_size > 0:
            logits_processor.append(NoRepeatNGramLogitsProcessor(ngram_size=no_repeat_ngram_size))
        '''
        logits_processor.append(NoBadWordsLogitsProcessor([[self.tokenizer.pad_token_id]],
                                                          self.tokenizer.pad_token_id))
        '''
        return logits_processor

    def sample(self,
               prompts: Union[str, List[str]] = None,
               input_ids: torch.Tensor = None,
               attention_mask: torch.Tensor = None,
               max_len: int = 20,
               sample: bool = True,
               top_k: int = None,
               top_p: float = None,
               temperature: float = None,
               no_repeat_ngram_size: int = 0,
               invalidate_eos: bool = True,
               features = None,
               labels = None,
               device = None) -> Dict[str, Union[torch.Tensor, List[str]]]:

        if device is None:
            device = self.device

        if temperature is None:
            temperature = self.temperature

        if prompts is not None:
            assert input_ids is None and attention_mask is None, 'repeated input'
            if isinstance(prompts, str):
                prompts = [prompts]

            encodings_dict = self.tokenizer(prompts, return_tensors="pt", padding=True)
            input_ids = encodings_dict['input_ids'].to(device)
            attention_mask = encodings_dict['attention_mask'].to(device)

        else:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

        model_kwargs = {'attention_mask': attention_mask}
        batch_size, input_seq_len = input_ids.shape

        logits_processor = self.get_processor(no_repeat_ngram_size=no_repeat_ngram_size)

        logits_warper = self.model._get_logits_warper(
            top_k=top_k, top_p=top_p, temperature=temperature, num_beams=1
        )

        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=device)
        output_logprob = torch.zeros([batch_size, 0], dtype=torch.float, device=device)
        eos_logprobs = torch.zeros([batch_size, 0], device=device)
        output_mask = torch.ones([batch_size, 0], dtype=torch.long, device=device)

        self.model.eval()
        with torch.no_grad():
            for step in range(max_len):

                # prepare model inputs
                model_inputs = self.model.prepare_inputs_for_generation(input_ids, **model_kwargs)

                # forward pass to get next token
                outputs = self.model(
                    **model_inputs,
                )

                # in the first decoding step, we want to use the 'real' last position for each sentence
                if step == 0:
                    last_non_masked_idx = torch.sum(attention_mask, dim=1) - 1
                    next_token_logits = outputs.logits[range(batch_size), last_non_masked_idx, :]
                else:
                    next_token_logits = outputs.logits[:, -1, :]

                negative_inf = HALF_NEGATIVE_INF if next_token_logits.dtype == torch.half else NEGATIVE_INF
                next_token_scores = logits_processor(input_ids, next_token_logits)
                if invalidate_eos:
                    next_token_scores[:, self.tokenizer.eos_token_id] = negative_inf  # no endoftext
                log_prob = F.log_softmax(next_token_logits, dim=-1)

                if sample:
                    # Temperature (higher temperature => more likely to sample low probability tokens)
                    next_token_scores = logits_warper(input_ids, next_token_scores)
                    probs = F.softmax(next_token_scores, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                else:
                    # Greedy decoding
                    next_tokens = torch.argmax(next_token_scores, dim=-1)

                # finished sentences should have their next token be a padding token
                next_tokens = next_tokens * unfinished_sequences + self.tokenizer.pad_token_id * (1 - unfinished_sequences)

                    # update output mask
                output_mask = torch.cat([output_mask, unfinished_sequences[:, None]], dim=-1)
                # update output log probability
                eos_logprob = log_prob[:, self.tokenizer.eos_token_id]
                eos_logprob = eos_logprob * unfinished_sequences + NEGATIVE_INF * (1 - unfinished_sequences)
                eos_logprobs = torch.cat([eos_logprobs, eos_logprob[:, None]], dim=-1)

                token_logprob = torch.gather(log_prob, 1, next_tokens[:, None]).squeeze(1)
                token_logprob = token_logprob * unfinished_sequences + NEGATIVE_INF * (1 - unfinished_sequences)
                output_logprob = torch.cat([output_logprob, token_logprob[:, None]], dim=-1)

                # update generated ids, model inputs for next step
                input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
                model_kwargs = self.model._update_model_kwargs_for_generation(
                    outputs, model_kwargs, is_encoder_decoder=self.model.config.is_encoder_decoder
                )

                # if eos_token was found in one sentence, set sentence to finished
                unfinished_sequences = unfinished_sequences.mul((next_tokens != self.tokenizer.eos_token_id).long())

                if unfinished_sequences.max() == 0:
                    break

        response_ids = input_ids[:, input_seq_len:]
        response_text = [self.tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                         for output in response_ids]

        prompt_ids = input_ids[:, :input_seq_len]
        if prompts is None:
            prompts = [self.tokenizer.decode(query, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                       for query in prompt_ids]
        eos_probs = eos_logprobs.exp()

        return {
            'query/input_ids': prompt_ids,
            'query/text': prompts,
            'query/mask': attention_mask,
            'response/input_ids': response_ids,
            'response/text': response_text,
            'response/mask': output_mask,
            'response/log_prob': output_logprob,
            'response/eos_prob': eos_probs,
        }

    def forward_pass(self,
                     query_input_ids: torch.Tensor,
                     query_mask: torch.Tensor,
                     response_input_ids: torch.Tensor,
                     response_mask: torch.Tensor,
                     invalidate_eos: bool = True,
                     features = None,
                     labels = None,
                     device = None):
        if device is None:
            device = self.device

        batch_size, query_seq_len = query_input_ids.shape
        input_ids = torch.cat([query_input_ids, response_input_ids], dim=-1)
        attention_mask = torch.cat([query_mask, response_mask], dim=-1)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        # forward pass to get next token
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=True,
        )
        # get the first logit
        query_logits = outputs.logits[:, :query_seq_len, :]
        last_non_masked_idx = torch.sum(query_mask, dim=1) - 1
        first_logits = query_logits[range(batch_size), last_non_masked_idx, :]
        # get the second to last logit
        response_logits = outputs.logits[:, query_seq_len:-1, :]
        logits = torch.cat([first_logits[:, None], response_logits], dim=1)

        negative_inf = HALF_NEGATIVE_INF if logits.dtype == torch.half else NEGATIVE_INF
        if invalidate_eos:
            logits[:, :, self.tokenizer.eos_token_id] = negative_inf  # no endoftext

        log_prob = F.log_softmax(logits, dim=-1)
        output_logprob = torch.gather(log_prob, 2, response_input_ids[:, :, None]).squeeze(2)
        output_entropy = logits_to_entropy(logits)
        eos_prob = F.softmax(logits, dim=-1)[:, :, self.tokenizer.eos_token_id]

        return {
            'response/log_prob': mask_pad(output_logprob, response_mask),
            'response/eos_prob': mask_pad(eos_prob, response_mask),
            'response/entropy': mask_pad(output_entropy, response_mask),
        }
