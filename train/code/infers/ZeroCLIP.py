import sys
from datetime import datetime
from pathlib import Path

from PIL import Image
import numpy as np
import torch
from torch import nn
from transformers.models.gpt2 import GPT2LMHeadModel, GPT2Tokenizer
from transformers.models.gpt_neo import GPTNeoForCausalLM
import clip

from clipcap import ClipCap


forbidden_tokens_file_path = Path(__file__).parent / "forbidden_tokens.npy"


def log_info(text, verbose=True):
    if verbose:
        dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        print(f'{dt_string} | {text}')
        sys.stdout.flush()


def add_context(x, y):
    return (x[0] + y[0], x[1] + y[1])


def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()


class CLIPTextGenerator:
    def __init__(self,
                 lm_model=None,
                 tokenizer=None,
                 clip=None,
                 clip_preprocess=None,
                 target_seq_length=15,
                 reset_context_delta=True,
                 num_iterations=5,
                 clip_loss_temperature=0.01,
                 clip_scale=1.,
                 ce_scale=0.2,
                 stepsize=0.3,
                 grad_norm_factor=0.9,
                 fusion_factor=0.99,
                 repetition_penalty=1.,
                 end_token='.',
                 end_factor=1.01,
                 forbidden_factor=20,
                 device=None,
                 **kwargs):
        self.device = device

        # Initialize Language model

        '''
        if lm_model == 'gpt-neo':
            self.lm_tokenizer = GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-125M')
            self.lm_model = GPTNeoForCausalLM.from_pretrained('EleutherAI/gpt-neo-125M', output_hidden_states=True)
        elif lm_model == 'gpt-2':
            self.lm_tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
            self.lm_model = GPT2LMHeadModel.from_pretrained('gpt2-medium', output_hidden_states=True)
        '''
        self.lm_tokenizer = tokenizer
        self.lm_model = lm_model

        self.lm_model.to(self.device)
        self.lm_model.eval()

        self.forbidden_tokens = np.load(forbidden_tokens_file_path)
        '''
        self.capital_letter_tokens = [self.lm_tokenizer.encoder[x] for x in self.lm_tokenizer.encoder.keys() if
                                      (x[0] == 'Ġ' and len(x) > 1 and x[1].isupper())]
        '''

        # Freeze LM weights
        for param in self.lm_model.parameters():
            param.requires_grad = False

        # Initialize CLIP
        self.clip = clip
        self.clip_preprocess = clip_preprocess
        # convert_models_to_fp32(self.clip)

        # Init arguments
        self.target_seq_length = target_seq_length
        self.reset_context_delta = reset_context_delta
        self.num_iterations = num_iterations
        self.clip_loss_temperature = clip_loss_temperature
        self.clip_scale = clip_scale
        self.ce_scale = ce_scale
        self.stepsize = stepsize
        self.grad_norm_factor = grad_norm_factor
        self.fusion_factor = fusion_factor
        self.repetition_penalty = repetition_penalty
        self.end_token = self.lm_tokenizer.encode(end_token)[0]
        self.end_factor = end_factor
        self.ef_idx = 1
        self.forbidden_factor = forbidden_factor

    def get_img_feature(self, img_path, weights):
        imgs = [Image.open(x) for x in img_path]
        clip_imgs = [self.clip_preprocess(x).unsqueeze(0).to(self.device) for x in imgs]

        with torch.no_grad():
            image_fts = [self.clip.encode_image(x) for x in clip_imgs]

            if weights is not None:
                image_features = sum([x * weights[i] for i, x in enumerate(image_fts)])
            else:
                image_features = sum(image_fts)

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            return image_features.detach()

    def get_img_features(self, img_path, weights):
        imgs = [Image.open(x) for x in img_path]
        clip_imgs = [self.clip_preprocess(x).unsqueeze(0).to(self.device) for x in imgs]

        with torch.no_grad():
            image_fts = [self.clip.encode_image(x) for x in clip_imgs]

            image_features = torch.cat(image_fts, dim=0)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features.detach()

    def get_txt_features(self, text):
        clip_texts = clip.tokenize(text, truncate=True).to(self.device)

        with torch.no_grad():
            text_features = self.clip.encode_text(clip_texts)

            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.detach()

    def get_combined_feature(self, img_path, texts, weights_i, weights_t):
        imgs = [Image.open(x) for x in img_path]
        clip_imgs = [self.clip_preprocess(x).unsqueeze(0).to(self.device) for x in imgs]
        clip_texts = [clip.tokenize(x).to(self.device) for x in texts]

        with torch.no_grad():
            image_fts = [self.clip.encode_image(x) for x in clip_imgs]
            text_fts = [self.clip.encode_text(x) for x in clip_texts]

            features = sum([x * weights_i[i] for i, x in enumerate(image_fts)])
            if weights_t is not None:
                features += sum([x * weights_t[i] for i, x in enumerate(text_fts)])

            features = features / features.norm(dim=-1, keepdim=True)
            return features.detach()

    def run(self, image_features, cond_text, beam_size, verbose=True):
        self.is_pretrained = isinstance(getattr(self.lm_model, 'model', None), ClipCap)
        self.image_features = image_features
        if verbose:
            print('pretrained model:', self.is_pretrained)

        context_tokens = self.lm_tokenizer.encode(cond_text)

        output_tokens, output_text = self.generate_text(image_features, context_tokens, beam_size, verbose=verbose)

        return output_text

    def run_model(self, feat, tokens, past_key_values=None):
        if self.is_pretrained:
            if torch.is_tensor(feat):
                feat = feat.repeat(tokens.shape[0], 1)
            return self.lm_model.model(tokens, features=feat, past_key_values=past_key_values)
        else:
            return self.lm_model(tokens, past_key_values=past_key_values)

    def generate_text(self, feat, context_tokens, beam_size, verbose=True):
        context_tokens = torch.tensor(context_tokens, device=self.device, dtype=torch.long).unsqueeze(0)

        gen_tokens = None
        scores = None
        seq_lengths = torch.ones(beam_size, device=self.device)
        is_stopped = torch.zeros(beam_size, device=self.device, dtype=torch.bool)

        for i in range(self.target_seq_length):
            probs = self.get_next_probs(i, feat, context_tokens)
            logits = probs.log()

            if scores is None:
                scores, next_tokens = logits.topk(beam_size, -1)
                context_tokens = context_tokens.expand(beam_size, *context_tokens.shape[1:])
                next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)

                if gen_tokens is None:
                    gen_tokens = next_tokens
                else:
                    gen_tokens = gen_tokens.expand(beam_size, *gen_tokens.shape[1:])
                    gen_tokens = torch.cat((gen_tokens, next_tokens), dim=1)
            else:
                logits[is_stopped] = -float(np.inf)
                logits[is_stopped, 0] = 0
                scores_sum = scores[:, None] + logits
                seq_lengths[~is_stopped] += 1
                scores_sum_average = scores_sum / seq_lengths[:, None]
                scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(
                    beam_size, -1)
                next_tokens_source = next_tokens // scores_sum.shape[1]
                seq_lengths = seq_lengths[next_tokens_source]
                next_tokens = next_tokens % scores_sum.shape[1]
                next_tokens = next_tokens.unsqueeze(1)
                gen_tokens = gen_tokens[next_tokens_source]
                gen_tokens = torch.cat((gen_tokens, next_tokens), dim=-1)
                context_tokens = context_tokens[next_tokens_source]
                scores = scores_sum_average * seq_lengths
                is_stopped = is_stopped[next_tokens_source]

            context_tokens = torch.cat((context_tokens, next_tokens), dim=1)
            is_stopped = is_stopped + next_tokens.eq(self.end_token).squeeze()

            ####
            tmp_scores = scores / seq_lengths
            tmp_output_list = gen_tokens.cpu().numpy()
            tmp_output_texts = [
                self.lm_tokenizer.decode(tmp_output)
                for tmp_output, tmp_length in zip(tmp_output_list, seq_lengths)
            ]
            tmp_order = tmp_scores.argsort(descending=True)
            tmp_output_texts = [tmp_output_texts[i] + ' %% ' + str(tmp_scores[i].cpu().numpy()) for i in tmp_order]
            log_info(tmp_output_texts, verbose=verbose)
            ####

            if is_stopped.all():
                break

        scores = scores / seq_lengths
        output_list = gen_tokens.cpu().numpy()
        output_texts = [
            self.lm_tokenizer.decode(output[: int(length)])
            for output, length in zip(output_list, seq_lengths)
        ]
        order = scores.argsort(descending=True)
        output_texts = [output_texts[i] for i in order]

        return context_tokens, output_texts

    def get_next_probs(self, i, feat, context_tokens):
        last_token = context_tokens[:, -1:]

        if self.reset_context_delta and context_tokens.size(1) > 1:
            context = self.run_model(feat, context_tokens[:, :-1])["past_key_values"]

        # Logits of LM with unshifted context
        logits_before_shift = self.run_model(feat, context_tokens)["logits"]
        logits_before_shift = logits_before_shift[:, -1, :]
        probs_before_shift = nn.functional.softmax(logits_before_shift, dim=-1)

        if context:
            context = self.shift_context(i, context, last_token, context_tokens, probs_before_shift)

        lm_output = self.run_model(None, last_token, past_key_values=context)
        logits, past = (
            lm_output["logits"],
            lm_output["past_key_values"],
        )
        logits = logits[:, -1, :]

        logits = self.update_special_tokens_logits(context_tokens, i, logits)

        probs = nn.functional.softmax(logits, dim=-1)
        probs = (probs ** self.fusion_factor) * (probs_before_shift ** (1 - self.fusion_factor))
        probs = probs / probs.sum()

        return probs

    def shift_context(self, i, context, last_token, context_tokens, probs_before_shift):
        context_delta = [tuple([np.zeros(x.shape).astype("float32") for x in p]) for p in context]

        window_mask = torch.ones_like(context[0][0]).to(self.device)

        for i in range(self.num_iterations):
            curr_shift = [tuple([torch.from_numpy(x).requires_grad_(True).to(device=self.device) for x in p_]) for p_ in
                          context_delta]

            for p0, p1 in curr_shift:
                p0.retain_grad()
                p1.retain_grad()

            shifted_context = list(map(add_context, context, curr_shift))

            shifted_outputs = self.run_model(None, last_token, past_key_values=shifted_context)
            logits = shifted_outputs["logits"][:, -1, :]
            probs = nn.functional.softmax(logits, dim=-1)

            loss = 0.0

            # CLIP LOSS
            clip_loss, clip_losses = self.clip_loss(probs, context_tokens)
            loss += self.clip_scale * clip_loss

            # CE/Fluency loss
            ce_loss = self.ce_scale * ((probs * probs.log()) - (probs * probs_before_shift.log())).sum(-1)
            loss += ce_loss.sum()

            loss.backward()

            # ---------- Weights ----------
            combined_scores_k = -(ce_loss)
            combined_scores_c = -(self.clip_scale * torch.stack(clip_losses))

            # minmax
            if combined_scores_k.shape[0] == 1:
                tmp_weights_c = tmp_weights_k = torch.ones(*combined_scores_k.shape).to(self.device)
            else:
                tmp_weights_k = ((combined_scores_k - combined_scores_k.min())) / (
                        combined_scores_k.max() - combined_scores_k.min())
                tmp_weights_c = ((combined_scores_c - combined_scores_c.min())) / (
                        combined_scores_c.max() - combined_scores_c.min())

            tmp_weights = 0.5 * tmp_weights_k + 0.5 * tmp_weights_c
            tmp_weights = tmp_weights.view(tmp_weights.shape[0], 1, 1, 1)

            factor = 1

            # --------- Specific Gen ---------
            sep_grads = None

            for b in range(context_tokens.shape[0]):
                tmp_sep_norms = [[(torch.norm(x.grad[b:(b + 1)] * window_mask[b:(b + 1)]) + 1e-15) for x in p_]
                                 for p_ in curr_shift]

                # normalize gradients
                tmp_grad = [tuple([-self.stepsize * factor * (
                        x.grad[b:(b + 1)] * window_mask[b:(b + 1)] / tmp_sep_norms[i][
                    j] ** self.grad_norm_factor).data.cpu().numpy()
                                   for j, x in enumerate(p_)])
                            for i, p_ in enumerate(curr_shift)]
                if sep_grads is None:
                    sep_grads = tmp_grad
                else:
                    for l_index in range(len(sep_grads)):
                        sep_grads[l_index] = list(sep_grads[l_index])
                        for k_index in range(len(sep_grads[0])):
                            sep_grads[l_index][k_index] = np.concatenate(
                                (sep_grads[l_index][k_index], tmp_grad[l_index][k_index]), axis=0)
                        sep_grads[l_index] = tuple(sep_grads[l_index])
            final_grads = sep_grads

            # --------- update context ---------
            context_delta = list(map(add_context, final_grads, context_delta))

            for p0, p1 in curr_shift:
                p0.grad.data.zero_()
                p1.grad.data.zero_()

            new_context = []
            for p0, p1 in context:
                new_context.append((p0.detach(), p1.detach()))
            context = new_context

        context_delta = [tuple([torch.from_numpy(x).requires_grad_(True).to(device=self.device) for x in p_])
                         for p_ in context_delta]
        context = list(map(add_context, context, context_delta))

        new_context = []
        for p0, p1 in context:
            new_context.append((p0.detach(), p1.detach()))
        context = new_context

        return context

    def update_special_tokens_logits(self, context_tokens, i, logits):
        for beam_id in range(context_tokens.shape[0]):
            for token_idx in set(context_tokens[beam_id][-4:].tolist()):
                factor = self.repetition_penalty if logits[beam_id, token_idx] > 0 else (1 / self.repetition_penalty)
                logits[beam_id, token_idx] /= factor

            if i >= self.ef_idx:
                factor = self.end_factor if logits[beam_id, self.end_token] > 0 else (1 / self.end_factor)
                logits[beam_id, self.end_token] *= factor
            if i == 0:
                start_factor = 1.6
                factor = start_factor if logits[beam_id, self.end_token] > 0 else (1 / start_factor)
                logits[beam_id, self.end_token] /= factor

            for token_idx in list(self.forbidden_tokens):
                factor = self.forbidden_factor if logits[beam_id, token_idx] > 0 else (1 / self.forbidden_factor)
                logits[beam_id, token_idx] /= factor

        return logits

    def clip_loss(self, probs, context_tokens):
        for p_ in self.clip.transformer.parameters():
            if p_.grad is not None:
                p_.grad.data.zero_()

        top_size = 512
        _, top_indices = probs.topk(top_size, -1)

        prefix_texts = [self.lm_tokenizer.decode(x).replace(self.lm_tokenizer.bos_token, '') for x in context_tokens]

        clip_loss = 0
        losses = []
        for idx_p in range(probs.shape[0]):
            top_texts = []
            prefix_text = prefix_texts[idx_p]
            for x in top_indices[idx_p]:
                top_texts.append(prefix_text + self.lm_tokenizer.decode(x))
            text_features = self.get_txt_features(top_texts)

            with torch.no_grad():
                similiraties = (self.image_features @ text_features.T)
                target_probs = nn.functional.softmax(similiraties / self.clip_loss_temperature, dim=-1).detach()
                target_probs = target_probs.type(torch.float32)

            target = torch.zeros_like(probs[idx_p])
            target[top_indices[idx_p]] = target_probs[0]
            target = target.unsqueeze(0)
            cur_clip_loss = torch.sum(-(target * torch.log(probs[idx_p:(idx_p + 1)])))

            clip_loss += cur_clip_loss
            losses.append(cur_clip_loss)

        return clip_loss, losses
