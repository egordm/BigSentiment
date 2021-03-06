import os
from dataclasses import dataclass
from typing import Union, List, Optional, Tuple, Dict, Set

import torch
# import wandb
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerBase


class ElectraPretrainingModel(torch.nn.Module):
    def __init__(
            self,
            discriminator: PreTrainedModel,
            generator: PreTrainedModel,
            tokenizer: PreTrainedTokenizer,
            device: Union[torch.device, int, str] = None,
            special_tokens: Optional[List[int]] = None,
            loss_weights: List[int] = None
    ):
        super().__init__()
        self.device = device
        self.loss_weights = loss_weights if loss_weights else [1, 50]

        # Initialize tokenizer and ignored token ids
        self.tokenizer = tokenizer
        self.special_tokens = special_tokens if special_tokens else tokenizer.all_special_ids

        # Initialize model components
        self.discriminator = discriminator.to(device)
        self.generator = generator.to(device)

        # Embeddings are shared
        self.discriminator.set_input_embeddings(self.generator.get_input_embeddings())

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            labels=None,
            masked_indices=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
    ):
        batch_size, sequence_len = input_ids.shape
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        labels = labels.to(self.device)
        masked_indices = masked_indices.to(self.device)

        # Create labels for the generator
        generator_labels = labels.clone()
        generator_labels[~masked_indices] = -100
        # Train the generator to predict the sentence / fill the masked spots
        generator_loss, generator_output = self.generator(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=generator_labels
        )[:2]

        # Replace the masked tokens with generated values
        masked_lm_positions = torch.nonzero(masked_indices.view(-1)).squeeze()
        fake_logits = generator_output.view((batch_size * sequence_len, -1))[masked_lm_positions]
        fake_argmaxes = fake_logits.argmax(-1)
        fake_tokens = input_ids.view(-1).scatter(-1, masked_lm_positions, fake_argmaxes).view((batch_size, sequence_len))
        fake_labels = (~torch.eq(labels, fake_tokens)).type(torch.float32)

        # Train discriminator to predict which positions are bad
        discriminator_loss, discriminator_output = self.discriminator(
            fake_tokens,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            position_ids,
            labels=fake_labels
        )[:2]
        discriminator_predictions = torch.round((torch.sign(discriminator_output) + 1) / 2).int().tolist()
        total_loss = generator_loss * self.loss_weights[0] + discriminator_loss * self.loss_weights[1]

        # Log some extras
        wandb.log({'discriminator_loss': discriminator_loss, 'generator_loss': generator_loss, 'total_loss': total_loss})
        return (
            total_loss,
            (discriminator_predictions, generator_output),
            (fake_tokens, labels)
        )

    def save_pretrained(self, directory):
        generator_path = os.path.join(directory, "generator")
        discriminator_path = os.path.join(directory, "discriminator")

        if not os.path.exists(generator_path):
            os.makedirs(generator_path)

        if not os.path.exists(discriminator_path):
            os.makedirs(discriminator_path)

        self.generator.save_pretrained(generator_path)
        self.discriminator.save_pretrained(discriminator_path)


@dataclass
class DataCollatorForElectra:
    tokenizer: PreTrainedTokenizerBase
    mlm: bool = True
    mlm_probability: float = 0.15
    special_tokens: Set[int] = None

    def __post_init__(self):
        if self.special_tokens is None:
            self.special_tokens = set(self.tokenizer.all_special_ids)

    def __call__(
            self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        batch = self.tokenizer.pad(examples, return_tensors="pt")

        input_ids = batch['input_ids']
        labels = input_ids.clone()

        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = torch.tensor([[int(t) in self.special_tokens for t in item] for item in input_ids])
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()

        # we replace masked input tokens with tokenizer.mask_token ([MASK])
        input_ids[masked_indices] = self.tokenizer.mask_token_id

        batch['input_ids'] = input_ids
        batch['labels'] = labels
        batch['masked_indices'] = masked_indices
        return batch
