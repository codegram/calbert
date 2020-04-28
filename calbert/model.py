import torch
from transformers import AlbertForMaskedLM


class CalbertForMaskedLM(AlbertForMaskedLM):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, input):
        input_ids, masked_lm_labels, attention_mask, token_type_ids = input.permute(
            1, 0, 2
        )

        position_ids = None
        head_mask = None
        inputs_embeds = None

        outputs = self.albert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        sequence_outputs = outputs[0]

        prediction_scores = self.predictions(sequence_outputs)

        outputs = (prediction_scores,) + outputs[
            2:
        ]  # Add hidden states and attention if they are here
        if masked_lm_labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size),
                masked_lm_labels.reshape(-1),
            )
            outputs = (masked_lm_loss,) + outputs

        return outputs
