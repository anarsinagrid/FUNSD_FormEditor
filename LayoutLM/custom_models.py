import torch
import torch.nn as nn
from transformers.models.layoutlmv3.modeling_layoutlmv3 import LayoutLMv3ForTokenClassification
from transformers.modeling_outputs import TokenClassifierOutput

class ConfidenceAwareLayoutLMv3ForTokenClassification(LayoutLMv3ForTokenClassification):
    def __init__(self, config):
        super().__init__(config)
        # Learnable projection for the 1D confidence score into the hidden space
        self.confidence_proj = nn.Linear(1, config.hidden_size)
    
    def forward(
        self,
        input_ids=None,
        bbox=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        pixel_values=None,
        ocr_confidence=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # We must get hidden states to modulate them
        outputs = self.layoutlmv3(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True if ocr_confidence is not None else output_hidden_states,
            return_dict=True,
            pixel_values=pixel_values,
        )

        sequence_output = outputs[0]
        
        # LayoutLMv3 appends image tokens to the end of the text tokens.
        # We only want to classify (and modulate) the text tokens.
        if input_ids is not None:
            seq_len = input_ids.shape[1]
            sequence_output = sequence_output[:, :seq_len, :]

        if ocr_confidence is not None:
            # ocr_confidence comes in as (batch_size, text_seq_len)
            conf_expanded = ocr_confidence.unsqueeze(-1).to(sequence_output.dtype)
            
            # Project confidence to hidden_size and add it
            conf_emb = self.confidence_proj(conf_expanded)
            sequence_output = sequence_output + conf_emb

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            # outputs[2:] corresponds to hidden_states and attentions if they exist
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions,
        )
