import struct
import base64
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn

import transformers
from transformers.utils import ModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.models.bert.modeling_bert import (
    BertModel,
    BertForMaskedLM,
)

from accelerate import Accelerator
accelerator = Accelerator()

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = accelerator.device


class InstructionTraceConfig(transformers.BertConfig):
    def __init__(
        self,
        next_token_id=1,
        embedding_size=128,
        embedding_dropout_prob=0.1,
        frozen_encoder=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.next_token_id = next_token_id
        self.embedding_size = embedding_size
        self.embedding_dropout_prob = embedding_dropout_prob
        self.frozen_encoder = frozen_encoder


class InstructionTracePositionEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.next_token_id = config.next_token_id

        self.token = nn.Embedding(config.vocab_size, config.hidden_size)
        ##self.function = nn.Linear(config.max_position_embeddings, config.hidden_size)  #embed pe into higher dim space
        self.instruction = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.argument = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        past_key_values_length=0,
    ):
        assert token_type_ids is None or token_type_ids.sum() == 0
        assert position_ids is None
        assert inputs_embeds is None
        assert past_key_values_length == 0
        assert input_ids.dim() <= 2

        #print(input_ids.dim())
        #print(input_ids.size())
        if input_ids.dim() == 1:
            tokens = input_ids.unsqueeze(0)
        else:
            tokens = input_ids
        
        #print(position_ids.dim())
        #print(position_ids.size())
        ##if position_ids.dim() == 2:
            ##position_ids = position_ids.unsqueeze(1)
        #print(position_ids.dim())
        #print(position_ids.size())

        starts = torch.roll(tokens == self.next_token_id, 1)
        starts[:, 0] = False
        instructions = torch.cumsum(starts, dim=-1)

        arguments = torch.zeros(instructions.shape, dtype=torch.long, device=device)
        for i, batch in enumerate(instructions):
            arguments[i] = torch.cat([torch.arange(v) for v in torch.bincount(batch)])

        tokens = self.token(tokens)
        ##functions = self.function(position_ids)
        instructions = self.instruction(instructions)
        arguments = self.argument(arguments)
        
        #print(tokens.size(), functions.size(), instructions.size(), arguments.size())

        ##embedded = tokens + functions + instructions + arguments
        embedded = tokens + instructions + arguments

        embedded = self.norm(embedded)
        embedded = self.dropout(embedded)

        return embedded


class InstructionTraceEncoderTransformer(BertModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer)

        self.embeddings = InstructionTracePositionEmbedding(config)

    def get_input_embeddings(self):
        return self.embeddings.token

    def set_input_embeddings(self, value):
        self.embeddings.token = value


class InstructionTraceEncoderTransformerForMaskedLM(BertForMaskedLM):
    def __init__(self, config):
        super().__init__(config)

        self.bert = InstructionTraceEncoderTransformer(config, add_pooling_layer=False)


@dataclass
class EmbeddingModelOutput(ModelOutput):
    embedded: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

    def encode(self):
        data = self.embedded.squeeze().tolist()
        data = struct.pack(f"{len(data)}f", *data)
        data = base64.b64encode(data).decode("utf-8")

        return data

    @classmethod
    def decode(cls, data):
        data = base64.b64decode(data)
        data = struct.unpack(f"{len(data)//4}f", data)
        data = torch.tensor(data).unsqueeze(0).to(device)

        return cls(embedded=data)


class InstructionTraceEmbeddingHead(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.linear1 = nn.Linear(config.hidden_size, config.hidden_size // 2)
        self.linear2 = nn.Linear(config.hidden_size // 2, config.embedding_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(config.embedding_dropout_prob)

    def forward(self, inputs):
        outputs = self.linear1(inputs)
        outputs = self.activation(outputs)
        outputs = self.linear2(outputs)
        outputs = self.dropout(outputs)

        return outputs


class InstructionTraceEncoderTransformerForSequenceEmbedding(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = InstructionTraceEncoderTransformer(config, add_pooling_layer=False)

        if config.frozen_encoder:
            for parameter in self.bert.parameters():
                parameter.requires_grad = False

        self.embedding = InstructionTraceEmbeddingHead(config)

    #def forward(self, input_ids, attention_mask, position_ids):
    def forward(self, input_ids, attention_mask):
        #outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids)
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Pool sequence model by taking the embedding of the [CLS] token - this
        # is how HuggingFace handles pooling, we just don't want the extra
        # dense layer that they add
        pooled = outputs.last_hidden_state[:, 0]

        # Mean pooling - Trex's approach
        # pooled = output.last_hidden_state.mean(dim=-2)

        embedded = self.embedding(pooled)

        return EmbeddingModelOutput(
            embedded=embedded,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class CosineInstructionTraceDifference(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.activation = nn.Sigmoid()

    def forward(self, first, second):
        cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
        difference = cos(first, second)

        return self.activation(difference)
    

class EuclidianInstructionTraceDifference(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.activation = nn.Sigmoid()

    def forward(self, first, second):
        difference = torch.sqrt(torch.sum((first - second) ** 2, dim=-1))

        return self.activation(difference)


class LearnedInstructionTraceDifference(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.linear = nn.Linear(config.embedding_size, 1)
        self.activation = nn.Sigmoid()

    def forward(self, first, second):
        difference = torch.abs(first - second)
        outputs = self.linear(difference).squeeze()
        outputs = self.activation(outputs)

        return outputs


@dataclass
class SimilarityModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    embedded1: Optional[Tuple[torch.FloatTensor]] = None
    embedded2: Optional[Tuple[torch.FloatTensor]] = None
    hidden_states1: Optional[Tuple[torch.FloatTensor]] = None
    hidden_states2: Optional[Tuple[torch.FloatTensor]] = None
    attentions1: Optional[Tuple[torch.FloatTensor]] = None
    attentions2: Optional[Tuple[torch.FloatTensor]] = None


class InstructionTraceEncoderTransformerForSequenceSimilarity(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.embedding = InstructionTraceEncoderTransformerForSequenceEmbedding(config)
        self.difference = CosineInstructionTraceDifference(config)

    def forward(
        self, input_ids1, attention_mask1, input_ids2, attention_mask2, labels=None
    ):
        embedded1 = self.embedding(input_ids=input_ids1, attention_mask=attention_mask1)
        embedded2 = self.embedding(input_ids=input_ids2, attention_mask=attention_mask2)

        logits = self.difference(embedded1.embedded, embedded2.embedded)

        if labels is not None:
            function = nn.BCELoss()
            loss = function(logits, labels.float())
        else:
            loss = None

        return SimilarityModelOutput(
            loss=loss,
            logits=logits,
            embedded1=embedded1.embedded,
            embedded2=embedded2.embedded,
            hidden_states1=embedded1.hidden_states,
            hidden_states2=embedded2.hidden_states,
            attentions1=embedded1.attentions,
            attentions2=embedded2.attentions,
        )
