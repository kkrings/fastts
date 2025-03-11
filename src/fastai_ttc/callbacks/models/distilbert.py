from typing import Annotated

import torch

DistilBERTArgs = tuple[
    Annotated[torch.Tensor, "input_ids"],
    Annotated[torch.Tensor, "attention_mask"],
]
