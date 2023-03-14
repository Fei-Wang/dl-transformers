from transformers import AutoConfig, AutoModelForSequenceClassification

from .modeling_rm import RMModel, RMConfig

AutoConfig.register("nami_rm", RMConfig)
AutoModelForSequenceClassification.register(RMConfig, RMModel)
