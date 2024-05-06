from transformers.models.bert.modeling_bert import BertForSequenceClassification

from model_wrapper.mask import mask_bert
from search_spaces import SmallSearchSpace


class BERTSuperNetMixin(object):

    # def __init__(self, config, search_space, add_pooling_layer=True):
    #     super().__init__(config=config, add_pooling_layer=add_pooling_layer)
    #     self.search_space = search_space
    search_space = None
    handles = None

    def select_sub_network(self, sub_network_config):
        head_mask, ffn_mask = self.search_space.config_to_mask(sub_network_config)
        head_mask = head_mask.to(device="cuda", dtype=self.dtype)
        ffn_mask = ffn_mask.to(device="cuda", dtype=self.dtype)
        self.handles = mask_bert(self.bert, ffn_mask, head_mask)

    def reset_super_network(self):
        for handle in self.handles:
            handle.remove()


class BERTSuperNetMixinSMALLSpace(BERTSuperNetMixin):

    @property
    def search_space(self):
        return SmallSearchSpace(self.config)


class SuperNetBertForSequenceClassification(BertForSequenceClassification, BERTSuperNetMixinSMALLSpace):
    def forward(self, inputs, **kwargs):
        return super().forward(**inputs)