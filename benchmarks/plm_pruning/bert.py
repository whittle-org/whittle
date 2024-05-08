from transformers.models.bert.modeling_bert import (
    BertForSequenceClassification,
    BertForMultipleChoice,
)

from model_wrapper.mask import mask_bert
from search_spaces import (
    SmallSearchSpace,
    LayerSearchSpace,
    FullSearchSpace,
    MediumSearchSpace,
)


class BERTSuperNetMixin(object):
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


class BERTSuperNetMixinLAYERSpace(BERTSuperNetMixin):
    @property
    def search_space(self):
        return LayerSearchSpace(self.config)


class BERTSuperNetMixinMEDIUMSpace(BERTSuperNetMixin):
    @property
    def search_space(self):
        return MediumSearchSpace(self.config)


class BERTSuperNetMixinLARGESpace(BERTSuperNetMixin):
    @property
    def search_space(self):
        return FullSearchSpace(self.config)


class BERTSuperNetMixinSMALLSpace(BERTSuperNetMixin):
    @property
    def search_space(self):
        return SmallSearchSpace(self.config)


class SuperNetBertForSequenceClassificationSMALL(
    BertForSequenceClassification, BERTSuperNetMixinSMALLSpace
):
    def forward(self, inputs, **kwargs):
        return super().forward(**inputs)


class SuperNetBertForMultipleChoiceSMALL(
    BertForMultipleChoice, BERTSuperNetMixinSMALLSpace
):
    def forward(self, inputs, **kwargs):
        return super().forward(**inputs)


class SuperNetBertForSequenceClassificationLAYER(
    BertForSequenceClassification, BERTSuperNetMixinLAYERSpace
):
    def forward(self, inputs, **kwargs):
        return super().forward(**inputs)


class SuperNetBertForMultipleChoiceLAYER(
    BertForMultipleChoice, BERTSuperNetMixinLAYERSpace
):
    def forward(self, inputs, **kwargs):
        return super().forward(**inputs)


class SuperNetBertForSequenceClassificationMEDIUM(
    BertForSequenceClassification, BERTSuperNetMixinMEDIUMSpace
):
    def forward(self, inputs, **kwargs):
        return super().forward(**inputs)


class SuperNetBertForMultipleChoiceMEDIUM(
    BertForMultipleChoice, BERTSuperNetMixinMEDIUMSpace
):
    def forward(self, inputs, **kwargs):
        return super().forward(**inputs)


class SuperNetBertForSequenceClassificationLARGE(
    BertForSequenceClassification, BERTSuperNetMixinLARGESpace
):
    def forward(self, inputs, **kwargs):
        return super().forward(**inputs)


class SuperNetBertForMultipleChoiceLARGE(
    BertForMultipleChoice, BERTSuperNetMixinLARGESpace
):
    def forward(self, inputs, **kwargs):
        return super().forward(**inputs)
