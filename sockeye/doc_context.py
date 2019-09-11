from typing import Optional, Tuple

from . import config

OUTSIDE_DECODER = "outside-decoder"
INSIDE_DECODER_PARALLEL = "inside-decoder-parallel"
INSIDE_DECODER_SEQUENTIAL = "inside-decoder-sequential"
ARCHITECTURE_CHOICES = [OUTSIDE_DECODER, INSIDE_DECODER_PARALLEL, INSIDE_DECODER_SEQUENTIAL]


class WindowConfig(config.Config):
    def __init__(self,
                 src_pre: int,
                 src_nxt: int,
                 tar_pre: int,
                 tar_nxt: int):
        super().__init__()
        self.src_pre = src_pre
        self.src_nxt = src_nxt
        self.tar_pre = tar_pre
        self.tar_nxt = tar_nxt

    @property
    def sizes(self) -> Tuple[int, int, int, int]:
        return self.src_pre, self.src_nxt, self.tar_pre, self.tar_nxt

    @property
    def number_source_side(self):
        return self.src_pre + self.src_nxt

    @property
    def number_target_side(self):
        return self.tar_pre + self.tar_nxt

    @property
    def use_source_side(self):
        return self.number_source_side > 0

    @property
    def use_target_side(self):
        return self.number_target_side > 0

    @property
    def source_side_only(self):
        return self.use_source_side and not self.use_target_side

    @property
    def target_side_only(self):
        return not self.use_source_side and self.use_target_side

    @property
    def use_both_sides(self):
        return self.use_source_side and self.use_target_side

    @property
    def is_used(self):
        return self.use_source_side or self.use_target_side


class DocumentContextConfig(config.Config):
    def __init__(self,
                 method: str,
                 window_config: WindowConfig,
                 source_train: Optional[str],
                 source_validation: Optional[str],
                 target_train: Optional[str],
                 target_validation: Optional[str],
                 bucket_width: int):
        super().__init__()
        self.method = method
        self.window_config = window_config
        self.source_train = source_train
        self.target_train = target_train
        self.source_validation = source_validation
        self.target_validation = target_validation
        self.bucket_width = bucket_width