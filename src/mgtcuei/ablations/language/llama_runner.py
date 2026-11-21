"""Llama 匹配对照组共享上游 GCN，各自记录读出来源。"""

from ..common import AblationPipeline
from ...utils import identity
from . import backbones
from .language_arms import model_path


class LanguageArmRunner(AblationPipeline):
    def __init__(self, root, arm):
        super().__init__(
            root, root / "results/ablations/language", arm["arm_id"], arm["candidate"],
            str(model_path(root, arm)),
        )
        self.readout_authority = {
            "llm_root": arm["authority_label"],
            "llm_authority": identity({key: arm[key] for key in ("repo_id", "revision", "geometry")}),
            "backbone_identity": arm["repo_id"],
            "revision": arm["revision"],
            "language_arm": self.arm_id,
        }

    def boundary(self):
        if self._boundary is None:
            self._boundary = backbones.load_llama_boundary(self.llm_root, str(self.device))
        return self._boundary
