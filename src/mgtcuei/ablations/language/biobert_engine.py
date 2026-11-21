"""将冻结训练引擎的检查点调用分派到 BioBERT 检查点约定。

复用 mgtcuei.engine.train 的打乱顺序、微批次累积、checkpoint_every_steps、早停、四级 checkpoint_key 选择及一次性验证阈值。BioBERT 的 [CLS] 读出可训练，标准 mgtcuei.checkpoint.restore 要求 trainable_classifier_parameters == 0，且拒绝 head. 前缀，因此使用 BioBERT 专用序列化。

BioBERTArmRunner 在其独立运行进程中替换 checkpoint.save 和 checkpoint.restore。仅 BioBERTEvidenceModel 使用定制分派，其他模型仍走原函数。install() 可重复调用，uninstall() 恢复原函数。
"""

from __future__ import annotations

from mgtcuei import checkpoint as _checkpoint  # noqa: E402

from .biobert_model import BioBERTEvidenceModel  # noqa: E402

_ORIGINAL_SAVE = _checkpoint.save
_ORIGINAL_RESTORE = _checkpoint.restore
_installed = False


def installed() -> bool:
    return _installed


def _save(path, model, optimizer, spec_id, state):
    if isinstance(model, BioBERTEvidenceModel):
        from . import biobert_model
        return biobert_model.save(path, model, optimizer, spec_id, state)
    return _ORIGINAL_SAVE(path, model, optimizer, spec_id, state)


def _restore(path, model, optimizer, spec_id, rng=True, expect_readout=True):
    if isinstance(model, BioBERTEvidenceModel):
        from . import biobert_model
        return biobert_model.restore(path, model, optimizer, spec_id, rng=rng)
    return _ORIGINAL_RESTORE(path, model, optimizer, spec_id,
                             rng=rng, expect_readout=expect_readout)


def install() -> None:
    """以幂等方式为冻结训练引擎安装 BioBERT 检查点分派。"""
    global _installed
    if _installed:
        return
    _checkpoint.save = _save
    _checkpoint.restore = _restore
    _installed = True


def uninstall() -> None:
    global _installed
    if not _installed:
        return
    _checkpoint.save = _ORIGINAL_SAVE
    _checkpoint.restore = _ORIGINAL_RESTORE
    _installed = False
