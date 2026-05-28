from __future__ import annotations

import types

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.modules.head import Detect, Detect26
from ultralytics.utils.loss import E2ELoss, v8DetectionLoss26


def _install_dual_task_criterion(model):
    """Replace ``init_criterion`` on a DetectionModel so it returns the dual-task loss.

    The model's ``loss()`` lazily calls ``init_criterion`` (also re-called on resume),
    so monkey-patching here ensures every code path picks up v8DetectionLoss26 / E2ELoss.
    """

    def init_criterion(self):
        return (
            E2ELoss(self, loss_fn=v8DetectionLoss26)
            if getattr(self, "end2end", False)
            else v8DetectionLoss26(self)
        )

    model.init_criterion = types.MethodType(init_criterion, model)
    model.criterion = None  # force lazy re-init on first loss() call


class DualTaskTrainer(DetectionTrainer):
    """DetectionTrainer that swaps the standard Detect head for Detect26.

    Pipeline:
      1. Let the parent build a DetectionModel from the .pt's yaml + load weights.
      2. Replace ``model.model[-1]`` with a fresh Detect26(nc=5, ex_nc=3).
      3. Copy pretrained box-regression weights (cv2 / one2one_cv2) over.
      4. Carry over stride and parse_model metadata (i / f / type).
      5. Re-run ``bias_init`` so cv3 / cv4 biases are properly seeded.
      6. Monkey-patch ``init_criterion`` so the model uses v8DetectionLoss26.
    """

    NC_SHAPE = 5
    EX_NC = 3

    # def get_model(self, cfg=None, weights=None, verbose=True):
    #     model = super().get_model(cfg, weights, verbose)
    #     old_head = model.model[-1]
    #     if not isinstance(old_head, Detect):
    #         raise TypeError(f"Expected a Detect-like head, got {type(old_head).__name__}.")

    #     # Resume / fine-tune from an already-dual-task checkpoint: keep the existing head.
    #     if isinstance(old_head, Detect26):
    #         model.nc = self.NC_SHAPE
    #         model.ex_nc = self.EX_NC
    #         _install_dual_task_criterion(model)
    #         return model

    #     # Extract per-level input channels from the original box-regression branch.
    #     ch = tuple(layer[0].conv.in_channels for layer in old_head.cv2)
    #     end2end = bool(getattr(old_head, "end2end", False))
    #     reg_max = int(getattr(old_head, "reg_max", 1))

    #     new_head = Detect26(
    #         nc=self.NC_SHAPE,
    #         ex_nc=self.EX_NC,
    #         reg_max=reg_max,
    #         end2end=end2end,
    #         ch=ch,
    #     )

    #     # 1) Preserve parse_model metadata so the model's forward routing still works.
    #     new_head.i = old_head.i
    #     new_head.f = old_head.f
    #     new_head.type = type(new_head).__name__

    #     # 2) Carry over stride (set by build_strides in DetectionModel.__init__).
    #     new_head.stride = old_head.stride.clone()
    #     new_head.training = old_head.training

    #     # 3) Transfer pretrained box-regression weights (same architecture as old Detect).
    #     try:
    #         new_head.cv2.load_state_dict(old_head.cv2.state_dict())
    #         if end2end and hasattr(old_head, "one2one_cv2") and hasattr(new_head, "one2one_cv2"):
    #             new_head.one2one_cv2.load_state_dict(old_head.one2one_cv2.state_dict())
    #     except Exception as exc:  # shape mismatch -> keep random init, just log it
    #         print(f"[DualTaskTrainer] Skipped cv2 weight transfer: {exc}")

    #     # 4) Match the device/dtype of the rest of the model.
    #     new_head = new_head.to(next(old_head.parameters()).device)

    #     model.model[-1] = new_head
    #     # Refresh parameter count cache for nice info() print.
    #     new_head.np = sum(p.numel() for p in new_head.parameters())

    #     # 5) Initialise biases now that stride is set on the new head.
    #     new_head.bias_init()

    #     # 6) Wire in dual-task loss and refresh top-level model attributes.
    #     model.nc = self.NC_SHAPE
    #     model.ex_nc = self.EX_NC
    #     model.yaml["nc"] = self.NC_SHAPE
    #     model.yaml["ex_nc"] = self.EX_NC
    #     _install_dual_task_criterion(model)

    #     return model

    def get_model(self, cfg=None, weights=None, verbose=True):
        # 1. 确保 cfg 是字典格式，以便我们可以安全地读取和动态修改它
        if isinstance(cfg, str):
            from ultralytics.nn.tasks import yaml_model_load
            cfg = yaml_model_load(cfg)
            
        # 2. 动态改写：如果检测到配置中的最后一层是单头 "Detect"
        # 自动将其改写为双头 "Detect26"，并传入对应的 [NC_SHAPE, EX_NC] 两个分类维度。
        # 这确保了 super().get_model 刚实例化模型时，就已经天然是双头 Detect26 结构！
        if isinstance(cfg, dict) and cfg.get("head") and cfg["head"][-1][2] == "Detect":
            cfg["head"][-1][2] = "Detect26"
            cfg["head"][-1][3] = [self.NC_SHAPE, self.EX_NC]

        # 3. 临时将 self.data["nc"] 覆盖修改为 5 (NC_SHAPE)，使分类维度和权重加载对齐
        orig_nc = self.data.get("nc", 15)
        self.data["nc"] = self.NC_SHAPE
        
        try:
            model = super().get_model(cfg, weights, verbose)
        finally:
            # 权重加载成功后，确保将 self.data["nc"] 还原回 15
            self.data["nc"] = orig_nc

        old_head = model.model[-1]
        if not isinstance(old_head, Detect):
            raise TypeError(f"Expected a Detect-like head, got {type(old_head).__name__}.")

        # 4. 完美接管：由于上面我们动态改写了 cfg，构建出来的 old_head 100% 是 Detect26。
        # 同时，所有权重（cv2、cv3、cv4 及其 one2one 对应分支）已经由 model.load() 完美、无损地从 checkpoint 还原！
        # 我们在此直接配置 Loss 并返回模型，不进行任何重写、替换或随机化。
        if isinstance(old_head, Detect26):
            model.nc = self.NC_SHAPE
            model.ex_nc = self.EX_NC
            model.yaml["nc"] = self.NC_SHAPE
            model.yaml["ex_nc"] = self.EX_NC
            _install_dual_task_criterion(model)
            return model

        # 5. 后备替换方案（情形 1）：仅当 cfg 中真的不是 Detect 时才会执行
        ch = tuple(layer[0].conv.in_channels for layer in old_head.cv2)
        end2end = bool(getattr(old_head, "end2end", False))
        reg_max = int(getattr(old_head, "reg_max", 1))

        new_head = Detect26(
            nc=self.NC_SHAPE,
            ex_nc=self.EX_NC,
            reg_max=reg_max,
            end2end=end2end,
            ch=ch,
        )

        new_head.i = old_head.i
        new_head.f = old_head.f
        new_head.type = type(new_head).__name__
        new_head.stride = old_head.stride.clone()
        new_head.training = old_head.training

        try:
            new_head.cv2.load_state_dict(old_head.cv2.state_dict())
            if end2end and hasattr(old_head, "one2one_cv2") and hasattr(new_head, "one2one_cv2"):
                new_head.one2one_cv2.load_state_dict(old_head.one2one_cv2.state_dict())
        except Exception as exc:
            print(f"[DualTaskTrainer] Skipped cv2 weight transfer: {exc}")

        new_head = new_head.to(next(old_head.parameters()).device)
        model.model[-1] = new_head
        new_head.np = sum(p.numel() for p in new_head.parameters())
        new_head.bias_init()

        model.nc = self.NC_SHAPE
        model.ex_nc = self.EX_NC
        model.yaml["nc"] = self.NC_SHAPE
        model.yaml["ex_nc"] = self.EX_NC
        _install_dual_task_criterion(model)

        return model

    def set_model_attributes(self):
        """Keep dataset names (15 classes) for the validator, but force head nc=5/ex_nc=3."""
        super().set_model_attributes()
        # Re-pin the dual-task dimensions in case the parent overwrote them.
        self.model.nc = self.NC_SHAPE
        self.model.ex_nc = self.EX_NC
        # The criterion may have been stripped on resume; re-install just in case.
        _install_dual_task_criterion(self.model)
        