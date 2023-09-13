"""
Metrics from: https://cocodataset.org/#stuff-eval
https://arxiv.org/pdf/1605.06211.pdf
"""

from typing import Dict

import torch
import torch.nn as nn
from torchmetrics import (
    PeakSignalNoiseRatio,
    Accuracy,
    ConfusionMatrix,
    JaccardIndex,
    MeanSquaredError,
)
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


class RgbMetrics(nn.Module):
    """RGB related metrics."""

    def __init__(self):
        super().__init__()
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity()
        self.mse = MeanSquaredError()
        self.l1 = nn.L1Loss()

    def get_batch_metrics(
        self,
        preds: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        num_iter_prop=0,
    ) -> Dict[str, torch.Tensor]:
        """Compute metrics for point-wise batch."""
        batch_metrics_dict = dict()
        batch_metrics_dict["rgb/psnr"] = self.psnr(preds["rgb"], targets["rgb"])

        if "depth" in targets:
            try:
                batch_metrics_dict["rgb/depth_error"] = self.mse(
                    preds["depth"], targets["depth"]
                )
                batch_metrics_dict["rgb/depth_error_l1"] = self.l1(
                    preds["depth"], targets["depth"]
                )
            except:
                # FIXME sth wrong with the data
                batch_metrics_dict["rgb/depth_error"] = self.mse(
                    preds["depth"], targets["depth"][0]
                )
                batch_metrics_dict["rgb/depth_error_l1"] = self.l1(
                    preds["depth"], targets["depth"][0]
                )

        return batch_metrics_dict

    def get_image_metrics(
        self, preds: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute metrics for image-wise batch."""
        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        target = torch.moveaxis(targets["rgb"], -1, 0)[None, ...]
        pred = torch.moveaxis(preds["rgb"], -1, 0)[None, ...]

        image_metrics_dict = dict()
        image_metrics_dict["rgb/psnr"] = float(self.psnr(pred, target).item())
        image_metrics_dict["rgb/ssim"] = float(self.ssim(pred, target))
        image_metrics_dict["rgb/lpips"] = float(self.lpips(pred, target))

        return image_metrics_dict


class SemanticMetrics(nn.Module):
    """Semantic map related metrics."""

    def __init__(self, num_semantic_classes: int):
        super().__init__()
        self.macc = Accuracy(
            task="multiclass",
            num_classes=num_semantic_classes,
            average="macro",
        )
        self.pacc = Accuracy(
            task="multiclass",
            num_classes=num_semantic_classes,
            average="micro",
        )
        self.fiou = JaccardIndex(
            task="multiclass",
            num_classes=num_semantic_classes,
            average="weighted",
        )
        self.miou = JaccardIndex(
            task="multiclass",
            num_classes=num_semantic_classes,
            average="macro",
        )

    def get_batch_metrics(
        self, preds: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute metrics for point-wise batch."""
        pred, target = preds["semantic_pred"], targets["semantic"]

        batch_metrics_dict = dict()
        batch_metrics_dict["semantic/mACC"] = self.macc(pred, target)
        batch_metrics_dict["semantic/pACC"] = self.pacc(pred, target)
        batch_metrics_dict["semantic/fwIoU"] = self.fiou(pred, target)
        batch_metrics_dict["semantic/mIoU"] = self.miou(pred, target)

        return batch_metrics_dict

    def get_image_metrics(
        self,
        preds: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute metrics for image-wise batch."""
        mask = targets["mask"]
        pred, target = preds["semantic_pred"][mask], targets["semantic"][mask]

        image_metrics_dict = dict()
        image_metrics_dict["semantic/mACC"]: self.macc(pred, target)
        image_metrics_dict["semantic/pACC"]: self.pacc(pred, target)
        image_metrics_dict["semantic/fwIoU"]: self.fiou(pred, target)
        image_metrics_dict["semantic/mIoU"]: self.miou(pred, target)

        return image_metrics_dict


class BlueprintMetrics(nn.Module):
    """Blueprint related metrics."""

    def __init__(self, num_semantic_classes: int):
        super().__init__()
        self.cacc = Accuracy(
            task="multiclass",
            num_classes=num_semantic_classes,
            average="none",
        )
        self.macc = Accuracy(
            task="multiclass",
            num_classes=num_semantic_classes,
            average="macro",
        )
        self.pacc = Accuracy(
            task="multiclass",
            num_classes=num_semantic_classes,
            average="micro",
        )
        self.ciou = JaccardIndex(
            task="multiclass",
            num_classes=num_semantic_classes,
            average="none",
        )
        self.fiou = JaccardIndex(
            task="multiclass",
            num_classes=num_semantic_classes,
            average="weighted",
        )
        self.miou = JaccardIndex(
            task="multiclass",
            num_classes=num_semantic_classes,
            average="macro",
        )
        self.confusion = ConfusionMatrix(
            task="multiclass",
            num_classes=num_semantic_classes,
            normalize="true",
        )

    def get_batch_metrics(
        self, preds: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute metrics for point-wise batch."""
        pred, target = preds["blunf_pred"], targets["semantic"]

        batch_metrics_dict = dict()
        batch_metrics_dict["blueprint/mACC"] = self.macc(pred, target)
        batch_metrics_dict["blueprint/pACC"] = self.pacc(pred, target)
        batch_metrics_dict["blueprint/fwIoU"] = self.fiou(pred, target)
        batch_metrics_dict["blueprint/mIoU"] = self.miou(pred, target)

        return batch_metrics_dict

    def get_image_metrics(
        self,
        preds: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute metrics for image-wise blueprint."""
        mask = targets["mask"]
        pred, target = preds["blunf_pred"][mask], targets["semantic"][mask]

        try:
            image_metrics_dict = dict()
            image_metrics_dict["blueprint/image/mACC"] = self.macc(pred, target)
            image_metrics_dict["blueprint/image/pACC"] = self.pacc(pred, target)
            image_metrics_dict["blueprint/image/fwIoU"] = self.fiou(pred, target)
            image_metrics_dict["blueprint/image/mIoU"] = self.miou(pred, target)
        except:
            pass

        return image_metrics_dict

    def get_global_metrics(
        self,
        preds: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute metrics for global blueprint."""
        mask = targets["global_mask"]
        pred = preds["global_blueprint_pred"][mask]
        target = targets["global_blueprint"][:, :, 2].to(int).view(-1)[mask]

        global_metrics_dict = dict()
        global_metrics_dict["blueprint/global/mACC"] = self.macc(pred, target)
        global_metrics_dict["blueprint/global/pACC"] = self.pacc(pred, target)
        global_metrics_dict["blueprint/global/fwIoU"] = self.fiou(pred, target)
        global_metrics_dict["blueprint/global/mIoU"] = self.miou(pred, target)
        global_metrics_dict["conf_matrix"] = self.confusion(pred, target)

        return global_metrics_dict
