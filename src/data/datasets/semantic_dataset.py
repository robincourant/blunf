"""
Only semantic dataset.
"""

from typing import Dict, List

import numpy as np
import numpy.typing as npt
from PIL import Image
import torch
from torchtyping import TensorType
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.utils.data_utils import get_semantics_and_mask_tensors_from_path

image_height, image_width, num_channels = None, None, None


class SemanticDataset(InputDataset):
    """Dataset that returns images and semantics and masks.

    Args:
        dataparser_outputs: description of where and how to read input images.
    """

    def __init__(self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1.0):
        super().__init__(dataparser_outputs, scale_factor)

        self.camera_scale_factor = dataparser_outputs.metadata["scale_factor"]
        assert "semantic_filenames" in dataparser_outputs.metadata.keys()
        self.semantic_filenames = self.metadata["semantic_filenames"]

        self.num_masks = 0
        if "semantic_metadata" in self.metadata and self.metadata["semantic_metadata"]:
            semantic_metadata = self.metadata["semantic_metadata"]
            self.num_masks = len(semantic_metadata["class_to_mask"])
            self.lut = self.get_lut(**semantic_metadata)

        if "semantic_masks" in self.metadata:
            self.semantic_masks = self.metadata["semantic_masks"]
            self.mask_indices = torch.tensor(
                [
                    self.semantic_masks.classes.index(mask_class)
                    for mask_class in self.semantic_masks.mask_classes
                ]
            ).view(1, 1, -1)

    @staticmethod
    def get_lut(
        num_classes: int, class_to_keep: List[int], class_to_mask: List[int]
    ) -> torch.Tensor:
        """
        Build a look-up table (LUT) that remap semantic classes to remove useless ones.
        """
        lut = np.arange(0, num_classes + 1)
        class_to_move = np.array([k for k in lut if k not in class_to_keep])
        lut[class_to_move] = -2
        lut[class_to_mask] = -1
        lut[class_to_keep] = np.arange(len(class_to_keep))
        return lut

    def get_numpy_image(self, image_filename: str) -> npt.NDArray[np.uint8]:
        """Returns the image of shape (H, W, 3).

        Args:
            image_idx: The image index in the dataset.
        """
        pil_image = Image.open(image_filename)
        if self.scale_factor != 1.0:
            width, height = pil_image.size
            newsize = (int(width * self.scale_factor), int(height * self.scale_factor))
            pil_image = pil_image.resize(newsize, resample=Image.BILINEAR)
        image = np.array(pil_image, dtype="uint8")  # shape is (h, w, 3 or 4)

        return image

    def get_semantic(
        self, image_idx: int
    ) -> TensorType["image_height", "image_width", "num_channels"]:
        """Returns a semantic image.

        Args:
            image_idx: The image index in the dataset.
        """
        semantic_filename = self.semantic_filenames[image_idx]
        raw_semantic = self.get_numpy_image(semantic_filename)

        # Remap classes to remove useless ones
        semantic = self.lut[raw_semantic] if hasattr(self, "lut") else raw_semantic
        ranking = self.ranking_lut[raw_semantic] if hasattr(self, "lut") else None
        if (semantic == -2).any():
            raise ValueError("Unrecognized classes (check the LUT)")

        semantic = torch.from_numpy(semantic)
        ranking = torch.from_numpy(ranking)

        return semantic, ranking

    def get_data(self, image_idx: int) -> Dict:
        """Returns the ImageDataset data as a dictionary.

        Args:
            image_idx: The image index in the dataset.
        """
        semantic, ranking = self.get_semantic(image_idx)
        data = dict(image_idx=image_idx, semantic=semantic)
        metadata = self.get_metadata(data, image_idx)
        data.update(metadata)
        return data

    def get_metadata(self, data: Dict, image_idx: int) -> Dict:
        metadata = dict()
        if hasattr(self, "semantic_masks"):
            # handle mask
            filepath = self.semantic_masks.filenames[data["image_idx"]]
            _, mask = get_semantics_and_mask_tensors_from_path(
                filepath=filepath,
                mask_indices=self.mask_indices,
                scale_factor=self.scale_factor,
            )
            if "mask" in data.keys():
                mask = mask & data["mask"]
            metadata["mask"] = mask

        if self.num_masks != 0:
            mask = (data["semantic"] != -1).unsqueeze(-1)
            if "mask" in metadata.keys():
                mask = mask & metadata["mask"]
            metadata["mask"] = mask

        return metadata

    def __getitem__(self, image_idx: int) -> Dict:
        data = self.get_data(image_idx)

        return data
