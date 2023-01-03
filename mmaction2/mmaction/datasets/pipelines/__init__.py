# Copyright (c) OpenMMLab. All rights reserved.
from .augmentations import (AudioAmplify, CenterCrop, ColorJitter, Flip, Fuse,
                            Imgaug, MelSpectrogram, MultiScaleCrop, Normalize,
                            PytorchVideoTrans, RandomCrop, RandomRescale,
                            RandomResizedCrop, Resize, TenCrop, ThreeCrop,
                            TorchvisionTrans, RandomErasing,
                            RandomRescaleWithGT, ResizeWithGT, RandomCropWithGT, FlipWithGT) # RandomErasing is added by Swin
from .compose import Compose
from .formatting import (Collect, FormatAudioShape, FormatGCNInput,
                         FormatShape, ImageToTensor, JointToBone, Rename,
                         ToDataContainer, ToTensor, Transpose, AddPadShape, FilterGT)
from .loading import (ArrayDecode, AudioDecode, AudioDecodeInit,
                      AudioFeatureSelector, BuildPseudoClip, DecordDecode,
                      DecordInit, DenseSampleFrames,
                      GenerateLocalizationLabels, ImageDecode,
                      LoadAudioFeature, LoadHVULabel, LoadLocalizationFeature,
                      LoadProposals, OpenCVDecode, OpenCVInit, PIMSDecode,
                      PIMSInit, PyAVDecode, PyAVDecodeMotionVector, PyAVInit,
                      RawFrameDecode, SampleAVAFrames, SampleUnlabelAVAFrames, SampleFrames,
                      SampleProposalFrames, UntrimmedSampleFrames,
                      RawFrameDecodeWithGT)
from .pose_loading import (GeneratePoseTarget, LoadKineticsPose,
                           PaddingWithLoop, PoseDecode, PoseNormalize,
                           UniformSampleFrames)

# ssad
from .ssad_formatting import ExtraAttrs, ExtraCollect, PseudoSamples, MultiBranch

__all__ = [
    'SampleFrames', 'PyAVDecode', 'DecordDecode', 'DenseSampleFrames',
    'OpenCVDecode', 'MultiScaleCrop', 'RandomResizedCrop', 'RandomCrop',
    'Resize', 'Flip', 'Fuse', 'Normalize', 'ThreeCrop', 'CenterCrop',
    'TenCrop', 'ImageToTensor', 'Transpose', 'Collect', 'FormatShape',
    'Compose', 'ToTensor', 'ToDataContainer', 'GenerateLocalizationLabels',
    'LoadLocalizationFeature', 'LoadProposals', 'DecordInit', 'OpenCVInit',
    'PyAVInit', 'SampleProposalFrames', 'UntrimmedSampleFrames',
    'RawFrameDecode', 'DecordInit', 'OpenCVInit', 'PyAVInit',
    'SampleProposalFrames', 'ColorJitter', 'LoadHVULabel', 'SampleAVAFrames',
    'AudioAmplify', 'MelSpectrogram', 'AudioDecode', 'FormatAudioShape',
    'LoadAudioFeature', 'AudioFeatureSelector', 'AudioDecodeInit',
    'ImageDecode', 'BuildPseudoClip', 'RandomRescale',
    'PyAVDecodeMotionVector', 'Rename', 'Imgaug', 'UniformSampleFrames',
    'PoseDecode', 'LoadKineticsPose', 'GeneratePoseTarget', 'PIMSInit',
    'PIMSDecode', 'TorchvisionTrans', 'PytorchVideoTrans', 'PoseNormalize',
    'FormatGCNInput', 'PaddingWithLoop', 'ArrayDecode', 'JointToBone', 'RandomErasing', # RandomErasing is added by Swin
    "AddPadShape", "FilterGT",
    "ExtraAttrs", "ExtraCollect", "PseudoSamples", "MultiBranch", "SampleUnlabelAVAFrames",
    "RandomRescaleWithGT", "ResizeWithGT", "RandomCropWithGT", "FlipWithGT", "RawFrameDecodeWithGT"
]
