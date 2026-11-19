# Edge Computing Assignment 4

**Hamza Khan | 22i-2149 | Section A**

Question 1: CIFAR-100 teacher/student knowledge distillation.
Question 2: measured, resource-constrained evolutionary NAS on CIFAR-10.

## Install

Python 3.12 was used on Windows; the actual Colab training environment is recorded
in `results/q1_results.json`. Install a suitable PyTorch CUDA build to train on GPU,
or a CPU build for CPU execution, then install the remaining requirements:

```sh
python -m pip install -r requirements.txt
```

`timm==1.0.31` supplies `tf_efficientnet_lite0`. The other backbones are torchvision
ResNet-50, MobileNetV2 and SqueezeNet 1.1. Q1 backbones initialize from public
ImageNet weights and are fully fine-tuned for CIFAR-100. The 100-class heads are new.
Q2 models are initialized randomly and use no pretrained weights.

CIFAR-10 and CIFAR-100 download automatically with torchvision integrity checks.
To reuse existing datasets, set `CIFAR10_ROOT` and `CIFAR100_ROOT` to their respective
dataset folders. Dataset archives/images are not committed.

## Run the experiments

```sh
python train_q1.py --teacher-epochs 15 --student-epochs 10
python nas_search.py
python benchmark.py
python export_onnx.py
python verify.py
```

`python run_all.py` runs these stages in order. Q1 automatically uses CUDA if
available, otherwise CPU. The submitted Q1 training ran on a Colab Tesla T4.
`train_q1.py` reuses an existing teacher checkpoint; use `train_teacher.py` first
to retrain the teacher independently. NAS and deployment probes run on CPU.

Standalone training pipelines:

```sh
python train_teacher.py --epochs 15
python train_baseline.py --architecture mobilenet_v2 --epochs 10
python train_distilled.py --architecture mobilenet_v2 --epochs 10
python train_baseline.py --architecture squeezenet --epochs 10
python train_distilled.py --architecture squeezenet --epochs 10
python train_baseline.py --architecture efficientnet_lite0 --epochs 10
python train_distilled.py --architecture efficientnet_lite0 --epochs 10
```

## Q1 protocol and loss

- Fixed seed 2149: 45,000 training / 5,000 validation images; 10,000 official test images.
- Resize native 32 x 32 RGB to 64 x 64 by bilinear interpolation. Normalize with
  ImageNet mean `(0.485,0.456,0.406)` and std `(0.229,0.224,0.225)`.
- Horizontal-flip augmentation; batch 128; SGD momentum 0.9, weight decay 0.0001,
  cosine LR decay. Teacher: 15 epochs. Each student: 10 epochs.
- Teacher, MobileNetV2 and EfficientNet-Lite0 use LR 0.01. SqueezeNet uses backbone
  LR 0.0001 and classifier/projector LR 0.001. Baseline and KD use the same settings
  within each architecture and the same shuffle/flip stream.
- The best validation checkpoint is selected, then tested. Test labels do not select
  checkpoints or hyperparameters.
- The teacher is frozen during KD. Both original and flipped views have matching
  cached teacher logits and pooled features. Cache tensors are stored in FP16 and
  converted to FP32 for the loss.

`distillation.py` implements:

```text
hard = cross_entropy(student_logits, true_labels)
soft = T^2 * KL(softmax(teacher_logits/T) || softmax(student_logits/T))
feature = mean(sum((normalize(projector(student_features))
                  - normalize(teacher_features))^2))
loss = 0.5*hard + 0.5*soft + 1.0*feature; T = 4
```

Feature dimensions: teacher 2048; MobileNetV2 1280; SqueezeNet 512;
EfficientNet-Lite0 1280. A learned linear projector maps each student feature vector
to 2048 dimensions. It is included in KD training checkpoints for reproducibility
but excluded from executable student/ONNX models and deployment parameter counts.

SqueezeNet uses its standard Fire backbone and a CIFAR-100 raw-logit convolution
head. The classifier's terminal ReLU is replaced with identity, and the new
classifier weights use normal initialization with std 0.01. Initial shared-LR,
clamped-head SqueezeNet pilots collapsed to chance. Their histories/results are
retained in `results/squeezenet_initial_pilot.json`; both final comparisons were
retrained with the corrected matched settings. `repair_squeezenet.py` records
that recovery experiment. Fresh runs of the main pipeline already use the fix.

## Q2 search and constraints

`nas_space.py` supports variable depth (2-5), channels, 3/5 kernels, standard or
depthwise-separable blocks, ReLU/ReLU6/Hardswish, and 24/32/40 input resolutions.
The sampled channel choices are 16/24/32/48/64/96. The initial population also
contains an explicit wider control to demonstrate measured rejection.

`nas_search.py` uses an elitist evolutionary strategy: population 6, three
generations, top two feasible parents, two mutated fields per child, no duplicates.
Each feasible candidate trains for two epochs on the fixed 45,000/5,000 CIFAR-10
split. Predefined limits are **median batch-1 CPU latency <= 1.2 ms** and
**TorchScript executable storage <= 180 KiB (184,320 bytes)**. Actual serialized
files and an isolated CPU probe determine feasibility, not estimated MACs.
Candidates that fail are rejected. Feasible candidates are profiled again after training.

The highest-validation feasible configuration is saved automatically and rebuilt
with fresh random weights for 15 final epochs. No candidate checkpoint is loaded
for retraining. Initial-parameter SHA-256 is recorded and verified. The test set is
evaluated only after architecture selection and fresh training.

Actual selected architecture: input 24 x 24; four blocks; channels 16/24/48/64;
kernels 3/5/3/5; standard/standard/depthwise/depthwise; Hardswish.
18 candidates were evaluated, 16 feasible. Final test accuracy: **68.23%**;
16,770 parameters; 112,960-byte executable; measured median **0.2108 ms**.
This is the best discovered architecture in this small-budget search, not a global
optimality claim.

## Inference without retraining

Choose any of these seven names:

```text
mobilenet_v2_baseline  mobilenet_v2_kd
squeezenet_baseline    squeezenet_kd
efficientnet_lite0_baseline  efficientnet_lite0_kd
nas_best
```

```sh
python inference.py --model mobilenet_v2_kd --image photo.png
python inference.py --model nas_best --image photo.png
python onnx_inference.py --model efficientnet_lite0_kd --image photo.png
python onnx_inference.py --model nas_best --image photo.png
```

`.pth` checkpoints contain weights plus architecture metadata. The provided loader
reconstructs them; `.pt` files are executable TorchScript, and `.onnx` files run with
ONNX Runtime CPU. Only the NAS model uses CIFAR-10 normalization:
mean `(0.4914,0.4822,0.4465)`, std `(0.2470,0.2435,0.2616)`.

## Measurements and evidence

The Windows CPU is Intel Core i7-14700K, using four threads. `cpu_probe.py` runs in
a fresh process per model: 30 warm-ups, 200 timed batch-1 forwards; median/p95 are
reported. Loading, image preprocessing and disk access are excluded. A separate
100-pass phase samples resident process memory every 1 ms. Recorded RSS includes
Python/PyTorch, model and runtime allocations. Windows peak working set is also
recorded; these figures are not activation-only memory or MCU RAM requirements.
Energy and MCU execution were not measured.

The NAS search performs its own constraint probe. `benchmark.py` remeasures the
same saved NAS checkpoint along with the six students; small timing differences
between these independent probes are expected. The NAS executable is preserved.

- `models/`: trained teacher, six final students, final NAS model and deployment exports.
- `nas_candidates/`: actual profiled candidate executables; rejected-before-training
  candidates contain initial weights and are explicitly labeled untrained in the history.
- `results/q1_results.json`: actual Colab environment, histories, seven test summaries,
  class names and per-model test predictions.
- `results/benchmark.csv`, `test_predictions.csv`, `latency_samples.csv`: CPU results
  and all underlying prediction/timing rows.
- `results/onnx_validation.csv`, `onnx_predictions.csv`: full-test ONNX CPU results
  and numerical parity checks for all seven deployment models.
- `results/nas_search_history.csv/.json`, `best_architecture.json`, `nas_final.json`:
  every candidate, resource checks, chosen configuration and fresh retraining.
- `results/verification.json`: independent consistency checks and model SHA-256 hashes.
- PDF report, actual Chrome/Colab screenshots and saved training notebook.

Question 1 accuracy is on CIFAR-100; Question 2 accuracy is on CIFAR-10. Do not
compare the two percentages as if they measured the same task.

## Submission

Submit the GitHub repository link and `22i-2149_A.pdf` to Assignment 4 on Classroom.
The latest instructor announcements supersede the older ZIP instruction on the
assignment cover. **No compressed folder is submitted.**
