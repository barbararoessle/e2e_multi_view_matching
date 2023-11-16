# End2End Multi-View Feature Matching with Differentiable Pose Optimization
This repository contains the implementation of the ICCV 2023 paper: End2End Multi-View Feature Matching with Differentiable Pose Optimization.

[Arxiv](https://arxiv.org/abs/2205.01694) | [Video](https://www.youtube.com/watch?v=uuLb6GfM9Cg) | [Project Page](https://barbararoessle.github.io/e2e_multi_view_matching/)

![](docs/static/images/pipeline.jpg)

## Cloning the repository
The multi-view matching model is implemented in this [fork](https://github.com/barbararoessle/SuperGluePretrainedNetwork), which is included as a submodules, therefore please use `--recursive` to clone the repository: 
```
git clone https://github.com/barbararoessle/e2e_multi_view_matching --recursive
```
Required python packages are listed in `requirements.txt`.
## Preparing datasets
### ScanNet
Extract the ScanNet dataset, e.g., using [SenseReader](https://github.com/ScanNet/ScanNet/tree/master/SensReader/python), and place the files `scannetv2_test.txt`, `scannetv2_train.txt`, `scannetv2_val.txt` from [ScanNet Benchmark](https://github.com/ScanNet/ScanNet/tree/master/Tasks/Benchmark) and the [preprocessed image overlap](https://kaldir.vc.in.tum.de/e2e_multi_view_matching/scannet_overlap.zip) (range [0.4, 0.8]) into the same directory `<data_dir>/scannet`. 
As a result, we have: 
```
<data_dir>/scannet>
└───scans
|   └───scene0000_00
|   |   |   color
|   |   |   depth
|   |   |   intrinsic
|   |   |   pose
|   ...
└───scans_test
|   └───scene0707_00
|   |   |   color
|   |   |   depth
|   |   |   intrinsic
|   |   |   pose
|   ...
└───overlap
|   └───scans
|   |   |   scene0000_00.json
|   |   |   ...
|   └───scans_test
|   |   |   scene0707_00.json
|   |   |   ...
|    scannetv2_train.txt
|    scannetv2_val.txt
|    scannetv2_test.txt
```
### MegaDepth
We follow the preprocessing done by [LoFTR](https://github.com/zju3dv/LoFTR/blob/master/docs/TRAINING.md): The depth maps are used from [original MegaDepth dataset](https://www.cs.cornell.edu/projects/megadepth/), download and extract `MegaDepth_v1` to `<data_dir>/megadepth>`. The [undistorted images and camera parameters](https://kaldir.vc.in.tum.de/e2e_multi_view_matching/preprocessed_megadepth.zip) follow the preprocessing of [D2-Net](https://github.com/mihaidusmanu/d2-net#downloading-and-preprocessing-the-megadepth-dataset), download and extract to `<data_dir>/megadepth>`. As a result, we have:
```
<data_dir>/megadepth>
|    MegaDepth_v1
|    Undistorted_SfM
|    scene_info
|    megadepth_train.txt
|    megadepth_val.txt
|    megadepth_test.txt
|    megadepth_valid_list.json
```
## Downloading pretrained models
Pretrained models are available [here](https://kaldir.vc.in.tum.de/e2e_multi_view_matching/pretrained_network_weights.zip).

## Evaluating on image pairs
Download test pair descriptions `scannet_test_1500` and `megadepth_test_1500_scene_info` from [LoFTR](https://github.com/zju3dv/LoFTR/tree/master/assets) into `assets/`. 
The option `eval_mode` specifies the relative pose estimation method, e.g., weighted eight-point with bundle adjustment (`w8pt_ba`) or RANSAC (`ransac`). 
### ScanNet
```
python3 eval_pairs.py --eval_mode w8pt_ba --dataset scannet --exp_name two_view_scannet --data_dir <path to datasets> --checkpoint_dir <path to pretrained models>
```
### MegaDepth
```
python3 eval_pairs.py --eval_mode w8pt_ba --dataset megadepth --exp_name two_view_megadepth --data_dir <path to datasets> --checkpoint_dir <path to pretrained models>
```

## Evaluating on multi-view
To run multi-view evaluation, the bundle adjustment using Ceres solver needs to be build. 
```
cd pose_optimization/multi_view/bundle_adjustment
mkdir build
cd build
cmake ..
make -j
```
It has the following dependencies:
- Ceres Solver, http://ceres-solver.org/installation.html (tested with version 2.0.0)
- Theia Vision Library, http://theia-sfm.org/building.html (tested with version 0.7.0)
- Eigen https://eigen.tuxfamily.org/dox/GettingStarted.html (tested with version 3.3.7)
- Boost https://www.boost.org/ (tested with version 1.71.0)
- GoogleTest https://github.com/google/googletest (tested with version 1.10.0)

### ScanNet
```
python3 eval_multi_view.py --dataset scannet --exp_name multi_view_scannet --data_dir <path to datasets> --checkpoint_dir <path to pretrained models>
```
### MegaDepth
To simplify internal processing, we convert the MegaDepth data to the same data format as ScanNet. It will be written to `<path to datasets>/megadepth_640`:
```
python3 convert_megadepth_to_scannet_format.py --dataset_dir <path to datasets>/megadepth --image_size 640
```
```
python3 eval_multi_view.py --dataset megadepth_640 --exp_name multi_view_megadepth --data_dir <path to datasets> --checkpoint_dir <path to pretrained models>
```

## Training
Training stage 1 trains without pose loss, stage 2 with pose loss. Checkpoints are written into a subdirectory of the provided checkpoint directory. The subdirectory is named by the training start time of stage 1 or 2 in the format `jjjjmmdd_hhmmss`, which is the experiment name. The experiment name can be specified to resume a training or it is used to initialize stage 2 or to run evaluation. 
## Training on image pairs
### ScanNet
**Stage 1**
```
python3 -u -m torch.distributed.launch --nproc_per_node=2 --rdzv_endpoint=127.0.0.1:29109 train.py --tuple_size 2 --dataset scannet --batch_size 32 --n_workers 12 --data_dir <path to datasets> --checkpoint_dir <path to write checkpoints>
```
Training stage 1 is trained until the validation matching loss is converged. 

**Stage 2**

Training stage 2 trains with pose loss and loads the checkpoint from stage 1, therefore the following options are added to stage 1 command: 
```
--init_exp_name <experiment name from stage 1> --pose_loss
```
Training stage 2 is trained until the validation rotation and translation losses are converged. 

### MegaDepth
To simplify internal processing, we convert the MegaDepth data to the same data format as ScanNet. Note that for image pairs `image_size=720` is used (following SuperGlue), whereas for multi-view `image_size=640` is used for computational reasons (following LoFTR). It will be written to `<path to datasets>/megadepth_720`:
```
python3 convert_megadepth_to_scannet_format.py --dataset_dir <path to datasets>/megadepth --image_size 720
```
**Stage 1**

Training is initialized with the provided pretrained weights of stage 1 on ScanNet. 
```
python3 -u -m torch.distributed.launch --nproc_per_node=1 --rdzv_endpoint=127.0.0.1:29110 train.py --tuple_size 2 --dataset megadepth_720 --batch_size 16 --n_workers 6 --data_dir  <path to datasets> --checkpoint_dir <path to write checkpoints> --init_exp_name pretrained_on_scannet_two_view_stage_1
```
Training stage 1 is trained until the validation matching loss is converged. 

**Stage 2**

Training stage 2 trains with pose loss and loads the checkpoint from stage 1, therefore option `pose_loss` is added and `init_exp_name` needs to be adjusted as follows: 
```
--init_exp_name <experiment name from stage 1> --pose_loss
```

## Training on multi-view
### ScanNet
**Stage 1**

```
python3 -u -m torch.distributed.launch --nproc_per_node=3 --rdzv_endpoint=127.0.0.1:29111 train.py --tuple_size 5 --dataset scannet --batch_size 8 --n_workers 5 --data_dir  <path to datasets> --checkpoint_dir <path to write checkpoints>
```
Training stage 1 is trained until the validation matching loss is converged. 

**Stage 2**

Training stage 2 trains with pose loss and loads the checkpoint from stage 1, therefore the following options are added to stage 1 command: 
```
--init_exp_name <experiment name from stage 1> --pose_loss
```
### MegaDepth
To simplify internal processing, we convert the MegaDepth data to the same data format as ScanNet. Note that for image pairs `image_size=720` is used (following SuperGlue), whereas for multi-view `image_size=640` is used for computational reasons (following LoFTR). It will be written to `<path to datasets>/megadepth_640`:
```
python3 convert_megadepth_to_scannet_format.py --dataset_dir <path to datasets>/megadepth --image_size 640
```
**Stage 1**

Training is initialized with the provided pretrained weights of stage 1 on ScanNet. 
```
python3 -u -m torch.distributed.launch --nproc_per_node=2 --rdzv_endpoint=127.0.0.1:29112 train.py --tuple_size 5 --dataset megadepth_640 --batch_size 2 --n_workers 4 --data_dir  <path to datasets> --checkpoint_dir <path to write checkpoints> --init_exp_name pretrained_on_scannet_multi_view_stage_1
```
Training stage 1 is trained until the validation matching loss is converged. 

**Stage 2**

Training stage 2 trains with pose loss and loads the checkpoint from stage 1, therefore option `pose_loss` is added and `init_exp_name` needs to be adjusted as follows: 
```
--init_exp_name <experiment name from stage 1> --pose_loss
```
## Citation
If you find this repository useful, please cite:
```
@inproceedings{roessle2023e2emultiviewmatching,
      title={End2End Multi-View Feature Matching with Differentiable Pose Optimization}, 
      author={Barbara Roessle and Matthias Nie{\ss}ner},
      booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
      month={October},
      year={2023}
}
```
## Acknowledgements
We thank [SuperGluePretrainedNetwork](https://github.com/magicleap/SuperGluePretrainedNetwork), [kornia](https://github.com/kornia/kornia), [ceres-solver](https://github.com/ceres-solver/ceres-solver) and [NeuralRecon](https://github.com/zju3dv/NeuralRecon), from which this repository borrows code.
