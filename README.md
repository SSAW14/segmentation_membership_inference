# Segmentations-Leak: Membership Inference Attacks and Defenses in Semantic Image Segmentation
by Yang He, Shadi Rahimian, Bernt Schiele and Mario Fritz, ECCV2020 (https://arxiv.org/abs/1912.09685).

**Note**: The current software is tested with PyTorch 0.4.1 and Python 3.6.

## Example
### Download
We provide example data and model weights at https://drive.google.com/drive/folders/1A4WBp5qxS8rn_EnbCY7H5VArtsErS8pv.

Download the required files and run
```bash
unzip examples.zip && weights.zip
```
### Attack
We attack an UperNet trained on Cityscapes training set with 2975 images. Our trained model with SGD or differential private SGD (DPSGD) can be found in 
https://drive.google.com/drive/folders/1A4WBp5qxS8rn_EnbCY7H5VArtsErS8pv.

- Attack a model with our structured loss map:
```bash
python attack.py -resume ./weights/loss.pth.tar -gpu [GPU_ID]
```
- Attack a model with concatenation of GT and predictions:
```bash
python attack.py -resume ./weights/concate.pth.tar -input concate -gpu [GPU_ID]
```
- Attack a model with different number of patches:
```bash
python attack.py -resume ./weights/loss.pth.tar -gpu [GPU_ID] -num-patch [NUM]
```

### Defense
We provide the defenses of Argmax, Gauss and DPSGD in this demo. For DPSGD, we provide the model trained with DPSGD as well as returned posteriors for the faster demo. In the demos below, concatenation is used to show the results, but feel free to change to structured loss map.

- Defense with Argmax:
```bash
python attack.py -resume ./weights/concate.pth.tar -input concate -gpu [GPU_ID] -argmax
```

- Defense with Gauss:
```bash
python attack.py -resume ./weights/concate.pth.tar -input concate -gpu [GPU_ID] -gauss [STD_NOISE]
```

- Defense with DPSGD:
```bash
python attack.py -resume ./weights/concate.pth.tar -input concate -gpu [GPU_ID] -dpsgd
```

## Reproducibility
To reproduce our results, we provide data splits for dependent and independet settings in splits folder. For Mapillary Vistas dataset, we convert the GT labels to Cityscapes label space.

## Citation
If our work is useful for your research, please consider citing:

    @inproceedings{he2020segmentations_leak,
      title={Segmentations-Leak: Membership Inference Attacks and Defenses in Semantic Image Segmentation},
      author={He, Yang and Rahimian, Shadi and Schiele, Bernt and Fritz, Mario},
      booktitle={ECCV},
      year={2020}
    }

## Questions

Please contact 'yang.he@cispa.saarland' or 'yang@mpi-inf.mpg.de'
