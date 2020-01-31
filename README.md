# Unet_DWT
Unet based on Wavelet coefficients for segmentation

Dataset format:

```bash
Folder-----
      img----
          img1.png
          img2.png
          .......
      label---
          label1.png
          label2.png
          .......
```

Running Command for train.py :
```bash
python3 train.py --train_dataset 'your directory' --val_dataset 'your directory' --model_name 'brainus_db2_lvl2' --checkpoint_path 'chk/brainus__lvl2'
```
### Architecture details

<p align="center">
  <img src="docs/img/arch2.jpg" width="600">
</p>

### Done as a part of the course project for 
Wavelets and Filter Banks, JHU Fall 2019.
