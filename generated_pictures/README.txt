(1) We only considered the regenerated images by best/average/worst model, model name is indicated as the file name
(2) The naming scheme of each image is as follows:
- The first letter (H/A/M) represents highest/average/lowest
- The second letter (5) represents 5 images
- The rest of letters represents corresponding loss (KL divergence/Reconstruction Loss/Total Loss)

For example:
H5RL-Set -> 5 images with highest reconstruction loss for chosen beta

(3) For each png file, there are 4 rows. The meaning of each row is as follows:
1st row: original 5 images by non-quantized model (float-32)
2nd row: generated 5 images by non-quantized model (float-32)
3rd row: original 5 images by quantized model (int-8)
4th row: generated 5 images by quantized model (int-8)