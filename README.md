<h2>TensorFlow-FlexUNet-Image-Segmentation-Cardiac-Diagnosis-MRI (2025/11/01)</h2>

This is the first experiment of Image Segmentation for<b>Automated Cardiac Diagnosis Challenge (ACDC) 3 classes</b>,
 based on our 
TensorFlowFlexUNet (TensorFlow Flexible UNet Image Segmentation Model for Multiclass) 
and a 512x512 pixels 
<a href="https://drive.google.com/file/d/1VdxnmE_BZknhJ30M0PHn3iYZlvRdvsFd/view?usp=sharing">
ACDC-ImageMask-Dataset.zip</a>.
which was derived by us from <br><br>
<b>ACDC_training_slices</b> subset in <a href="https://www.kaggle.com/datasets/anhoangvo/acdc-dataset/data">
<b>
ACDC Dataset
</b>
</a>

<br>
<br>
<hr>
<b>Acutual Image Segmentation for 512x512 pixels Cardiac images</b><br>
As shown below, the inferred masks predicted by our segmentation model trained on the 
PNG dataset appear similar to the ground truth masks,but this model failed to segment the region in the third case.
<br>
<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/ACDC/mini_test/images/10007.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/ACDC/mini_test/masks/10007.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/ACDC/mini_test_output/10007.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/ACDC/mini_test/images/10080.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/ACDC/mini_test/masks/10080.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/ACDC/mini_test_output/10080.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/ACDC/mini_test/images/10203.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/ACDC/mini_test/masks/10203.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/ACDC/mini_test_output/10203.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<h3>1. Dataset Citation</h3>
The dataset used was obtained from the kaggle web-site:<br><br>
<a href="https://www.kaggle.com/datasets/anhoangvo/acdc-dataset/data">
<b>
ACDC Dataset
</b>
</a>
<br><br>
<b>About Dataset</b><br>
This preprocessed dataset, derived from the <b>Automated Cardiac Diagnosis Challenge (ACDC)</b>, is meticulously curated for 
cardiac image segmentation endeavors, with a primary focus on delineating the left ventricle (LV), 
right ventricle (RV), and myocardium (MYO). Featuring a diverse range of cardiac magnetic resonance (MR) 
images along with corresponding segmentation labels, 
this dataset enables researchers to explore the nuances of cardiac anatomy and pathology.
<br>

Additionally, this dataset is complemented by detailed scribble annotations, providing a valuable resource for 
scribble-supervised learning—a method crucial for weakly supervised learning approaches. <br>
The annotations are provided as described in the paper:
<a href="https://arxiv.org/pdf/2007.01152">
<b> Learning to Segment from Scribbles using Multi-scale Adversarial Attention Gates.</b><br>
</a>

For comprehensive access to the original challenge and in-depth information, please refer to the 
<a href="https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html"><b>Automated Cardiac Diagnosis Challenge (ACDC)</b></a>.
<br>
The preprocessing of this dataset is carried out utilizing code available on the associated GitHub repository.<br>
<br>Researchers and medical practitioners alike can harness this preprocessed dataset to propel segmentation algorithms forward, 
contribute to the field of medical image analysis, and ultimately enhance patient care in cardiovascular medicine.
ties for medical students interested in breast cancer detection and diagnosis.
  <br><br>
<b>Licence:</b><br>
<a href="https://opensource.org/license/mit">MIT</a>
<br>

<h3>
<a id="2">
2 ACDC ImageMask Dataset
</a>
</h3>
<h4>2.1 Download ImageMask Dataset</h4>
 If you would like to train this ACDC Segmentation model by yourself,
 please download the dataset from the google drive 
<a href="https://drive.google.com/file/d/1VdxnmE_BZknhJ30M0PHn3iYZlvRdvsFd/view?usp=sharing">
ACDC-ImageMask-Dataset.zip</a>.
<br>
, expand the downloaded ImageMaskDataset and put it under <b>./dataset</b> folder to be
<pre>
./dataset
└─ACDC
    ├─test
    │   ├─images
    │   └─masks
    ├─train
    │   ├─images
    │   └─masks
    └─valid
        ├─images
        └─masks
</pre>
<br>
<b>ACDC Statistics</b><br>
<img src ="./projects/TensorFlowFlexUNet/ACDC/ACDC_Statistics.png" width="512" height="auto"><br>
<br>
As shown above, the number of images of train and valid datasets is not large to use for a training set of our segmentation model.
<br>
<br>
<h4>2.2 ImageMask Dataset Derivation</h4>
The folder struture of ACDC dataset of the kaggle is the following 
<pre>
./acdc_preprocessed
├─ACDC_testing_volumes
├─ACDC_training_slices
└─ACDC_training_volumes
</pre>
 We generated a 512x512 pixels PNG ImageMask dataset from h5 files in <b>ACDC_training_slices</b> subset in <b>/acdc_preprocessed</b>.<br>
All images and masks (labels) in the h5 files in the subset were resized to be 512x512 pixels, and  
the masks were colorized by using the following Grayscale and BGR triplet mapping table.<br><br>
<table border="1" style="border-collapse: collapse;">
<tr><th>Grayscale</th><th>Color name </th><th>BGR triplet</th>
</tr>
<tr><td>1</td> <td>blue</td><td>(255,0,0)</td></tr>
<tr><td>2</td> <td>green</td><td>(0,255,0)</td></tr>
<tr><td>3</td> <td>red</td><td>(0,0,255)</td></tr>
</tr>
</table>
<br>
<h4>2.3 Train Images and Masks Sample</h4>
<b>Train_images_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/ACDC/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_masks_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/ACDC/asset/train_masks_sample.png" width="1024" height="auto">
<br>
<h3>
3 Train TensorFlowFlexUNet Model
</h3>
 We trained ACDC TensorFlowFlexUNet Model by using the following
<a href="./projects/TensorFlowFlexUNet/ACDC/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorFlowFlexUNet/ACDC and run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorFlowFlexUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>

<b>Model parameters</b><br>
Defined a small <b>base_filters = 16 </b> and large <b>base_kernels = (9,9)</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet.py</a> 
and a large num_layers (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
;You may specify your own UNet class derived from our TensorFlowFlexModel
model         = "TensorFlowFlexUNet"
generator     =  False
image_width    = 512
image_height   = 512
image_channels = 3
num_classes    = 4

base_filters   = 16
base_kernels   = (9,9)
num_layers     = 8
dropout_rate   = 0.04
dilation       = (1,1)
</pre>
<b>Learning rate</b><br>
Defined a very small learning rate.  
<pre>
[model]
learning_rate  = 0.00007
</pre>
<b>Loss and metrics functions</b><br>
Specified "categorical_crossentropy" and <a href="./src/dice_coef_multiclass.py">"dice_coef_multiclass"</a>.<br>
<pre>
[model]
loss           = "categorical_crossentropy"
metrics        = ["dice_coef_multiclass"]
</pre>
<b>Dataset class</b><br>
Specifed <a href="./src/ImageCategorizedMaskDataset.py">ImageCategorizedMaskDataset</a> class.<br>
<pre>
[dataset]
class_name    = "ImageCategorizedMaskDataset"
</pre>
<br>
<b>Learning rate reducer callback</b><br>
Enabled learing_rate_reducer callback, and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.4
reducer_patience   = 4
</pre>
<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 10
</pre>

<b>RGB Color map</b><br>
rgb color map dict for ACDC 1+3 classes.<br>
<pre>
[mask]
mask_file_format = ".png"
; 1+3 classes         blue,          green,     red 
rgb_map = {(0,0,0):0,(0,0,255):1, (0,255,0):2, (255,0,0):3,}
</pre>

<b>Epoch change inference callback</b><br>
Enabled <a href="./src/EpochChangeInfereuncer.py">epoch_change_infer callback</a></b>.<br>
<pre>
[train]
epoch_change_infer       = True
epoch_change_infer_dir   =  "./epoch_change_infer"
num_infer_images         = 6
</pre>

By using this callback, on every epoch_change, the inference procedure can be called
 for 6 images in <b>mini_test</b> folder. This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> <br> 

<b>Epoch_change_inference output at starting (epoch 1,2,3)</b><br>
<img src="./projects/TensorFlowFlexUNet/ACDC/asset/epoch_change_infer_at_start.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at middlepoint (epoch 16,17,18)</b><br>
<img src="./projects/TensorFlowFlexUNet/ACDC/asset/epoch_change_infer_at_middlepoint.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at ending (epoch 33,34,35)</b><br>
<img src="./projects/TensorFlowFlexUNet/ACDC/asset/epoch_change_infer_at_end.png" width="1024" height="auto"><br>
<br>
In this experiment, the training process was stopped at epoch 35 by EearlyStopping callback.<br><br>
<img src="./projects/TensorFlowFlexUNet/ACDC/asset/train_console_output_at_epoch35.png" width="920" height="auto"><br>
<br>

<a href="./projects/TensorFlowFlexUNet/ACDC/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/ACDC/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorFlowFlexUNet/ACDC/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/ACDC/eval/train_losses.png" width="520" height="auto"><br>

<br>

<h3>
4 Evaluation
</h3>
Please move to <b>./projects/TensorFlowFlexUNet/ACDC</b> folder,<br>
and run the following bat file to evaluate TensorFlowFlexUNet model for ACDC.<br>
<pre>
./2.evaluate.bat
</pre>
This bat file simply runs the following command.
<pre>
python ../../../src/TensorFlowFlexUNetEvaluator.py ./train_eval_infer_aug.config
</pre>

Evaluation console output:<br>
<img src="./projects/TensorFlowFlexUNet/ACDC/asset/evaluate_console_output_at_epoch35.png" width="920" height="auto">
<br><br>

<a href="./projects/TensorFlowFlexUNet/ACDC/evaluation.csv">evaluation.csv</a><br>
The loss (categorical_crossentropy) to this ACDC/test was low and dice_coef_multiclass high as shown below.
<br>
<pre>
categorical_crossentropy,0.0112
dice_coef_multiclass,0.9942

</pre>
<br>

<h3>
5 Inference
</h3>
Please move <b>./projects/TensorFlowFlexUNet/ACDC</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorFlowFlexUNet model for ACDC.<br>
<pre>
./3.infer.bat
</pre>
This simply runs the following command.
<pre>
python ../../../src/TensorFlowFlexUNetInferencer.py ./train_eval_infer_aug.config
</pre>
<hr>
<b>mini_test_images</b><br>
<img src="./projects/TensorFlowFlexUNet/ACDC/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorFlowFlexUNet/ACDC/asset/mini_test_masks.png" width="1024" height="auto"><br>

<hr>
<b>Inferred test masks</b><br>
<img src="./projects/TensorFlowFlexUNet/ACDC/asset/mini_test_output.png" width="1024" height="auto"><br>
<br>
<hr>
<b>Enlarged images and masks of 512x512 pixels Cardiac images</b><br>
<table>
<tr>
<th>Image</th>
<th>Mask (ground_truth)</th>
<th>Inferred-mask</th>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/ACDC/mini_test/images/10005.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/ACDC/mini_test/masks/10005.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/ACDC/mini_test_output/10005.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/ACDC/mini_test/images/10053.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/ACDC/mini_test/masks/10053.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/ACDC/mini_test_output/10053.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/ACDC/mini_test/images/10203.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/ACDC/mini_test/masks/10203.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/ACDC/mini_test_output/10203.png" width="320" height="auto"></td>
</tr>



<tr>
<td><img src="./projects/TensorFlowFlexUNet/ACDC/mini_test/images/10564.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/ACDC/mini_test/masks/10564.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/ACDC/mini_test_output/10564.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/ACDC/mini_test/images/10695.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/ACDC/mini_test/masks/10695.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/ACDC/mini_test_output/10695.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/ACDC/mini_test/images/10719.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/ACDC/mini_test/masks/10719.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/ACDC/mini_test_output/10719.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>
<h3>
References
</h3>
<b>1. Automated Cardiac Diagnosis Challenge (ACDC)</b><br>
<a href="https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html">https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html
</a>.
<br>
<br>

<b>2. Learning to Segment from Scribbles using Multi-scale Adversarial Attention Gates</b><br>
Gabriele Valvano, Andrea Leo, Sotirios A. Tsaftaris <br>
<a href="https://arxiv.org/pdf/2007.01152">https://arxiv.org/pdf/2007.01152</a>
<br><br>

<b>3. An Exploration of 2D and 3D Deep Learning Techniques for Cardiac MR Image Segmentation</b>
 Christian F. Baumgartner, Lisa M. Koch, Marc Pollefeys, Ender Konukoglu<br.
<a href="https://arxiv.org/pdf/1709.04496">https://arxiv.org/pdf/1709.04496</a>
<br><br>

<b>4. acdc_segmenter</b><br>
Christian F. Baumgartner<br>
<a href="https://github.com/baumgach/acdc_segmenter">https://github.com/baumgach/acdc_segmenter</a>
<br>
<br>
<b>5. TensorFlow-FlexUNet-Image-Segmentation-Model</b><br>
Toshiyuki Arai antillia.com <br>
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model">
https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model
</a>
<br>
<br>

