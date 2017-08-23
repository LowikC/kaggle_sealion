# Sea lion competition

## Ideas

- predict density map
-- map is created with gaussian around each dot
-- gaussian width can be linked to the type of sealion
-- how to detect several type? predict 5 density map?

- predict count
-- can be a regression or a classification (0, 1, ..., Nmax)
-- seems harder to generalize than with the density map

- detect each object
-- extract fixed size patch for training
-- OR use superpixels and extract superpixels around dots


- use superpixels to have the direction of the sealion and use it for the gaussian on the density map

## TODO

- extract dots from training image
- generate density map
- create generator of (NxN) images from full image, with some overlapping
-- each patch should have a count and a density map
-- patch outside of the mask should be excluded (what percentage outside the mask?)

## Notes
30/04/2017: first try with the full training pipeline.
It seems that many patches have no sealion, so I need to rework on the sampling. Besides, even if a patch has a sealion, say an adult_female, it remains 4 density maps (male, pups, ...) without sealion.
The second issue is that the loss seems really low, even if the coutning error is huge: some values are negative, the total count can be >200, while still having low values almost everywhere on the predicted map (because we sum over a large 224x224 patch). So I should double check if I need a ReLU layer (to avoid negative values) and if the output should be normalized in some way.

01/05/2017:
- Regarding the sampling, the idea would to choose first if we sample background or dots (1 vs 5 proba for instance, or lower), if background, sample a random location, if dots, choose a dot, and its location in the patch randomly.
It can be done by loading only the dots and shape of the image, and once the location is decided, we can load the appropriate block. Thus, the random index of the Iterator would be directly on the training image, and we avoid to load block with no dots in them.
For debugging, I will probably simplify the problem, and try to predict the total count of sealion (not by type). I can still use the same pre-compuuted density maps, and sum over the last axis.
- I found a new paper that doesn't try to predict a density map but directly the count in the receptive field. To read.
- Using a lr=0.001 instead of 1e-5 seems much better. To try with higher lr. But the final count error is still huge. Need to try again several losses.

02/05/2017
- Work on normalization of the output: scaling + log
The loss did decrease, but I still get shit at the end when trying to predict... :(
- I look at the code of crowdnet found on GitHub: the build the density map in different way: use a kdtree to found the closest neighbor of each point, and use the distance to the closest point as the Gaussian sigma when applying the Gaussian mask. The way they do it is really not optimal, it took 10min to build one density map... We could do it much faster, by computing the size of the Gaussian mask, and not applying a gaussian filter on the full image for only one point.
- Idea for segmentation: 
* get superpixels graph
* define a min - maximum area for each type of sealion
* start from the superpixel containing the dot
* find the closest superpixel neighbor (closest = same color)
* continue until the area covered is > area_min and < area_max and closest distance is too big.
03/05/2017: This is depressing
- Kept trying ideas but nothing works...
- Tried to create a segmentation map using superpixels but color are so close to the background that it seems hard to get something ok automatically.
- Build a simple segmentation map using the existing density map
- Use it to train UNet: still very bas result, we just learn to predict something random on all images...
04/05/2017:
- Debug the Unet model, by trying to learn an identity (same input/output). This seems to work (that's both depressing and encouraging ...): loss decreases rather fast, validation loss is at the same level as training loss, and the final results are as expected.
05/05/2017:
- Someone reach rmse=12 on the leaderboard
- Debug the regression of density map:
-- without normalization and input noise, it doesn't work.
-- with normalization (log, scale 0, 1 and 0-mean, 1-std normalization): loss is a bit high, and the prediction is not perfect, but it seems much better.
-- with normalization, but no batchnorm: even better, val loss at 0.03, good looking results (still a bit noisy, but the overall shape is there).
- with normalization, batchnorm only in the "down" path: val loss at 1.2, bad looking (but not completly random).
- Tried again to learn the density map:
-- normalized image, put in 0, 1 and then normalized 0-mean, 1-std
-- log output + normalization
-- no batch norm as suggested by above tests
-- loss decreases slowly, reaching 0.78 (val) after 2 epochs (100 * 8 samples each). BUT, it seems we start to learn something! Will try with more samples.
-- with 10 epochs, 200 steps each (batch_size = 8), the loss is not better (0.78), BUT the prediction shows that we actually learn pretty well to segment the exact shape of sealion. That's very encouraging, even if results are hardly exploitable for now.
- To test: learn on patch with FC network, then apply on test and use a second network to predict the count.
06/05/2017:
-- Need to see how to exploit and improve previous results.
-- The count of the predicted density map is completly wrong, I need to take it into account while learning
-- I implemented a model with 2 losses: mse on the density map + rmse on the predicted count. With the log normalization, it doesn't learn anything.
-- Trying with SGD instead of Adam and ELU instead of ReLU
-- Re-read some papers:
--- http://www.ee.cuhk.edu.hk/~xgwang/papers/zhangLWYcvpr15.pdf: alternate between the 2 losses (and use appropriate weights, 10 for dmap, 1 for count)
--- https://www.robots.ox.ac.uk/~vgg/publications/2015/Xie15/weidi15.pdf: scale the dmap by 100
--- https://papers.nips.cc/paper/5539-depth-map-prediction-from-a-single-image-using-a-multi-scale-deep-network.pdf: depth map, predict log(y), and use a scale invariant loss.
--- https://arxiv.org/pdf/1605.02305.pdf: use classification instead of regression (for depth map): uniformly quantize the output, in log space. It's a good, I will try this tomorrow.
07/05/2017: Election day
- With ELU and SGD, loss goes down to 0.41, and keep decreasing. Good news is that the count seems not to bad, but pups are hard to detect.
07/05/2017
- Created function to compute the prediction on a full image
- Start learning with quantized target in log space, with nbins=128 (more takes a lot of memory)
08/05/2017
- First try with quantization failed: we predict only 0 (the classes are too much imbalanced). Loss decreases to 2.20 but stop decreasing after 2 epochs
- Spend a lot of time to find a way to put weights on each class. But second try is still bad: put low weights on class #0 (background, 0.001), and predict only 1...
- With a weight = 0.01, we predict different values, but still far from the target. I shoudl try to debug it with identity + noise.
09/05/2017
- Manually alternate between loss: the loss on count doesn't seem to learn anything (loss doesn't decrease). I'd like to try to learn it on bigger patches (like 224x3).
- Besides, new epochs with mse didn't improve the previous model.
10/05/2017
- Will try to learn a FCN on sealion patches. Then apply it on bigger image and learn a second model to predict the count.
- Wrote notebook to train Xception on 91x91 patches. Patches are generated on the fly, with a probability distribution computed on the dataset for each type of sealion.
- Size of the patches, the type of CNN, normalization of the image, ... are hyperparameters to be tested.
11/05/2017
- Converted Xception into a fully convolutional model, and train it on patches: I get the same accuracy as with the Dense layers (anf GlobalPooling).
- Will read the "Tiramisu" paper. Code is available here:  https://github.com/titu1994/Fully-Connected-DenseNets-Semantic-Segmentation
- Finetuning regression of density map for adult_males only doesn't work.
- It seems that the superpixels can be used to find the orientation of sealion: I will check on more images.
12/05/2017
- Superpixels are not bad to find the orientations of sealion. I implemented the new density map.
I will double check the dots (using those posted on Kaggle), annotate a few sealions with full mask to get some stats + validation set, and generate all density maps.
13/05/2017
14/05/2017
- Spent time to get the rights dots. Compared with coordinates on Kaggle forum.
- Also checked the missmatched images. Some of them seems perfectly ok, so I will keep them.
15/05/2017
- Computed stats on sizes and area (also for superpixel) and launch ellipses detection for all training images.
- Ideas:
--  could we classify superpixels?
-- could we use superpixel + ellipses mask to get an accurate segmentation.
-- could we build a segmentation map using the ellipses gaussian?
- Implemented the Tiramisu CNN, to be test
- Computed ellipses on corrected dots
- Computed dmap with ellipses
- Write method to convert the density map to a segmentation map. It is worth trying to learn Unet on this.
16/05/2017
- Tried training on CPU with density map and segmentation map from ellipses.
-- Try several class weights for segmentation
-- We learn something, but it is not good enough
17/05/2017
- Training on GPU: mush faster (like x10)
- Add a random flip
- Add callbacks (LRonPlateau, Checkpoint and TensorBoard)
- Try training on more epochs (20 epochs, 400 steps x 8, 100 steps for validation)
- Regression training gives visually ok results, but not on pups. And I still don't know how to exploit them. I tried to add a GlobalAvgPooling + Dense to predict the counts, but it doesn't work (plugged on the smalles feature map).
- Segmentation training gives weird results visually: many classes are mixed up with the background.
18/05/2017:
- Let's rethink about the problem, I tried to progress by small steps:
-- I will implement the pipeline to create a dummy submission and evaluate it on validation data
-- The fist submission can be all 0 (I already know that the public LB gives 29 for this).
-- The second can be just the mean of each classes.
- Then, I will retry to work on a simpler CNN model: finetune inception or resnet, convert to fully convolutionnal (or not), create a heatmap, and use it to predict counts.
- I have all data on AWS, so it should be quite fast to train some models.

-- The absolute value seems wrong on local CV, but the relative value is ok:I got 38 for all-0 (29 on LB), which is higher than 33 for mean prediction (vs 27 on LB).
-- By using a different random split, that guaranty a similar mean, I got something closer to the public LB result (24.7 (vs 27.5) and 30.23 (vs 29)
- For small patches for pups, disk I/O is the bottleneck. I need a better way to load and create random patches...
-- I decided to samples several patches from each block loaded in memory. It speed up a lot the learning.
-- Finetuning the last layer, loss decrease from 1.5 to 0.3
- At last! It seems I got some interesting results on pups (to be validated thoroughly): resNet50, finetune for pups vs all (fullt convolutional), then add 2 conv layer to predict counts on large patch (384 x 384).
-- After a few tests, it seems not bad at all! I'm really happy. Next steps: Same for all sealion type. I think I should automate this a bit, but I will try first with subadult_males to see how it goes (this is the most difficult one I think, easily confused with adult_females and adult_males.
19/05/2017
- I can't get better than 80% accuracy for subadult or juveniles (not tried yet for females and adult_males).
- The loss is very high, compared to pups (3.5 vs 0.3). That's seems weird.
- Interesting point: the final rmse when predicting the mean is bad because of adult_females, juveniles and pups, that high max and high variance. So I could still predict the mean for adult_males and subadult_males, and just improve the 3 others.
- For juveniles, the counts are not that bad, but I have to correct the prediction by substracting on offset, and take the floor of the predicted count.
- For adult_females, counts are also ok
- Next step: build the automated pipelines for all types, and validate. I should build new train/val/test set, to get a good mean.
- Tried to learn to predict counts for all classes except pups: we learn something, but the count prediction is not good, probably because counts for adult and subadult males are too low. I could try to normaluze this to have roughly the same mean between all type of sealions
- Implemented normalization of counts, to be tested tomorrow
20/05/2017
- Tried to learn the to predict counts for all types but pups, with normalized means, but results are bads. I could give it another try, but I will first try to validate results type by type.

21/05/2017
- Run the prediction over all validation set for pups: my rmse decreases significantly, but is not that good (24).
- Ran over all the test set: it took 30hours (!!)
- Submit with mean count for all types except pups, where I used the predicted scores: get a rmse on public LB of 29... that worse that the mean benchmark and very disapointing.
- Big counts have a huge impact on the rmse, so I try to just divide all counts by a constant to get the same mean as the training set: rmse decrease to 26 on public LB. That is something I need to take care of.

30/05/2017
- Back after 1 week break.
- A post on Kaggle mentions that semantic segmentation works fine. Both 2nd and 3rd on public LB use it (predict square around sealion)

02/05/2017
- Tested several new things:
-- learn classification then counts for all types of sealion: the counts loss is still very bad and doesn't move much.
-- tried a simpler network: I get the same accuracy as by finetuning ResNet ! But the prediction of the heatmap on a test image seems much worse.
- Someone one Kaggle tried to do a regression of the counts on the full image resized to 512x512 with a very simple CNN and get 24 on public leaderboard! I could try it to improve my score
- Someone posted a very interesting link to new research from Lempitsky, on counting pinguins. The setting is very similar to the Kaggle challenge, and I will try some ideas from the paper, like learning segmentation map with ignored regions around sealions, and then use this prediction to refine the regression density target.

