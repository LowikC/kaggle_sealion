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

