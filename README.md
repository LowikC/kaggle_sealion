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

