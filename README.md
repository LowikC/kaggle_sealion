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


