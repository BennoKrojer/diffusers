# Discriminative Image Generation (DiscIG)


## Modeling
`python3 evaluate_wse.py` is the main file.

You could either create a dataset like the others in `datasets_loading.py` or just use some of the structure in `evaluate_wse.py`.

Ultimately, the most important line you can call is in the method score_batch (this method is kind of specialized to how a batch looks like in my Dataset classes):
dists = model(prompt=text, image=resized_img, sampling_steps=20, unconditional=True)

Where the inputs should be:
text: a list of strings of length BATCHSIZE
resized_img: an image tensor of shape (BATCHSIZE,3,512,512) (see `dataset_loading.py` for how to get this)

The output is of shape (BATCHSIZE, sampling_steps) and you just wanna average them usually such that they only have shape (BATCHSIZE), like in the lines 40 to 50. This will give a score for your batch of image-text-pairs. Smaller is better (which is why they are called distances).

For sampling_steps you can decide: more is better (less variance) but makes it slower. I would say 20 could be enough.

## Dataset
Copy one of the many Dataset classes in `dataset_loading.py`.
Add them with a name in the method `dataset_loading.py:get_dataset()`.
Then get one like this: get_dataset('yourName', 'datasets/'yourName').
That's it. A lot of the arguments are legacy.
