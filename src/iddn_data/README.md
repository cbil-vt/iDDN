# Data and images for iDDN

We include two test examples for iDDN tutorial and test cases.
The images are used in the tutorial.

## Usage
If you want to load a data, use the `load_data` module.

```python
from iddn_data import load_data
data = load_data.load_example(file_name="example.npz")
```

We can also get the full path of images in this folder.
Then we can display it in the Jupyter notebook. 

```python
from iddn_data import load_data
img_path = load_data.get_image_path("three_layer.png")
```

## Datasets
`example.npz` is from a three layer simulation 50+50+50 nodes. 
Edges among mRNAs (first 50 nodes), No edges among TFs and miRNAs.

`two_part_network.npz` is the same test example of DDN3.0 package.
It is a bipartite graph.
