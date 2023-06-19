"""Touch up the conda recipe from grayskull using conda-souschef."""
from souschef.recipe import Recipe
from os.path import join

fpath = join("dist-matrix", "meta.yaml")
fpath2 = join("scratch", "meta.yaml")
my_recipe = Recipe(load_file=fpath)
# remove
my_recipe["requirements"]["host"].replace("flit_core >=3.2,<4", "flit")
my_recipe.save(fpath)
my_recipe.save(fpath2)
