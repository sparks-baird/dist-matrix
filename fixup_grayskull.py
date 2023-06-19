"""Touch up the conda recipe from grayskull using conda-souschef."""
from souschef.recipe import Recipe
from os.path import join

fpath = join("dist-matrix", "meta.yaml")
fpath2 = join("scratch", "meta.yaml")
my_recipe = Recipe(load_file=fpath)
my_recipe["requirements"]["host"].append("flit")
my_recipe.save(fpath)
my_recipe.save(fpath2)
