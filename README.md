# extract_number_graph_article

The goal of this article is to automatically extract measured data from graphs. The application is for cases that you only have the graphs in some kind of graphical format, but not the individual measurements. The idea is to train a neural network which can extract the measurements from the images. 

The following important files are present:

- `.devcontainer.json` and `docker/` these represent the development environment used to run the model, primarliy meant for running inside VS Code. 
- `lab_book.ipynb` shows my notes as the project unfolded. This notebook has the style of a labbok such as it is used in chemistry labs. 