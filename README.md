# ClusteredHeatMap
Python library for hierarchical clustering algorithms and clustered heatmap / clustergram visualisation.
Written as part of a bachelor's thesis about distance estimation with missing values and its effects
on hierarchical clustering of biomedical data.

## Usage
Install the library by adding this line to your `requirements.txt`:
```
clusteredheatmap @ git+https://github.com/jorisfu/clusteredheatmap.git
```

You first need to create a `ClusteredHeatMap` object with the data you want to cluster:
```python3
from clusteredheatmap.chm import ClusteredHeatMap

c = ClusteredHeatMap(
    df_wide, # This needs to be a pandas dataframe in wide format
    distance="nan_euclidean", # Various distances are supported. [TODO list]
    linkage="complete", # Various linkage methods are supported as well. [TODO list]
    
    # This optionally maps certain columns based on their labels to groups.
    # These groups can then be visualized using group markers in the plot
    column_group_mappings={
        "Group": {"P10": "CTL", "P11": "CTL", "P3": "AD"},
        "Age": {"P10": "<65", "P11": ">65", "P3": "<65", "P4": ">65"},
    },

    # Same thing for rows. Multiple mappings are always supported.
    row_group_mappings={
        "Protgroup": {
            "A3KMH1-3": "Cool Proteins",
            "A6NHQ2": "Cool Proteins",
            "Q9Y6R0": "Uncool Proteins",
            "P09382": "Uncool Proteins",
        },
        "Coolness group": {
            "A3KMH1-3": "Very cool",
            "A6NHQ2": "Semi-cool",
            "Q9Y6R0": "Terrible",
            "P09382": "Terrible",
        },
    },

    # What the data on each axis means
    data_column_title="Sample",
    data_row_title="Protein",
)
```

With this object `c`, you can visualize the clustered heatmap like this:

```python3
from clusteredheatmap.visu.plotly.builder import PlotlyVisuBuilder

fig = PlotlyVisuBuilder(c, vertical_layout="dgh", horizontal_layout="dgh").autobuild()
fig.show()
```

90% of the time you're just gonna use `.autobuild()` to obtain a visualization.
You can however also use the visualization builder manually to get
more control over visualization parameters. Example:

```python3
from clusteredheatmap.visu.plotly.builder import PlotlyVisuBuilder

b = PlotlyVisuBuilder(c, vertical_layout="dgh", horizontal_layout="dgh")
b.add_heatmap(zmin=-2.5, zmid=0, zmax=3.5)
b.add_col_dendrogram()
b.add_row_dendrogram()
b.add_col_group_markers(_color_overrides={"Age": {"<65": "#FCE300", ">65": "#ABD310"}})
b.add_row_group_markers(
    _color_overrides={
        "Protgroup": {"Cool Proteins": "#30ff65", "Uncool Proteins": "#ff1234"}
    }
)
b.get_figure().show()
```

All parameters are documented in the docstrings for the builder functions.
