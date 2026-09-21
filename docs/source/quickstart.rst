Quickstart
==========

Creating a clustered heatmap visualisation is divided into two separate classes: 
The ``ClusteredHeatMap`` class handles hierachical clustering and requires
corresponding arguments. It returns an object that describes the calculated heatmap.
This object can then be visualised using the ``PlotlyVisuBuilder``.

The simplest way to create a clustered heatmap visualisation is shown below.
Keep in mind that your data needs to be in pandas wide / pivot table format
for clusteredheatmap to properly interpret it.
.. code-block:: python3
  from clusteredheatmap.chm import ClusteredHeatMap
  from clusteredheatmap.visu.ploty.builder import PlotlyVisuBuilder

  # We assume `df` is your DataFrame in wide format
  c = ClusteredHeatMap(df)
  fig = PlotlyVisuBuilder(c).autobuild()
  fig.show()

``autobuild`` is a convenient way to test if your data is
plausible, but it lacks flexibility. Usually, you would
call the individual builder methods like this to get your
clustered heatmap:

.. code-block:: python3
  b = PlotlyVisuBuilder(c)
  b.add_heatmap()
  b.add_col_dendrogram()
  b.add_row_dendrogram()
  fig = b.get_figure()
  fig.show()

See :ref:`chm` and :ref:`pvb` for full documentation on these classes.
The sections below describe common parameters to change as well 
as additional builder methods you might need.

Setting distance and linkage metrics
************************************
TODO don't forget different metrics for each axis

Adding group markers
********************
TODO also colour overrides here

Dealing with symmetric data
***************************
TODO

Axis annotations and ticks
**************************
TODO col/row titles here also

Rearranging the layout
**********************
TODO also resize subplots here

Customising the heatmap
***********************
TODO

Dealing with missing (NaN) values
*********************************

clusteredheatmap was designed from the ground up to support missing values.
However, you need to use one of the distance functions
natively supporting missing values to perform hierachical clustering.
See :ref:`dists` for documentation on these functions. You need to provide
the name of the distance function to use like this:

.. code-block:: python3
   c = ClusteredHeatMap(df, distance="dixon_pds_sqeuclidean")

An alternative is to use complete-case analysis on any other distance function,
effectively making any function support NaNs. This is generally not recommended
though.

.. code-block:: python3
   c = ClusteredHeatMap(
      df,
      distance="euclidean",
      use_completecase_analysis=True,
   )

See also :ref:`chm` for more in-depth documentation.

For the visualisation, the ``add_heatmap`` builder method
supports two additional arguments if you have NaNs.
Note that you don't have to set these, as the defaults
should be fine for most cases.

.. code-block:: python3
  # b is your builder
  b.add_heatmap(
    nan_color = "#000000",
    nan_label = "Hole in data",
  )
