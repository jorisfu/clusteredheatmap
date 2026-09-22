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
  from clusteredheatmap.visu.plotly.builder import PlotlyVisuBuilder

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

.. _setdistandlink:
Setting distance and linkage metrics
************************************

The ``linkage`` and ``distance`` parameters decide which linkage
and distance metric to use. You can use any linkage and distance
supported by scipy, see `scipy.cluster.hierarchy.linkage <https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html>`__
and `scipy.spatial.distance.pdist <https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html>`__.
There are also some additional distance methods included for data with missing values, see :ref:`dealingwithnans`.

.. code-block:: python3

  c = ClusteredHeatMap(
    df,
    distance="cityblock",
    linkage="ward",
  )

You can also set different distance and linkage metrics for each axis (i.e. rows/columms)
by passing a tuple. The first element specifies the row metric, the second element specifies the column metric.

.. code-block:: python3

  c = ClusteredHeatMap(
    df,
    distance=("cityblock", "euclidean"),
    linkage=("ward", "single"),
  )

Some distance functions support additional keyword arguments. If you need to pass these,
use the ``distance_args`` parameter like this:

.. code-block:: python3

  c = ClusteredHeatMap(
    df,
    distance="mesquita_eed",
    distance_args={"min_k": 3, "max_k": 6} # Put keyword args into dictionary
  )

``distance_args`` can also be a tuple if different distances are used for rows/columns.

You can also pass custom functions, see :ref:`chm` for more info.

Adding group markers
********************

You can add group mappings to your clustered heatmap that you can then visualise in 
group markers. Suppose you have four columns "Alice", "Bob", "Charlie", "David" in your ``df``.
You can assign them to different groups like this:

.. code-block:: python3

   c = ClusteredHeatMap(
     df,
     column_group_mappings={
        "Attitude": {"Alice": "Cool", "Bob": "Uncool", "Charlie": "Cool", "David": "Uncool"},
        "Coolness": {"Alice": "Very cool", "Charlie": "Extremely cool"}
     }
   )

Note how you can use multiple group mappings, and no group mapping must map every column to a group.
To visualise these mappings using group markers, you need to call ``add_col_group_markers`` on
the builder. By default, colours are automatically assigned to groups.
You can manually override assigned colours by setting colours for all mapped groups
of one mapping like this:

.. code-block:: python3

  # b is your PlotlyVisuBuilder for c
  b.add_col_group_markers(
    _color_overrides={
      "Coolness": {"Very cool": "#FCE300", "Extremely cool": "#7070FF"}
    }
  )

Note that you have to specify colours for every group of a mapping if you want to 
override colours.

For rows, the equivalent parameter is ``row_group_mappings``
and the corresponding builder method is ``add_row_group_markers``.

Customising the heatmap
***********************
The ``add_heatmap`` builder method has some customisation parameters.

Setting min/mid/max values for the colour scale
-----------------------------------------------
By default, the minimum and maximum values in your ``df`` correspond to the
lowest and highest colour of the heatmap's colour scale and the middle
of the colour scale is the centre between the two.
You can change this by setting the ``_zmin, _zmid, _zmax`` parameters:

.. code-block:: python3

  b.add_heatmap(
    _zmin=-1.5,
    _zmid=0.0,
    _zmax=2.0,
  )

The only restraint to these parameters is ``_zmin`` < ``_zmid`` < ``_zmax``.
For ``_zmid``, you can also pass ``"median"`` or ``"mean"`` to set
the colour scale midpoint to the median or mean of your data.


Setting a custom colour scale
-----------------------------
You can set the colour scale to use in the ``colorscale`` parameter.
All colour scales integrated into plotly are supported,
see `their documentation <https://plotly.com/python/builtin-colorscales/>`__.
You can also pass your own custom colour scale like this:

.. code-block:: python3

   b.add_heatmap(
     colorscale=[[0.0, "#F45182"], [0.5, "#000000"], [1.0, "#318439"]]
   )

Note that custom colour scales must have their minimum (0.0), mid (0.5), and maximum (1.0)
values defined to work properly. Between these, any arbitrary further granularity is
allowed.

Customising colour scale legend ticks
-------------------------------------
The colour bar legend corresponding to the colour scale has ticks
for the minimum, mid, and maximum colour scale values that display
the matching values of your data.
You can prefix them using the ``ticktext_prefix`` parameter, e.g.:

.. code-block:: python3

   b.add_heatmap(
     ticktext_prefix=("5% quantile: ", "Median: ", "95% quantile: ")
   )

You can also change how the floats are formatted by passing a python
formatting string to ``tickfloatformat``:

.. code-block:: python3

   b.add_heatmap(
     tickfloatformat="{:6.4f}",
   )


Axis annotations and ticks
**************************
You can set the title for your rows, columns and z-values
in the ``ClusteredHeatMap`` constructor.
For example, if your columns are samples, your rows are proteins,
and your values are intensities, you can do the following:

.. code-block:: python3

  c = ClusteredHeatMap(
    df,
    data_row_title="Protein",
    data_column_title="Sample",
    data_z_title="Intensity",
  )

These labels propagate to the relevant parts of the visualisation.

You can also show ticks for your rows/columns in the visualisation
by calling ``add_row_ticks`` or ``add_col_ticks`` on the
``PlotlyVisuBuilder``. Note that you need to specify
where to place them by setting the ``anchor_subplot`` and
``side``:

.. code-block:: python3

   b.add_row_ticks(
      anchor_subplot="h", 
      side="right", # alternative: "left"
   )

   b.add_col_ticks(
      anchor_subplot="h", 
      side="bottom", # alternative: "top"
   )

The allowed values of ``anchor_subplot`` equal the 
subplot identifiers as explained in :ref:`layout`.

.. _layout:
Rearranging the layout
**********************
You can change the order of dendrogram, heatmap and group markers
by using layout strings when initialising the ``PlotlyVisuBuilder``:

.. code-block:: python3

   b = PlotlyVisuBuilder(
     c,
     vertical_layout="dhg", # Top-to-bottom
     horizontal_layout="hd", # Left-to-right
   )

in the ``vertical_layout``, "g" is the column group markers and "d" is 
the column dendrogram.
In the ``horizontal_layout``, "g" is the row group markers and "d" is
the row dendrogram.
For both strings, "h" is the heatmap and must be given.
Any other subplots can be left out if they are not needed.
Note that you cannot call builder methods for subplots
that are not specified in the layout strings.

You can also change the size of subplots when calling their builder methods using
the ``relative_height`` and/or ``relative_width`` parameters.
These parameters take arbitrary integers.

To maintain aligned axes with the heatmap, you can only change
the width of subplots on the horizontal axis (i.e. row dendrogram and group markers)
and for subplots on the vertical axis (i.e. column dendrogram and group markers),
you can only change the relative height.
For group markers, the relative sizes are per group marker.
The code below shows all possible options.

.. code-block:: python3

   b.add_heatmap(
     relative_width=60, # default: 80
     relative_height=50, # default: 80
   )

   b.add_col_dendrogram(
     relative_height=20 # default: 30
   )

   b.add_row_dendrogram(
     relative_width=20 # default: 30
   )

   b.add_col_group_markers(
     relative_height_per_marker=3, # default: 2
   )

   b.add_row_group_markers(
     relative_width_per_marker=2, # default: 1
   )

Dealing with symmetric data
***************************
If your input ``df`` is a symmetric matrix, you can specify this
by setting the ``is_symmetric`` parameter of the ``ClusteredHeatMap``
to ``True``. In this case, only the rows are clustered and the dendrogram
and distance matrix are copied to the columns, saving time on execution.
clusteredheatmap will not check if your matrix is actually symmetric,
so use this only if you are sure about the nature of your data.

.. _dealingwithnans:
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
    nan_color="#000000",
    nan_label="Hole in data",
  )


