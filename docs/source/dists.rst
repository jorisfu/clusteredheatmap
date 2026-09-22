.. _dists:
Distance Functions for NaNs
===========================

clusteredheatmap includes some distance functions
that can handle missing values / NaN values natively.
To use these methods below, provide their function
name as the ``distance`` argument for ``ClusteredHeatMap``.
Additional keyword arguments these support go into the
``distance_args`` argument. See :ref:`setdistandlink` for
more info.

The nandist library
***********************
`nandist <https://pypi.org/project/nandist/>`__ is integrated by default.
Supported nandist functions are exposed under the following names:

- ``nandist_chebyshev``
- ``nandist_cityblock``
- ``nandist_cosine``
- ``nandist_euclidean``
- ``nandist_minkowski``

Other methods
*************

.. automodule:: clusteredheatmap.algos.distance
   :members:
   :exclude-members: DistanceError, get_preferred_pdist_implementation
