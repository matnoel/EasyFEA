.. _begin:

Beginner's Guide
================

Like every python script, you first start by importing modules contained within ht epython package.

.. code-block:: python

    from EasyFEA import Display, Mesher, ElemType, Materials, Simulations
    from EasyFEA.Geoms import Point, Domain

Most EasyFEA simulations requires few modules, in this scirpt you simply require:

+ 

The simplest and quickest introduction is shown below:

.. literalinclude:: ../examples/HelloWorld.py
   :language: python
   :lines: 5-
