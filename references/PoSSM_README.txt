The contents of this document describe synthetic aperture sonar datasets that are fully model-derived. 
The data are suitable for bistatic, coherent signal processing and will form acoustic seafloor imagery. 
A description of the data and the model is found in "Description_of_Simulated_SAS_Data_Package.pdf" and 
a description of the format in which the raw binary data is stored is found in "PSU_GDF_Format_20240612.pdf". 
The format description also includes MATLAB code that will effectively parse the data to aid in 
signal processing and image reconstruction. It is left to the researcher to develop a beamforming 
algorithm suitable for coherent signal and image processing.

The data is provided in the Generic Data Format (GDF). Also included are comma separated value 
spreadsheets describing the locations of scatterers and objects of interest within each scene. 
In total, there are 11 60 m x 90 m scenes, labeled scene00 through scene10, with scene00 provided 
with the scatterers in isolation, i.e. no seafloor texture. This is provided for beamformer testing 
purposes and should result in an image similar to the one labeled "PoSSM-scene00-scene00-starboard-0.tif" 
in the Related Data Sets tab. The 10 other scenes have varying degrees of model variation as 
described in "Description_of_Simulated_SAS_Data_Package.pdf". In addition to the binary GDF data, 
a beamformed GeoTIFF image and a single-look complex (SLC, science file) data of each scene is provided. 
The SLC data (science) is stored in the  Hierarchical Data Format 5 (https://www.hdfgroup.org/), 
and appended with ".hdf5" to indicate the HDF5 format. The data are stored as 32-bit real and 
32-bit complex values. A viewer is available that provides basic graphing, image display, and 
directory navigation functions (https://www.hdfgroup.org/downloads/hdfview/). 
The HDF file contains all the information necessary to reconstruct a synthetic aperture sonar image. 
All major and contemporary programming languages have library support for encoding/decoding the HDF5 format.

Supporting documentation that outlines positions of the seafloor scatterers is included in 
"Scatterer_Locations_Scene00.csv", while the locations of the objects of interest for scene01-scene10 
are included in "Object_Locations_All_Scenes.csv". Portable Network Graphic (PNG) images that 
plot the location of objects of all the objects of interest in each scene in Along-Track and 
Cross-Track notation are provided.

Scene00 - 338 Point Scatterers at known locations
Scene01 - Lobster pots, rocks and cylinders at known locations on a uniform seafloor model
Scene02 - Lobster pots, rocks and cylinders at known locations on a rough uniform seafloor model
Scene03 - Lobster pots, rocks and cylinders at known locations on a rough uniform seafloor model
Scene04 - Lobster pots, rocks and cylinders at known locations on a rippled sand seafloor model
Scene05 - Lobster pots, rocks and cylinders at known locations on a mixed rippled sand and smooth uniform seafloor model
Scene06 - Lobster pots, rocks and cylinders at known locations on a uniform seafloor model
Scene07 - Lobster pots, rocks and cylinders at known locations on a rough uniform seafloor model
Scene08 - Lobster pots, rocks and cylinders at known locations on a rough uniform seafloor model
Scene09 - Lobster pots, rocks and cylinders at known locations on a rippled sand seafloor model
Scene10 - Lobster pots, rocks and cylinders at known locations on a mixed rippled sand and smooth uniform seafloor model