# Hydrologic Learning Pipeline
## Author(s): Nicholas Majeske
## Contact: nmajeske@iu.edu

### Source Code Description
1. Arguments.py : Custom argument parser that parses arguments from sys.argv into a Container object
2. Container.py : Generalized data container class engineered to facilitate partitioning of data. All classes holding data and employ partitioning will inherit from this class.
3. Data.py : Overarching container class that holds all data
4. Driver.py : Handles (distributed or non-distributed) invocation of Execute.py
5. Execute.py : The primary script of this project which defines data loading, preprocessing, model training/evaluation, plotting, etc.
6. Experiments.py : Script defining arguments for all experiments designed, executed, and analyzed for this project.
7. GatherErrors.py : Script that gathers subbasin errors of all evaluated models in "Evaluations/556/\<ModelName\>/" and reports statistics over those errors.
8. GNN.py : Class defining the spatiotemporal model proposed and tested in this project.
9. Graph.py : Data class defining all graph construction methods.
10. LSTM.py : Class defining the temporal model developed in prior work.
11. MapModelIdsToConfigurations.py : Script that gathers model IDs and their associated configuration into a single file for easier look-up.
12. MapModelIdsToErrors.py : Script that gathers model IDs and their associated subbasin errors into a single file for easier look-up.
13. Model.py : Baseline class for PyTorch machine learning models. All models implemented will inherit from this class and PyTorch's Module class.
14. NetworkProperties.py : Script defining a set of functions for computing core metrics to characterize a graph.
15. Plotting.py : Class defining all plotting routines
16. SpatialData.py : Data class defining all routines for loading, preprocessing, partitioning, etc for spatially distributed data.
17. SpatiotemporalData.py : Data class defining all routines for loading, preprocessing, partitioing, etc for spatially and temporally distributed data.
18. Utility.py : Class defining all miscellaneous but common routines.
19. Variables.py : Class defining all variables and their default values across many, if not all, classes of this project.
