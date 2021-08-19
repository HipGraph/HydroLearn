# HydroLearn: A Pipeline for Training Hydrologic ML Models
## Author(s): Nicholas Majeske
## Contact: nmajeske@iu.edu

### Source Code Description
1. Arguments.py : Custom argument parser that parses arguments from sys.argv into a Container object
2. Container.py : Generalized data container class to facilitate partitioning of data. All other data classes that employ partitioning will inherit from this class.
3. Data.py : The all-encompassing data class that contains all data.
4. Driver.py : Handles (distributed or non-distributed) invocation of Execute.py
5. Execute.py : The pipeline script which defines data loading, pre-processing, model training/evaluation, and plotting.
6. Experiments.py : A script where experiments can be defined and executed. Each experiment class defined the experiment via a set of arguments and then builds those arguments and executes the pipeline.
7. GatherErrors.py : Script that gathers subbasin errors of all evaluated models in "Evaluations/\<ExperimentName\>/\<ModelName\>/" and reports statistics, etc..
8. Models/Model.py : The base class for PyTorch machine learning models. All models implemented will inherit from this class and, by inheritance, PyTorch's Module class. See Models/LSTM.py for an example.
11. MapModelIdsToConfigurations.py : A script that gathers model IDs and their associated configuration into a single file for easier look-up.
12. MapModelIdsToErrors.py : A script that gathers model IDs and their associated subbasin errors into a single file for easier look-up.
15. Plotting.py : A class defining all plotting routines
16. SpatialData.py : A data class defining all routines for loading, pre-processing, partitioning, etc for data that is spatially distributed.
17. SpatiotemporalData.py : A data class defining all routines for loading, pre-processing, partitioning, etc for data that is both spatially and temporally distributed.
18. Utility.py : A class defining all miscellaneous/common routines.
19. Variables.py : A data class defining all variables and their default values.
