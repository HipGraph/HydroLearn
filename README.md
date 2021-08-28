# HydroLearn: A Pipeline for Training Hydrologic ML Models
## Author(s): Nicholas Majeske
## Contact: nmajeske@iu.edu

### Script Descriptions
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

### Cloning

git clone git@github.com:HipGraph/HydroLearn.git --recurse-submodules

### Data Integration
In order to feed new data into HydroLearn, users will need to complete the following steps:
1. Verify data format
    Dataset loading is currently implemented for spatial and spatiotemporal data. Loading assumes each data file is comma-separated (.csv) and requires the following format:
    - Spatial Data
        For spatial data containing S spatial elements and F spatial features, loading requires the file to contain S lines of F comma-separated features.
    - Spatiotemporal Data
        For spatiotemporal data containing T time-steps, S spatial elements, and F spatiotemporal features, loading requires the file to contain TxS lines of F comma-separated features.
    For both spatial and spatiotemporal data, spatial elements must be listed contiguously (see Data/WabashRiver/Observed/Spatiotemporal.csv). Finally, labels for each time-step and spatial element are required.
2. Create a dataset directory and add data files
    All datasets are stored in their own sub-directory under Data. Simply create a new directory under Data and add all data files to it.
3. Implement a DatasetVariables module
    The pipeline recognizes datasets by searching the Data directory (recursively) for all instances of DatasetVariables.py. Users must implement this module and place the script file at the root of their dataset directory. As an example, the Wabash River ground truth dataset is setup with its DatasetVariables.py module in Data/WabashRiver/Observed/. To facilitate user implementation of the DatasetVariables module, a template is included under Data/DatasetVariablesTemplate.py and lists all variables that must be defined. It is recommended that users start witj this template and follow the Wabash River DatasetVariables module as an example.

### Model Integration
In order add new models to HydroLearn, users will need to complete the following steps:
1. Implement a Model module
    The pipeline recognizes models by searching the Models directory (non-recursively) for all modules with the exception of __init__.py and Model.py. 
    Model operations including initialization, optimization, prediction, etc, are defined and operated by the model itself while HydroLearn simply calls a select few. 
    Currently, HydroLearn is designed to work with models implemented in PyTorch but is flexible enough to allow the incorporation of models implemented in Tensorflow (see GEOMAN.py). 
    As a result, models currently implemented for HydroLearn inherit from a Model class implemented in the Models/Model.py module and this model inherits from PyTorch's neural network module. 
    To facilitate user implementation of a model module, a template is included under Models/ModelTemplate.py. 
    It is recommended that users start with this template and follow the LSTM model module under Models/LSTM.py as an example. 
