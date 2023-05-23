import pandas as pd
import numpy as np

import Utility as util


def impute(df, data_type, method, var):
    debug = 0
    if debug:
        print(df)
    imputed_mask = df.isna()
    if data_type == "spatial":
        if method in ["frwd-fill", "back-fill", "frwdback-fill"]:
            raise ValueError("Cannot frwd-fill, back-fill, or frwdback-fill non-temporal data-type \"spatial\"")
        elif method in ["min", "max", "mean", "median", "std"]:
            # Function pd.DataFrame.transform(method) causes error: "ValueError: Function did not transform"
            #   However, the source code simply gets (via getattr()) and and applies function "method". 
            #   The code below is equivalent to: df = df.fillna(df.transform(method))
            df.fillna(getattr(df, method)(numeric_only=True), inplace=True)
        else:
            raise NotImplementedError(method)
    elif data_type == "temporal":
        if method == "frwd-fill":
            df.fillna(method="ffill", inplace=True)
        elif method == "back-fill":
            df.fillna(method="bfill", inplace=True)
        elif method == "frwdback-fill":
            df.fillna(method="ffill", inplace=True)
            df.fillna(method="bfill", inplace=True)
        elif method in ["min", "max", "mean", "median", "std"]:
            df.fillna(getattr(df, method)(), inplace=True)
        else:
            raise NotImplementedError(method)
    elif data_type == "spatiotemporal":
        if method == "frwd-fill":
            df[var.feature_fields] = df.groupby(var.spatial_label_field)[var.feature_fields].fillna(method="ffill")
        elif method == "back-fill":
            df[var.feature_fields] = df.groupby(var.spatial_label_field)[var.feature_fields].fillna(method="bfill")
        elif method == "frwdback-fill":
            df[var.feature_fields] = df.groupby(var.spatial_label_field)[var.feature_fields].fillna(method="ffill")
            df[var.feature_fields] = df.groupby(var.spatial_label_field)[var.feature_fields].fillna(method="bfill")
        elif method in ["min", "max", "mean", "median", "std"]:
            df.fillna(df.groupby(var.spatial_label_field).transform(method), inplace=True)
        elif "periodic-" in method:
            _method = method.split("-")[-1]
            if debug:
                print(var)
            spatial_labels = df[var.spatial_label_field].unique()
            n_spatial = len(spatial_labels)
            temporal_labels = df[var.temporal_label_field].unique()
            n_temporal = len(temporal_labels)
            if debug:
                print(spatial_labels)
                print(n_spatial)
                print(temporal_labels)
                print(n_temporal)
            periodic_indices = util.temporal_labels_to_periodic_indices(
                temporal_labels, 
                var.temporal_seasonality_period, 
                var.temporal_resolution, 
                var.temporal_label_format
            )
            if var.shape == ["spatial", "temporal", "feature"]:
                periodic_indices = np.tile(periodic_indices, n_spatial)
            elif var.shape == ["temporal", "spatial", "feature"]:
                periodic_indices = np.repeat(periodic_indices, n_spatial)
            else:
                raise ValueError(var.shape)
            if _method in ["min", "max", "mean", "median", "std"]:
                if debug:
                    print("A")
                    indices = np.repeat(np.arange(4) * 41941, 4) + np.arange(16) % 4
                    print("B")
                df["__periodic_indices__"] = periodic_indices
                if debug:
                    print(df.loc[indices,:])
                df.fillna(
                    df.groupby(["__periodic_indices__", var.spatial_label_field]).transform(
                        _method#, numeric_only=True
                    ), 
                    inplace=True
                )
                df.drop("__periodic_indices__", axis=1, inplace=True)
                if debug:
                    print("C")
                    print(df.loc[indices,:])
                    print("D")
                    print(df.loc[indices,:])
                    print("E")
                    missing = missing_values(df, data_type, var)
                    print(missing)
                    print("F")
                    print(missing.sum())
                    print(missing.loc[missing["dv_va"] != 0])
                # Impute with mean across all of time for each spatial element
                df, _ = impute(df, data_type, _method, var)
                # Impute with mean across all of time and spatial elements
                df, _ = impute(df, "spatial", _method, var)
                if debug:
                    missing = missing_values(df, data_type, var)
                    print(missing)
                    print("G")
                    print(missing.sum())
                    print(missing.loc[missing["dv_va"] != 0])
                    input("WAIT")
        else:
            raise NotImplementedError(method)
    elif data_type == "graph":
        raise NotImplementedError(data_type)
    else:
        raise NotImplementedError(data_type)
    return df, imputed_mask


def missing_values(df, data_type, var, normalize=True):
    if data_type == "spatial":
        n_spatial = len(df[var.spatial_label_field])
        missing = df.drop(var.spatial_label_field, axis=1).isna().sum().reset_index()
        missing = pd.DataFrame(
            util.to_dict(list(missing["index"]), [[val] for val in missing[missing.columns[-1]]])
        )
        if normalize:
            missing = (missing / n_spatial * 100).round(2)
    elif data_type == "temporal":
        n_temporal = len(df[var.temporal_label_field])
        missing = df.drop(var.temporal_label_field, axis=1).isna().sum().reset_index()
        missing = pd.DataFrame(
            util.to_dict(list(missing["index"]), [[val] for val in missing[missing.columns[-1]]])
        )
        if normalize:
            missing = (missing / n_temporal * 100).round(2)
    elif data_type == "spatiotemporal":
        n_temporal = len(df[var.temporal_label_field].unique())
        missing = df.drop([var.spatial_label_field, var.temporal_label_field], axis=1).isna().groupby(
            df[var.spatial_label_field],
            sort=False
        ).sum().reset_index()
        if normalize:
            cols = util.list_subtract(var.feature_fields, [var.spatial_label_field, var.temporal_label_field])
            missing[cols] = (missing[cols] / n_temporal * 100).round(2)
    elif data_type == "graph":
        raise NotImplementedError(data_type)
    else:
        raise NotImplementedError(data_type)
    return missing


def missing_value_matrix(df, data_type, var):
    if data_type == "spatial":
        n_spatial = len(df)
        n_feature = len(df[var.feature_fields].columns)
        M = np.transpose(df[var.feature_fields].isna().to_numpy())
    elif data_type == "temporal":
        pass
    elif data_type == "spatiotemporal":
        n_temporal = len(df[var.temporal_label_field].astype(str).unique())
        n_spatial = len(df[var.spatial_label_field].astype(str).unique())
        n_feature = len(df[var.feature_fields].columns)
        M = {}
        for feature_field in var.feature_fields:
            if feature_field == var.spatial_label_field or feature_field == var.temporal_label_field:
                continue
            cols = [var.temporal_label_field, var.spatial_label_field, feature_field]
            _df = df[cols].pivot(index=var.spatial_label_field, columns=var.temporal_label_field)
            M[feature_field] = _df.isna().to_numpy()
        missing = df.drop([var.spatial_label_field, var.temporal_label_field], axis=1).isna().groupby(
            df[var.spatial_label_field],
            sort=False
        )
    elif data_type == "graph":
        raise NotImplementedError(data_type)
    else:
        raise NotImplementedError(data_type)
    return M
