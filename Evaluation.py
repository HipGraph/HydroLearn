import numpy as np

import Utility as util
from Container import Container


##############################
### Classification Metrics ###
##############################
# Source : https://en.wikipedia.org/wiki/Confusion_matrix


def div(numer, denom, on_div_0=1):
    numer = numer.astype(float)
    denom = denom.astype(float)
    out = on_div_0 * np.ones_like(numer)
    where = denom != 0
    return np.divide(numer, denom, out=out, where=where)


def confusion_matrix(y, yhat, **kwargs):
    # Unpack args
    axis = kwargs.get("axis", 0)
    mask = kwargs.get("mask", None)
    classes = kwargs.get("classes", None)
    debug = kwargs.get("debug", 0)
    # Compute
    if classes is None:
        classes = np.unique(y)
    n_classes = len(classes)
    shape = (n_classes, n_classes)
    if not axis is None:
        if isinstance(axis, int):
            axis = (axis,)
        shape += tuple(np.take(y.shape, util.list_subtract(list(range(y.ndim)), list(axis))))
    if debug:
        print("axis =", axis)
        print("shape =", shape)
        input()
    cm = np.zeros(shape, dtype=int)
    for i, class_i in enumerate(classes):
        for j, class_j in enumerate(classes):
            if mask is None:
                cm[i,j] += np.sum(np.logical_and(y == class_i, yhat == class_j).astype(int), axis)
            else:
                cm[i,j] += np.sum(
                    np.logical_and(np.logical_and(y == class_i, yhat == class_j), mask).astype(int), 
                    axis
                )
    if debug:
        print("cm =", cm.shape)
        print(cm)
        input()
    return cm


def negative(y, yhat, **kwargs):
    cm = confusion_matrix(y, yhat, **kwargs)
    return np.reshape(np.sum(cm[0,:], 0), cm.shape[2:])


def positive(y, yhat, **kwargs):
    cm = confusion_matrix(y, yhat, **kwargs)
    if 0:
        cm = confusion_matrix(y, yhat, **kwargs, debug=1)
        intr = confusion_matrix(y, yhat, **kwargs)[:,1]
        res = np.sum(confusion_matrix(y, yhat, **kwargs)[:,1], 0)
        print(y.shape, yhat.shape)
        print("cm =", cm.shape)
        print(cm)
        print("intr =", intr.shape)
        print(intr)
        print("res =", res.shape)
        print(res)
        input()
    return np.reshape(np.sum(cm[1,:], 0), cm.shape[2:])


def predicted_negative(y, yhat, **kwargs):
    cm = confusion_matrix(y, yhat, **kwargs)
    return np.reshape(np.sum(cm[:,0], 0), cm.shape[2:])


def predicted_positive(y, yhat, **kwargs):
    cm = confusion_matrix(y, yhat, **kwargs)
    if 0:
        cm = confusion_matrix(y, yhat, **kwargs, debug=1)
        intr = confusion_matrix(y, yhat, **kwargs)[:,1]
        res = np.sum(confusion_matrix(y, yhat, **kwargs)[:,1], 0)
        print(y.shape, yhat.shape)
        print("cm =", cm.shape)
        print(cm)
        print("intr =", intr.shape)
        print(intr)
        print("res =", res.shape)
        print(res)
        input()
    return np.reshape(np.sum(cm[:,1], 0), cm.shape[2:])


def false_negative(y, yhat, **kwargs):
    cm = confusion_matrix(y, yhat, **kwargs)
    return cm[1,0]


def false_positive(y, yhat, **kwargs):
    cm = confusion_matrix(y, yhat, **kwargs)
    return cm[0,1]


def true_negative(y, yhat, **kwargs):
    cm = confusion_matrix(y, yhat, **kwargs)
    return cm[0,0]


def true_positive(y, yhat, **kwargs):
    cm = confusion_matrix(y, yhat, **kwargs)
    return cm[1,1]


def false_negative_rate(y, yhat, **kwargs):
    cm = confusion_matrix(y, yhat, **kwargs)
    return cm[1,0] / np.sum(cm[1,:], 0)


def false_positive_rate(y, yhat, **kwargs):
    cm = confusion_matrix(y, yhat, **kwargs)
    return cm[0,1] / np.sum(cm[0,:], 0)


def true_negative_rate(y, yhat, **kwargs):
    cm = confusion_matrix(y, yhat, **kwargs)
    return cm[0,0] / np.sum(cm[0,:], 0)


def true_positive_rate(y, yhat, **kwargs):
    cm = confusion_matrix(y, yhat, **kwargs)
    return cm[1,1] / np.sum(cm[1,:], 0)


def prevalence(y, yhat, **kwargs):
    n = negative(y, yhat, **kwargs)
    p = positive(y, yhat, **kwargs)
    return div(p, (p + n))


def informedness(y, yhat, **kwargs): # informedness, bookmaker (BM)
    tnr = true_negative_rate(y, yhat, **kwargs)
    tpr = true_positive_rate(y, yhat, **kwargs)
    return tpr + tnr - 1


def accuracy(y, yhat, **kwargs): # accuracy
    n = negative(y, yhat, **kwargs)
    p = positive(y, yhat, **kwargs)
    tn = true_negative(y, yhat, **kwargs)
    tp = true_positive(y, yhat, **kwargs)
    res = div(tp + tn, p + n)
    if 0:
        print("n =", n.shape)
        print(n)
        print("p =", p.shape)
        print(p)
        print("tn =", tn.shape)
        print(tn)
        print("tp =", tp.shape)
        print(tp)
        print("res =", res.shape)
        print(res)
        input()
    return div(tp + tn, p + n)


def prevalence_threshold(y, yhat, **kwargs): # prevalence threshold (PT)
    fpr = false_positive_rate(y, yhat, **kwargs)
    tpr = true_positive_rate(y, yhat, **kwargs)
    return div((tpr * fpr)**(1/2) - fpr, tpr - fpr)


def balanced_accuracy(y, yhat, **kwargs): # balanced accuracy
    tnr = true_negative_rate(y, yhat, **kwargs)
    tpr = true_positive_rate(y, yhat, **kwargs)
    return (tpr + tnr) / 2


def false_omission_rate(y, yhat, **kwargs): # FOR
    pn = predicted_negative(y, yhat, **kwargs)
    fn = false_negative(y, yhat, **kwargs)
    return div(fn, pn)


def false_discovery_rate(y, yhat, **kwargs): # FDR
    pp = predicted_positive(y, yhat, **kwargs)
    fp = false_positive(y, yhat, **kwargs)
    return div(fp, pp)


def negative_predictive_value(y, yhat, **kwargs): # NPV
    pn = predicted_negative(y, yhat, **kwargs)
    tn = true_negative(y, yhat, **kwargs)
    return div(tn, pn)


def positive_predictive_value(y, yhat, **kwargs): # PPV
    pp = predicted_positive(y, yhat, **kwargs)
    tp = true_positive(y, yhat, **kwargs)
    return div(tp, pp)


def f1_score(y, yhat, **kwargs): # F1
    ppv = positive_predictive_value(y, yhat, **kwargs)
    tpr = true_positive_rate(y, yhat, **kwargs)
    return div(2 * ppv * tpr, ppv + tpr)


def fowlkes_mallows_index(y, yhat, **kwargs): # FM
    ppv = positive_predictive_value(y, yhat, **kwargs)
    tpr = true_positive_rate(y, yhat, **kwargs)
    return (ppv * tpr)**(1/2)


def positive_likelihood_ratio(y, yhat, **kwargs): # PLR
    fpr = false_positive_rate(y, yhat, **kwargs)
    tpr = true_positive_rate(y, yhat, **kwargs)
    return div(tpr, fpr)


def negative_likelihood_ratio(y, yhat, **kwargs): # NLR
    tnr = true_negative_rate(y, yhat, **kwargs)
    fnr = false_negative_rate(y, yhat, **kwargs)
    return div(fnr, tnr)


def markedness(y, yhat, **kwargs): # MK
    ppv = positive_predictive_value(y, yhat, **kwargs)
    npv = negative_predictive_value(y, yhat, **kwargs)
    return ppv + npv - 1


def diagnostic_odds_ratio(y, yhat, **kwargs): # DOR
    nlr = negative_likelihood_ratio(y, yhat, **kwargs)
    plr = positive_likelihood_ratio(y, yhat, **kwargs)
    return div(plr, nlr)


def matthews_correlation_coefficient(y, yhat, **kwargs): # MCC
    tpr = true_positive_rate(y, yhat, **kwargs)
    tnr = true_negative_rate(y, yhat, **kwargs)
    ppv = positive_predictive_value(y, yhat, **kwargs)
    npv = negative_predictive_value(y, yhat, **kwargs)
    fnr = false_negative_rate(y, yhat, **kwargs)
    fpr = false_positive_rate(y, yhat, **kwargs)
    _for = false_omission_rate(y, yhat, **kwargs)
    fdr = false_discovery_rate(y, yhat, **kwargs)
    return (tpr * tnr * ppv * npv)**(1/2) - (fnr * fpr * _for * fdr)**(1/2)


def threat_score(y, yhat, **kwargs): # TS
    fn = false_negative(y, yhat, **kwargs)
    fp = false_positive(y, yhat, **kwargs)
    tp = true_positive(y, yhat, **kwargs)
    if 0:
        print("fn =", fn.shape)
        print("fp =", fp.shape)
        print("tp =", tp.shape)
        input()
    return div(tp, tp + fn + fp)


# Aliases


def precision(y, yhat, **kwargs):
    return positive_predictive_value(y, yhat, **kwargs)


def recall(y, yhat, **kwargs):
    return true_positive_rate(y, yhat, **kwargs)


classification_metricfn_dict = {
#    "CM": confusion_matrix, 
    "N": negative, 
    "P": positive, 
    "PN": predicted_negative, 
    "PP": predicted_positive, 
    "FN": false_negative, # type 2 error, miss, underestimation
    "FP": false_positive, # type 1 error, false alarm, overestimation
    "TN": true_negative, # correct rejection
    "TP": true_positive, # hit
    "FNR": false_negative_rate, # miss rate
    "FPR": false_positive_rate, # probability of false alarm, fall-out
    "TNR": true_negative_rate, # specificity (SPC), selectivity
    "TPR": true_positive_rate, # recall, sensitivity
    "PREV": prevalence, 
    "BM": informedness, # Informedness, bookmaker informedness (BM)
    "ACC": accuracy, 
    "PT": prevalence_threshold,
    "BA": balanced_accuracy, 
    "FOR": false_omission_rate, 
    "FDR": false_discovery_rate, 
    "NPV": negative_predictive_value, 
    "PPV": positive_predictive_value, # precision
    "F1": f1_score, 
    "FM": fowlkes_mallows_index, 
    "PLR": positive_likelihood_ratio, 
    "NLR": negative_likelihood_ratio, 
    "MK": markedness, # deltaP
    "DOR": diagnostic_odds_ratio, 
    "MCC": matthews_correlation_coefficient, 
    "TS": threat_score, # critical success index (CSI), Jaccard index
    "PREC": precision, 
    "RECA": recall, 
}



##########################
### Regression Metrics ###
##########################


def mean_absolute_error(y, yhat, **kwargs):
    # Unpack args
    axis = kwargs.get("axis", 0)
    mask = kwargs.get("mask", None)
    # Setup
    if not mask is None:
        yhat = np.copy(yhat)
        yhat[~mask] = y[~mask]
    # Compute
    return np.mean(np.abs(y - yhat), axis)


def mean_square_error(y, yhat, **kwargs):
    # Unpack args
    axis = kwargs.get("axis", 0)
    mask = kwargs.get("mask", None)
    # Setup
    if not mask is None:
        yhat = np.copy(yhat)
        yhat[~mask] = y[~mask]
    # Compute
    return np.mean((y - yhat)**2, axis)


def mean_absolute_percentage_error(y, yhat, **kwargs):
    # Unpack args
    axis = kwargs.get("axis", 0)
    eps = kwargs.get("eps", np.finfo(np.float64).eps)
    mask = kwargs.get("mask", None)
    # Setup
    if not mask is None:
        yhat = np.copy(yhat)
        yhat[~mask] = y[~mask]
    # Compute
    return np.mean(100 * np.minimum(np.abs((y - yhat) / np.maximum(np.abs(y), eps)), 1), axis)


def root_mean_square_error(y, yhat, **kwargs):
    # Unpack args
    mask = kwargs.get("mask", None)
    # Setup
    if not mask is None:
        yhat = np.copy(yhat)
        yhat[~mask] = y[~mask]
    # Compute
    return mean_square_error(y, yhat, **kwargs)**(1/2)


def normalized_root_mean_square_error(y, yhat, **kwargs):
    # Unpack args
    axis = kwargs.get("axis", 0)
    mins, maxes = kwargs["mins"], kwargs["maxes"]
    eps = kwargs.get("eps", np.finfo(np.float64).eps)
    mask = kwargs.get("mask", None)
    # Setup
    if not mask is None:
        yhat = np.copy(yhat)
        yhat[~mask] = y[~mask]
    # Compute
    return root_mean_square_error(y, yhat, **kwargs) / np.maximum(maxes - mins, eps)


def under_prediction_rate(y, yhat, **kwargs):
    # Unpack args
    axis = kwargs.get("axis", 0)
    margin = kwargs.get("margin", 5/100)
    mask = kwargs.get("mask", None)
    # Setup
    if not mask is None:
        yhat = np.copy(yhat)
        yhat[~mask] = y[~mask]
    # Compute
    up_count = np.sum(yhat < ((1 - margin) * y), axis)
    N = np.prod(np.take(y.shape, axis))
    return up_count / N


def over_prediction_rate(y, yhat, **kwargs):
    # Unpack args
    axis = kwargs.get("axis", 0)
    margin = kwargs.get("margin", 5/100)
    mask = kwargs.get("mask", None)
    # Setup
    if not mask is None:
        yhat = np.copy(yhat)
        yhat[~mask] = y[~mask]
    # Compute
    op_count = np.sum(yhat > ((1 + margin) * y), axis)
    N = np.prod(np.take(y.shape, axis))
    return op_count / N


def miss_rate(y, yhat, **kwargs):
    # Compute
    return under_prediction_rate(y, yhat, **kwargs) + over_prediction_rate(y, yhat, **kwargs)


def root_relative_square_error(y, yhat, **kwargs):
    # Unpack args
    axis = kwargs.get("axis", 0)
    mean = kwargs["means"]
    eps = kwargs.get("eps", np.finfo(np.float64).eps)
    mask = kwargs.get("mask", None)
    # Setup
    if not mask is None:
        y = np.copy(y)
        yhat = np.copy(yhat)
        y[~mask] = np.broadcast_to(mean, y.shape)[~mask]
        yhat[~mask] = np.broadcast_to(mean, yhat.shape)[~mask]
    # Compute
    numer = np.sum((y - yhat)**2, axis)
    denom = np.sum((y - mean)**2, axis)
    return np.divide(numer, denom)**(1/2)


def correlation(y, yhat, **kwargs):
    axis = kwargs.get("axis", 0)
    mean = kwargs["means"]
    eps = kwargs.get("eps", np.finfo(np.float64).eps)
    mask = kwargs.get("mask", None)
    # Setup
    if not mask is None:
        y = np.copy(y)
        yhat = np.copy(yhat)
        y[~mask] = np.broadcast_to(mean, y.shape)[~mask]
        yhat[~mask] = np.broadcast_to(mean, yhat.shape)[~mask]
    # Compute
    numer = np.sum((yhat - mean) * (y - mean), axis)
    denom = np.sum((yhat - mean)**2, axis)**(1/2) * np.sum((y - mean)**2, axis)**(1/2)
    return np.divide(numer, denom)


regression_metricfn_dict = {
    "MAE": mean_absolute_error,
    "MSE": mean_square_error,
    "MAPE": mean_absolute_percentage_error,
    "RMSE": root_mean_square_error,
    "NRMSE": normalized_root_mean_square_error,
    "UPR": under_prediction_rate,
    "OPR": over_prediction_rate,
    "MR": miss_rate, 
    "RRSE": root_relative_square_error, 
    "CORR": correlation, 
}


#####################
### Other Methods ###
#####################


metric_fn_dict = util.merge_dicts(classification_metricfn_dict, regression_metricfn_dict)


def evaluate(y, yhat, datatype, metrics="*", **kwargs):
    # Handle arguments
    if not (isinstance(metrics, str) or (isinstance(metrics, (tuple, list)) and isinstance(metrics[0], str))):
        type_str = type(metrics)
        if isinstance(metrics, (tuple, list)):
            type_str = "%s of %s" % (type(metrics), type(metrics[0]))
        raise TypeError("Input metrics may be str or tuple/list of str. Received %s" % (type_str))
    if metrics == "*":
        metrics = list(metric_fn_dict.keys())
    elif metrics == "classification":
        metrics = list(classification_metricfn_dict.keys())
    elif metrics == "regression":
        metrics = list(regression_metricfn_dict.keys())
    else:
        raise ValueError("Input metrics may be one of \"*\", \"classification\", or \"regression\" when str.")
    # Start
    con = Container()
    if 0:
        print(y.shape)
        print(y)
        print(yhat.shape)
        print(yhat)
        input()
    if datatype == "spatial":
        kwargs = util.merge_dicts({"axis": tuple(_ for _ in range(y.ndim-1))}, kwargs)
        for metric in metrics:
            scores = metric_fn_dict[metric](y, yhat, **kwargs)
            con.set(metric, scores)
            if 0:
                print(metric_fn_dict[metric])
                print(metric, "=", scores, con.get(metric))
                input()
    elif datatype == "temporal": 
        kwargs = util.merge_dicts({"axis": tuple(_ for _ in range(y.ndim-1))}, kwargs)
        for metric in metrics:
            con.set(metric, metric_fn_dict[metric](y, yhat, **kwargs))
    elif datatype == "spatiotemporal":
        kwargs = util.merge_dicts({"axis": tuple(_ for _ in range(y.ndim-2))}, kwargs)
        for metric in metrics:
            con.set(metric, metric_fn_dict[metric](y, yhat, **kwargs))
    elif datatype == "graph":
        raise NotImplementedError()
    return con


def evaluate_datasets(ys, yhats, datasets, datatypes, partitions, metrics="*", **kwargs):
    """ Computes a variety of scoring metrics to evaluate the performance of a model given groundtruth and predicted values.

    Arguments
    ---------
    ys : ndarray or tuplei/list of ndarray
    yhats : ndarray or tuple/list of ndarray
    datasets: Data.Data object or tuple/list of Data.Data objects
    datatypes : str or tuple/list of str
    partitions : str or tuple/list of str
    metrics : str or tuple/list of str
    kwargs : ...

    Returns
    -------

    """
    # Handle args
    if not isinstance(ys, (tuple, list)):
        ys = [ys]
    if not isinstance(yhats, (tuple, list)):
        yhats = [yhats for _ in ys]
    if not isinstance(datasets, (tuple, list)):
        datasets = [datasets for _ in ys]
    if isinstance(datatypes, str):
        datatypes = [datatypes for _ in ys]
    if isinstance(partitions, str):
        partitions = [partitions for _ in ys]
    # Start
    con = Container()
    for i in range(len(ys)):
        stats = datasets[i].get(datatypes[i]).statistics.to_dict()
        axis = (-1,)
        indices = datasets[i].get(datatypes[i]).misc.response_indices
        if datatypes[i] in ("spatial", "temporal"):
            pass
        elif datatypes[i] == "spatiotemporal":
            axis = (-2, -1)
            indices = (
                datasets[i].get(datatypes[i]).original.get("spatial_indices", partitions[i]), 
                datasets[i].get(datatypes[i]).misc.response_indices
            )
        elif datatypes[i] == "graph":
            raise NotImplementedError()
        else:
            raise ValueError()
        stats = datasets[i].get(datatypes[i]).filter_axis(stats, axis, indices)
        _kwargs = util.merge_dicts(
            util.remap_dict(
                stats, 
                {"minimums": "mins", "maximums": "maxes", "medians": "meds", "standard_deviations": "stds"}
            ), 
            kwargs
        )
        eval_con = evaluate(ys[i], yhats[i], datatypes[i], metrics, **_kwargs)
        names = eval_con.get_names()
        values = [eval_con.get(_) for _ in names]
        con.set(names, values, partitions[i], multi_value=1)
    return con


def evaluation_to_report(eval_con, datasets, datatypes, partitions, n_decimal=8):
    if isinstance(partitions, str):
        partitions = [partitions]
    if not isinstance(datasets, (tuple, list)):
        datasets = [datasets for _ in partitions]
    if not isinstance(datatypes, (tuple, list)):
        datatypes = [datatypes for _ in datatypes]
    metrics = eval_con.get_names()
    lines = []
    for metric in metrics:
        for dataset, datatype, partition in zip(datasets, datatypes, partitions):
            feature_labels = dataset.get(datatype).misc.response_features
            scores = eval_con.get(metric, partition)
            # Add partition-level metric score
            if 0:
                print(eval_con)
                print(metric)
                print(scores.shape)
                print(scores)
                input()
            score = np.round(np.mean(scores), n_decimal)
            if int(score) == score:
                score = int(score)
            lines.append("%s %s = %s" % (partition, metric, score))
            for j, feature_label in enumerate(feature_labels):
                # Add feature-level metric score for this partition
                score = np.round(np.mean(scores[...,j]), n_decimal)
                if int(score) == score:
                    score = int(score)
                lines.append("\t%s %s = %s" % (feature_label, metric, score))
                if datatype == "spatiotemporal":
                    spatial_label_field = dataset.get(datatype).misc.spatial_label_field
                    spatial_labels = dataset.get(datatype).original.get("spatial_labels", partition)
                    for k, spatial_label in enumerate(spatial_labels):
                        # Add spatial-level metric score for this feature and partition
                        score = np.round(scores[k,j], n_decimal)
                        if int(score) == score:
                            score = int(score)
                        lines.append("\t\t%s %8s %s = %s" % (spatial_label_field, spatial_label, metric, score))
    return "\n".join(lines)


def curate_evaluation_report(ys, yhats, datasets, datatypes, partitions, metrics, **kwargs):
    return evaluation_to_report(
        evaluate_datasets(ys, yhats, datasets, datatypes, partitions, metrics, **kwargs), 
        datasets, datatypes, partitions
    )
