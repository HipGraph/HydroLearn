import itertools
import gzip
import pandas as pd
import os
import sys
import time
import re
import requests
import urllib.request
import webbrowser
import shutil
import glob
import numpy as np
import datetime as dt
from io import StringIO
from inspect import currentframe

from Data.DataSelection import DataSelection
from Data import Imputation
from Container import Container
import Utility as util


def get_token_regex(token, starts=True, inside=True, ends=True):
    if isinstance(token, list):
        return "|".join([get_token_regex(_token, starts, inside, ends) for _token in token])
    token = token.replace(" ", "\\s")
    regex = []
    if starts: regex.append("^%s\s" % (token))
    if inside: regex.append("\s%s\s" % (token))
    if ends: regex.append("\s%s$" % (token))
    regex = "|".join(regex)
    return regex


def find_all(string, tokens):
    try:
        matches = []
        for token in tokens:
            matches += list(re.finditer(get_token_regex(token), string))
    except:
        print(string, type(string))
    return matches


def get_substring(string, match_tokens, method="all_before", include_token=False, use_match=0):
    substring = None
    if method == "all_before":
        regex = "|".join([get_token_regex(token) for token in match_tokens])
        if include_token:
#            indices = list(match.end() for match in re.finditer(regex, string))
            indices = [match.end() for match in find_all(string, match_tokens)]
        else:
#            indices = list(match.start() for match in re.finditer(regex, string))
            indices = [match.start() for match in find_all(string, match_tokens)]
        indices = sorted(indices)
        if len(indices) > 0:
            idx = indices[use_match]
            substring = string[:idx].strip()
    elif method == "just_after":
        regex = "|".join(["(^|\s)%s\s" % (token) for token in match_tokens])
        if include_token:
            indices = list(match.start() for match in re.finditer(regex, string))
        else:
            indices = list(match.end() for match in re.finditer(regex, string))
        if len(indices) > 0:
            idx = indices[use_match]
            substring = string[idx:].split()[0]
            if include_token:
                substring = " ".join(string[idx:].split()[:2])
    else:
        raise NotImplementedError(method)
    return substring


def data_dir():
    return os.sep.join(os.path.realpath(__file__).replace(".py", "").split(os.sep)[:-1])


def download_dir():
    from pathlib import Path
    return os.sep.join([str(Path.home()), "Downloads"])


class Integrator:

    debug = 2

    def __init__(self):
        os.makedirs(self.root_dir(), exist_ok=True)
        os.makedirs(self.cache_dir(), exist_ok=True)
        os.makedirs(self.acquire_dir(), exist_ok=True)
        os.makedirs(self.convert_dir(), exist_ok=True)

    def name(self):
        return self.__class__.__name__

    def root_dir(self):
        return os.sep.join([data_dir(), self.name()]) 

    def cache_dir(self):
        return os.sep.join([self.root_dir(), "Integration", "Cache"])

    def acquire_dir(self):
        return os.sep.join([self.root_dir(), "Integration", "Acquired"])

    def convert_dir(self):
#        return os.sep.join([self.root_dir(), "Integration", "Converted"])
        return self.root_dir()

    def spatial_labels_inpath(self):
        return None

    def temporal_labels_inpath(self):
        return None

    def spatial_features_inpath(self):
        return None

    def temporal_features_inpath(self):
        return None

    def spatiotemporal_features_inpath(self):
        func_name = "%s.%s" % (self.__class__.__name__, currentframe().f_code.co_name)
        raise NotImplementedError("Implement %s() please!" % (func_name))

    def graph_features_inpath(self):
        return None

    def spatial_labels_fname(self):
        return "SpatialLabels.csv"

    def temporal_labels_fname(self):
        return "TemporalLabels.csv"

    def spatial_features_fname(self):
        return "Spatial.csv"

    def temporal_features_fname(self):
        return "Temporal.csv"

    def spatiotemporal_features_fname(self):
        return "Spatiotemporal.csv.gz"

    def graph_features_fname(self):
        return "Graph.csv"

    def spatial_labels_outpath(self):
        return os.sep.join([self.convert_dir(), self.spatial_labels_fname()])

    def temporal_labels_outpath(self):
        return os.sep.join([self.convert_dir(), self.temporal_labels_fname()])

    def spatial_features_outpath(self):
        return os.sep.join([self.convert_dir(), self.spatial_features_fname()])

    def temporal_features_outpath(self):
        return os.sep.join([self.convert_dir(), self.temporal_features_fname()])

    def spatiotemporal_features_outpath(self):
        return os.sep.join([self.convert_dir(), self.spatiotemporal_features_fname()])

    def graph_features_outpath(self):
        return os.sep.join([self.convert_dir(), self.graph_features_fname()])

    def acquire(self, args):
        func_name = "%s.%s" % (self.__class__.__name__, currentframe().f_code.co_name)
        raise NotImplementedError("Implement %s() please!" % (func_name))

    def convert(self, args):
        func_name = "%s.%s" % (self.__class__.__name__, currentframe().f_code.co_name)
        raise NotImplementedError("Implement %s() please!" % (func_name))


class Acquisition:
    
    def get_meteostat_data(coords, interval=[None, None], resolution=[1, "days"], **point_kwargs):
        from meteostat import Point, Stations, Hourly, Daily, Monthly
        from geopy import distance
        debug = 1
        Point.method = point_kwargs.get("method", "nearest")
        Point.radius = point_kwargs.get("radius", None)
        Point.alt_range = point_kwargs.get("alt_range", None)
        Point.max_count = point_kwargs.get("max_count", 4)
        Point.adapt_temp = point_kwargs.get("adapt_temp", True)
        Point.weight_dist = point_kwargs.get("weight_dist", 7/8)
        Point.weight_alt = point_kwargs.get("weight_alt", 1.0 - Point.weight_dist)
        if debug:
            print(
                Point.method, 
                Point.radius, 
                Point.alt_range, 
                Point.max_count, 
                Point.weight_dist, 
                Point.weight_alt, 
            )
        start_dt, end_dt = interval[0], interval[1]
        frmt = "%Y-%m-%d"
        temporal_labels = util.generate_temporal_labels(start_dt, end_dt, resolution, frmt, [1, 1])
        n_temporal = len(temporal_labels)
        if debug and coords.shape[-1] > 2:
            print(util.get_stats(coords[:,-1]))
        dfs = []
        elev = None
        for i in range(coords.shape[0]):
            lat, lon = coords[i,:2]
            if coords.shape[-1] > 2:
                elev = coords[i,-1]
            point = Point(lat, lon, elev)
            if resolution[1] == "hours":
#                df = Hourly(point, start_dt, end_dt)._data
                df = Hourly(point)._data
            elif resolution[1] == "days":
#                df = Daily(point, start_dt, end_dt)._data
                df = Daily(point)._data
            elif resolution[1] == "months":
#                df = Monthly(point, start_dt, end_dt)._data
                df = Monthly(point)._data
            else:
                raise NotImplementedError(resolution)
            if debug:
                print("%d/%d =>" % (i+1, coords.shape[0]))
            if df.empty:
                print(lat, lon, elev)
                print(df)
                print(((n_temporal - len(df)) + df.isna().sum()) / n_temporal * 100)
                raise ValueError("Empty DataFrame!")
            df = df.reset_index(level=1).reset_index(drop=True)
            df["time"] = df["time"].dt.strftime(frmt)
            _df = pd.DataFrame(index=range(n_temporal), columns=df.columns)
            _df["time"] = temporal_labels
            _df = _df.astype(df.dtypes)
            df = Conversion.dataframe_insert(_df, df, "time")
            dfs.append(df)
            if debug > 1:
                print(df)
        return dfs

    def get_latlon_elevation(lat, lon):
        import requests
        import urllib
        import pandas as pd
        lats, lons = lat, lon
        if isinstance(lat, float) and isinstance(lon, float):
            lats, lons = [lat], [lon]
        url = r"https://nationalmap.gov/epqs/pqs.php?"
        """Query service using lat, lon. add the elevation values as a new column."""
        elevations = []
        for _lat, _lon in zip(lats, lons):
            # define rest query params
            params = {"output": "json", "x": _lon, "y": _lat, "units": "Meters"}
            # format query string and return query value
            result = requests.get((url + urllib.parse.urlencode(params)))
            elevations.append(result.json()["USGS_Elevation_Point_Query_Service"]["Elevation_Query"]["Elevation"])
        if isinstance(lat, float) and isinstance(lon, float):
            return elevations[0]
        return elevations


class Conversion:

    def dataframe_insert(df_a, df_b, on):
        if not isinstance(on, str):
            raise ValueError("Input \"on\" must be a str. Received %s" % (type(on)))
        elif not (on in df_a.columns and on in df_b.columns):
            raise ValueError("Input \"on\" must be a column of df_a and df_b")
        mask_ab = df_a[on].isin(df_b[on])
        mask_ba = df_b[on].isin(df_a[on])
        df_a.loc[mask_ab] = df_b.loc[mask_ba].set_index(mask_ab[mask_ab].index)
        return df_a
        
    def name_from_huc_name(huc_name, **kwargs):
        if isinstance(huc_name, list):
            return [name_from_huc_name(_huc_name for _huc_name in huc_name)]
        elif not isinstance(huc_name, str):
            raise ValueError("Argument \"huc_name\" must be list or str. Received %s" % (type(huc_name)))
        return get_substring(huc_name, **kwargs)

    def name_from_huc12_name(huc12_name):
        clean_tokens = []
        clean_tokens += ["UPPER", "LOWER", "MIDDLE", "FRONTAL"]
        clean_tokens += ["EAST\sFORK", "NORTH\sFORK", "WEST\sFORK", "SOUTH\sFORK"]
#        clean_tokens += ["EAST", "NORTH", "WEST", "SOUTH"]
        clean_tokens += ["HEADWATERS"]
        regex_tokens = []
        regex_tokens += ["^%s\s" % (token) for token in clean_tokens]
        regex_tokens += ["\s%s\s" % (token) for token in clean_tokens]
        clean_regex = "|".join(regex_tokens)
        def get_name(name):
            if "-" in name:
                return re.split("-", name)[0]
            return name
        def clean_name(name):
            name = re.sub(clean_regex, "", name).strip()
            name = re.sub(clean_regex, "", name).strip()
            return name
        if isinstance(huc12_name, list):
            names = []
            for i, _huc12_name in enumerate(huc12_name):
                name = _huc12_name.upper()
                name = get_name(name)
                name = clean_name(name)
                names.append(name)
            return names
        elif not isinstance(huc12_name, str):
            raise ValueError("Argument \"huc_name\" must be list or str. Received %s" % (type(huc_name)))
        return clean_name(get_name(huc12_name))
    
    def name_from_gauge_name(gauge_names):
        if isinstance(gauge_names, str):
            return name_from_gauge_name([gauge_names])[0]
        elif not isinstance(gauge_names, (list, tuple)):
            raise NotImplementedError("Unknown type (%s) for argument \"gauge_name\"" % (type(gauge_name)))
        def clean_name(name, token_codes_map):
            name = name.upper().strip()
            name = re.sub("^\(.+\)\s|\s\(.+\)$|\s\(.+\)\s", " ", name) # remove anything with (...)
            name = re.sub("\s\s+", " ", name) # remove 2 or more spaces "  ..."
            for token, codes in token_codes_map.items():
                if len(codes) > 0:
                    regex = get_token_regex(codes)
                    name = re.sub(regex, " %s " % (token), name)
            return name
        watertoken_codes_map = {
            "BASIN": [], 
            "BRANCH": [], 
            "BOGUE": [], 
            "CANAL": [], 
            "CHANNEL": [], 
            "COULEE": [], 
            "CREEK": ["C", "CR"], 
            "DITCH": [], 
            "DIVERSION": ["DIV"], 
            "DRAIN": [], 
            "FLUME": [], 
            "FORK": [], 
            "LAKE": ["LK", "LAKES"], 
            "LATERAL": [], 
            "MANOMETER": [], 
            "OUTFALL": [], 
            "OUTLET": [], 
            "POND": [], 
            "RIVER": ["R"], 
            "SEWER": ["SWR"], 
            "SLOUGH": [], 
            "SPRING": ["SP", "SPRINGS"], 
            "TAILWATER": [], 
            "TARN": [], 
            "TRIBUTARY": ["TRIB"], 
            "WASTEWAY": ["WSTWY"], 
        }
        loctoken_codes_map = {
            "ABOVE": ["AB", "ABV"], 
            "AT": ["@"], 
            "BELOW": ["BL", "BLW"], 
            "IN": [], 
            "NEAR": ["NR"], 
            "ON": [], 
        }
        forktoken_codes_map = {
            "EAST FORK": ["EF"], 
            "NORTH FORK": ["NF"], 
            "WEST FORK": ["WF"], 
            "SOUTH FORK": ["SF"], 
            "MIDDLE FORK": ["MF"], 
            "CLEAR FORK": [], 
            "DRY FORK": [], 
        }
        branchtoken_codes_map = {
            "EAST BRANCH": [], 
            "NORTH BRANCH": [], 
            "WEST BRANCH": [], 
            "SOUTH BRANCH": [], 
            "MIDDLE BRANCH": [], 
        }
        misctoken_codes_map = {
            "FORK": ["F", "FK"], 
            "LITTLE": ["LTLE"], 
            "CANYON": ["CNYN"], 
            "SOUTH": ["SO"], 
        }
        token_codes_map = util.merge_dicts(
            util.merge_dicts(watertoken_codes_map, loctoken_codes_map), 
            util.merge_dicts(forktoken_codes_map, misctoken_codes_map), 
        )
        stream_tokens = ["CREEK", "RIVER"]
        water_tokens = list(watertoken_codes_map.keys())
        other_tokens = util.list_subtract(list(watertoken_codes_map.keys()), stream_tokens)
        loc_tokens = list(loctoken_codes_map.keys())
        fork_tokens = list(forktoken_codes_map.keys())
        branch_tokens = list(branchtoken_codes_map.keys())
        names, subnames = [], []
        for i, gauge_name in enumerate(gauge_names):
            cleaned_name = clean_name(gauge_name, token_codes_map)
            name = get_substring(cleaned_name, loc_tokens, "all_before", False, 0)
            if name is None:
                name = cleaned_name
            if True or not re.match("^DRAIN.*[0-9]\+$", name) is None:
                name = get_substring(name, water_tokens, "all_before", True, -1)
            if name is None:
                if 0:
                    raise ValueError(gauge_name, cleaned_name, name)
                print(gauge_name, cleaned_name, name)
                name = "?"
            names.append(name)
            subname = get_substring(name, fork_tokens, "all_before", True, -1)
            if not subname is None:
                names[-1] = names[-1].replace(subname, "").strip()
            elif "TRIBUTARY" in name:
                names[-1] = re.sub("^(EAST\s|NORTH\s|WEST\s|SOUTH\s|)TRIBUTARY\sTO|\sTRIBUTARY$", "", names[-1]).strip() 
                subname = "TRIBUTARY"
            elif "BRANCH" in name:
                names[-1] = re.sub("^(EAST\s|NORTH\s|WEST\s|SOUTH\s)BRANCH\s", "", names[-1]).strip()
                subname = "BRANCH"
            elif "UPPER" in name or "LOWER" in name:
                names[-1] = re.sub("^(UPPER|LOWER)\s", "", names[-1]).strip()
                subname = "UPPER" if "UPPER" in name else "LOWER"
            subnames.append(subname)
            for _ in ["CREEK", "RIVER", "LAKE"]:
                if len(name.split()) > 2 and _ in name:
                    names[-1] = re.sub(
                        get_token_regex(util.list_subtract(water_tokens, [_]), 0, 0), 
                        "", 
                        names[-1]
                    ).strip()
        def special_case(name):
            if re.match("(^|\s)UNNAMED(\s|$)", name):
                return name
                return "?"
            if name == "DRAINAGE DITCH":
                return name
                return "?"
            if re.match("^148TH\sAVE", name):
                return name
                return "?"
            return re.sub("\s(TVPI|MVID)(\s|$)(EAST|NORTH|WEST|SOUTH|)", "", name).strip()
        if 1:
            for i, (gauge_name, name, subname) in enumerate(zip(gauge_names, names, subnames)):
                names[i] = special_case(names[i])
        # Replace stream names that are missing spaces with space-seperated version
        #   e.g. replace "THREEMILE" with "THREE MILE" if both exist
        _names = np.unique(names)
        if 0:
            for _name in _names:
                if not " " in _name:
                    continue
                for i, (gauge_name, name, subname) in enumerate(zip(gauge_names, names, subnames)):
                    if names[i] == _name.replace(" ", ""):
                        print("%-*s -> %-*s" % (
                            max(len(_) for _ in names), name, max(len(_) for _ in names), name)
                        )
                        input()
                        names[i] = _name
        if 1:
            for i, (gauge_name, name, subname) in enumerate(zip(gauge_names, names, subnames)):
                print("%04d : %-*s -> %-*s - %-20s" % (
                    i, max(len(_) for _ in gauge_names), gauge_name, max(len(_) for _ in names), name, subname)
                )
            input()
        return names


class Electricity(Integrator):

    def spatiotemporal_features_inpath(self):
        return os.sep.join([self.acquire_dir(), "electricity.txt.gz"])

    def acquire(self, args):
        if not os.path.exists(self.spatiotemporal_features_inpath()):
            print("Download \"%s\" from sub-directory \"%s\" of repository at \"%s\" and save to acquisition directory \"%s\"" % (
                os.path.basename(self.spatiotemporal_features_inpath()), 
                "./electricity", 
                "https://github.com/laiguokun/multivariate-time-series-data", 
                self.acquire_dir(), 
            ))

    def convert(self, args):
        path = os.sep.join([self.acquire_dir(), "LD2011_2014.txt", "LD2011_2014.txt"])
        df = pd.read_csv(path, sep=";", decimal=",")
        if args.debug:
            print(df)
        spatial_labels = np.array(list(df.columns[1:]))
        temporal_labels = np.array(util.generate_temporal_labels("2012-01-01_00-00-00", 24*365*3, [1, "hours"]))
        temporal_labels = np.array([tmp.replace(" ", "_").replace(":", "-") for tmp in df.iloc[:,0]])
        features = df[df.columns[1:]].to_numpy()
        # Clearn by removing the first year (contains all zeros) and down-sampling to kWh (f / 4)
        if args.debug:
            print(spatial_labels.shape)
            print(spatial_labels)
            print(temporal_labels.shape)
            print(temporal_labels)
            print(features.shape)
            print(features)
        start = 4 * 24 * 365 
        temporal_labels = temporal_labels[start:]
        features = features[start:,:]
        n_temporal = len(temporal_labels)
        if args.debug:
            print(spatial_labels.shape)
            print(spatial_labels)
            print(temporal_labels.shape)
            print(temporal_labels)
            print(features.shape)
            print(features)
        sample_index = np.arange(0, n_temporal, 4)
        temporal_labels = temporal_labels[sample_index]
        features = (features / 4)[sample_index,:]
        if args.debug:
            print(spatial_labels.shape)
            print(spatial_labels)
            print(temporal_labels.shape)
            print(temporal_labels)
            print(features.shape)
            print(features)
        # Remove clients with too many missing values
        n_temporal, n_spatial = features.shape
        index = []
        threshold = 5000
        for i in range(n_spatial):
#            print(i, np.sum(features[:,i] == 0))
            if np.sum(features[:,i] == 0) > threshold:
                index.append(i)
        spatial_labels = np.delete(spatial_labels, index, -1)
        features = np.delete(features, index, -1)
        n_temporal, n_spatial = features.shape
        if args.debug:
            print(spatial_labels.shape)
            print(spatial_labels)
            print(temporal_labels.shape)
            print(temporal_labels)
            print(features.shape)
            print(features)
        df = pd.DataFrame({
            "date": np.repeat(temporal_labels, n_spatial), 
            "client": np.tile(spatial_labels, n_temporal), 
            "power_kWh": np.reshape(features, -1), 
        })
        if self.debug:
            print(df.loc[df["date"] == temporal_labels[0]])
            print(df.loc[df["date"] == temporal_labels[1]])
            print(df.loc[df["date"] == temporal_labels[-2]])
            print(df.loc[df["date"] == temporal_labels[-1]])
        pd.DataFrame({"date": temporal_labels}).to_csv(self.temporal_labels_outpath(), index=False)
        pd.DataFrame({"client": spatial_labels}).to_csv(self.spatial_labels_outpath(), index=False)
        df.to_csv(self.spatiotemporal_features_outpath(), index=False, compression="gzip")

    def _convert(self, args):
        # Convert Electricity
        paths_exist = [
            os.path.exists(self.temporal_labels_outpath()), 
            os.path.exists(self.spatial_labels_outpath()), 
            os.path.exists(self.spatiotemporal_features_outpath()), 
        ]
        if not all(paths_exist):
            #   Load original data
            path = self.spatiotemporal_features_inpath()
            assert os.path.exists(path), "Missing data file \"%s\"" % (path)
            df = pd.read_csv(path, header=None)
            if self.debug:
                print(df)
            #   Convert temporal labels
            temporal_labels = util.generate_temporal_labels(
                dt.datetime(2012, 1, 1),
                dt.datetime(2015, 1, 1),
                dt.timedelta(hours=1),
                bound_inclusion=[True, False]
            )
            temporal_labels = np.array(temporal_labels)
            n_temporal = temporal_labels.shape[0]
            #   Convert spatial labels
            spatial_labels = np.array([str(i+1) for i in range(len(df.columns))])
            n_spatial = spatial_labels.shape[0]
            #   Convert spatiotemporal features
            spatmp = df.to_numpy()
            if self.debug:
                print("Temporal Labels =", temporal_labels.shape)
                print(temporal_labels)
                print("Spatial Labels =", spatial_labels.shape)
                print(spatial_labels)
                print("Spatiotemporal =", spatmp.shape)
                print(spatmp)
            df = pd.DataFrame({
                "date": np.repeat(temporal_labels, n_spatial), 
                "client": np.tile(spatial_labels, n_temporal), 
                "power_kWh": np.reshape(spatmp, -1), 
            })
            if self.debug:
                print(df.loc[df["date"] == temporal_labels[0]])
                print(df.loc[df["date"] == temporal_labels[1]])
                print(df.loc[df["date"] == temporal_labels[-2]])
                print(df.loc[df["date"] == temporal_labels[-1]])
            pd.DataFrame({"date": temporal_labels}).to_csv(self.temporal_labels_outpath(), index=False)
            pd.DataFrame({"client": spatial_labels}).to_csv(self.spatial_labels_outpath(), index=False)
            df.to_csv(self.spatiotemporal_features_outpath(), index=False, compression="gzip")


class Traffic(Integrator):

    def spatiotemporal_features_inpath(self):
        return os.sep.join([self.acquire_dir(), "traffic.txt.gz"])

    def acquire(self, args):
        if not os.path.exists(self.spatiotemporal_features_inpath()):
            print("Download \"%s\" from sub-directory \"%s\" of repository at \"%s\" and save to acquisition directory \"%s\"" % (
                os.path.basename(self.spatiotemporal_features_inpath()), 
                "./traffic", 
                "https://github.com/laiguokun/multivariate-time-series-data", 
                self.acquire_dir(), 
            ))

    def convert(self, args):
        # Convert Traffic
        paths_exist = [
            os.path.exists(self.temporal_labels_outpath()), 
            os.path.exists(self.spatial_labels_outpath()), 
            os.path.exists(self.spatiotemporal_features_outpath()), 
        ]
        if not all(paths_exist):
            #   Load original data
            path = self.spatiotemporal_features_inpath()
            assert os.path.exists(path), "Missing data file \"%s\"" % (path)
            df = pd.read_csv(path, header=None)
            if self.debug:
                print(df)
            #   Convert temporal labels
            temporal_labels = util.generate_temporal_labels(
                dt.datetime(2015, 1, 1),
                dt.datetime(2017, 1, 1),
                dt.timedelta(hours=1),
                bound_inclusion=[True, False]
            )
            temporal_labels = np.array(temporal_labels)
            n_temporal = temporal_labels.shape[0]
            #   Convert spatial labels
            spatial_labels = np.array([str(i+1) for i in range(len(df.columns))])
            n_spatial = spatial_labels.shape[0]
            #   Convert spatiotemporal features
            spatmp = df.to_numpy()
            if self.debug:
                print("Temporal Labels =", temporal_labels.shape)
                print(temporal_labels)
                print("Spatial Labels =", spatial_labels.shape)
                print(spatial_labels)
                print("Spatiotemporal =", spatmp.shape)
                print(spatmp)
            df = pd.DataFrame({
                "date": np.repeat(temporal_labels, n_spatial), 
                "sensor": np.tile(spatial_labels, n_temporal), 
                "occupancy": np.reshape(spatmp, -1), 
            })
            if self.debug:
                print(df.loc[df["date"] == temporal_labels[0]])
                print(df.loc[df["date"] == temporal_labels[1]])
                print(df.loc[df["date"] == temporal_labels[-2]])
                print(df.loc[df["date"] == temporal_labels[-1]])
            pd.DataFrame({"date": temporal_labels}).to_csv(self.temporal_labels_outpath(), index=False)
            pd.DataFrame({"sensor": spatial_labels}).to_csv(self.spatial_labels_outpath(), index=False)
            df.to_csv(self.spatiotemporal_features_outpath(), index=False, compression="gzip")


class Solar_Energy(Integrator):

    def name(self):
        return "Solar-Energy"

    def spatiotemporal_features_inpath(self):
        return os.sep.join([self.acquire_dir(), "solar_AL.txt.gz"])

    def acquire(self, args):
        if not os.path.exists(self.spatiotemporal_features_inpath()):
            print("Download \"%s\" from sub-directory \"%s\" of repository at \"%s\" and save to acquisition directory \"%s\"" % (
                os.path.basename(self.spatiotemporal_features_inpath()), 
                "./solar-energy", 
                "https://github.com/laiguokun/multivariate-time-series-data", 
                self.acquire_dir(), 
            ))

    def convert(self, args):
        # Convert Solar-Energy
        paths_exist = [
            os.path.exists(self.temporal_labels_outpath()), 
            os.path.exists(self.spatial_labels_outpath()), 
            os.path.exists(self.spatiotemporal_features_outpath()), 
        ]
        if not all(paths_exist):
            #   Load original data
            path = self.spatiotemporal_features_inpath()
            assert os.path.exists(path), "Missing data file \"%s\"" % (path)
            df = pd.read_csv(path, header=None)
            if self.debug:
                print(df)
            #   Convert temporal labels
            temporal_labels = util.generate_temporal_labels(
                dt.datetime(2006, 1, 1),
                dt.datetime(2007, 1, 1),
                dt.timedelta(minutes=10),
                bound_inclusion=[True, False]
            )
            temporal_labels = np.array(temporal_labels)
            n_temporal = temporal_labels.shape[0]
            #   Convert spatial labels
            spatial_labels = np.array([str(i+1) for i in range(len(df.columns))])
            n_spatial = spatial_labels.shape[0]
            #   Convert spatiotemporal features
            spatmp = df.to_numpy()
            if self.debug:
                print("Temporal Labels =", temporal_labels.shape)
                print(temporal_labels)
                print("Spatial Labels =", spatial_labels.shape)
                print(spatial_labels)
                print("Spatiotemporal =", spatmp.shape)
                print(spatmp)
            df = pd.DataFrame({
                "date": np.repeat(temporal_labels, n_spatial), 
                "plant": np.tile(spatial_labels, n_temporal), 
                "power_MW": np.reshape(spatmp, -1), 
            })
            if self.debug:
                print(df.loc[df["date"] == temporal_labels[0]])
                print(df.loc[df["date"] == temporal_labels[1]])
                print(df.loc[df["date"] == temporal_labels[-2]])
                print(df.loc[df["date"] == temporal_labels[-1]])
            pd.DataFrame({"date": temporal_labels}).to_csv(self.temporal_labels_outpath(), index=False)
            pd.DataFrame({"plant": spatial_labels}).to_csv(self.spatial_labels_outpath(), index=False)
            df.to_csv(self.spatiotemporal_features_outpath(), index=False, compression="gzip")


class Exchange_Rate(Integrator):

    def name(self):
        return "Exchange-Rate"

    def spatiotemporal_features_inpath(self):
        return os.sep.join([self.acquire_dir(), "exchange_rate.txt.gz"])

    def acquire(self, args):
        if not os.path.exists(self.spatiotemporal_features_inpath()):
            print("Download \"%s\" from sub-directory \"%s\" of repository at \"%s\" and save to acquisition directory \"%s\"" % (
                os.path.basename(self.spatiotemporal_features_inpath()), 
                "./exchange_rate", 
                "https://github.com/laiguokun/multivariate-time-series-data", 
                self.acquire_dir(), 
            ))

    def convert(self, args):
        # Convert Solar-Energy
        paths_exist = [
            os.path.exists(self.temporal_labels_outpath()), 
            os.path.exists(self.spatial_labels_outpath()), 
            os.path.exists(self.spatiotemporal_features_outpath()), 
        ]
        if not all(paths_exist):
            #   Load original data
            path = self.spatiotemporal_features_inpath()
            assert os.path.exists(path), "Missing data file \"%s\"" % (path)
            df = pd.read_csv(path, header=None)
            if self.debug:
                print(df)
            #   Convert temporal labels
            temporal_labels = util.generate_temporal_labels(
                dt.datetime(1990, 1, 1),
                dt.datetime(2010, 10, 11),
                dt.timedelta(days=1),
                bound_inclusion=[True, False]
            )
            temporal_labels = np.array(temporal_labels)
            n_temporal = temporal_labels.shape[0]
            #   Convert spatial labels
            spatial_labels = np.array(
                [
                    "Australia", 
                    "British", 
                    "Canada", 
                    "Switzerland", 
                    "China", 
                    "Japan", 
                    "New_Zealand", 
                    "Singapore"
                ]
            )
            n_spatial = spatial_labels.shape[0]
            #   Convert spatiotemporal features
            spatmp = df.to_numpy()
            if self.debug:
                print("Temporal Labels =", temporal_labels.shape)
                print(temporal_labels)
                print("Spatial Labels =", spatial_labels.shape)
                print(spatial_labels)
                print("Spatiotemporal =", spatmp.shape)
                print(spatmp)
            df = pd.DataFrame({
                "date": np.repeat(temporal_labels, n_spatial), 
                "country": np.tile(spatial_labels, n_temporal), 
                "exchange_rate": np.reshape(spatmp, -1), 
            })
            if self.debug:
                print(df.loc[df["date"] == temporal_labels[0]])
                print(df.loc[df["date"] == temporal_labels[1]])
                print(df.loc[df["date"] == temporal_labels[-2]])
                print(df.loc[df["date"] == temporal_labels[-1]])
            pd.DataFrame({"date": temporal_labels}).to_csv(self.temporal_labels_outpath(), index=False)
            pd.DataFrame({"country": spatial_labels}).to_csv(self.spatial_labels_outpath(), index=False)
            df.to_csv(self.spatiotemporal_features_outpath(), index=False, compression="gzip")


class METR_LA(Integrator):

    def name(self):
        return "METR-LA"

    def spatiotemporal_features_inpath(self):
        return os.sep.join([self.acquire_dir(), "metr-la.h5"])

    def spatial_features_inpath(self):
        return os.sep.join([self.acquire_dir(), "graph_sensor_locations.csv"])

    def graph_features_inpath(self):
        return os.sep.join([self.acquire_dir(), "adj_mx.pkl"])

    def hdf5_key(self):
        return "df"

    def acquire(self, args):
        paths_exist = [
            os.path.exists(self.spatiotemporal_features_inpath()), 
            os.path.exists(self.spatial_features_inpath()), 
            os.path.exists(self.graph_features_inpath()), 
        ]
        if not all(paths_exist):
            print(
                "Download \"%s\" from \"%s\" and save to acquisition directory \"%s\"" % (
                    os.path.basename(self.spatiotemporal_features_inpath()), 
                    "https://drive.google.com/drive/folders/10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX", 
                    self.acquire_dir(), 
                )
            )
            print(
                "Download \"%s\" and \"%s\" from sub-directory \"%s\" of repository at \"%s\" and save to acquisition directory \"%s\"" % (
                    os.path.basename(self.spatial_features_inpath()), 
                    os.path.basename(self.graph_features_inpath()), 
                    "./data/sensor_graph/", 
                    "https://github.com/liyaguang/DCRNN", 
                    self.acquire_dir(), 
                )
            )

    def convert(self, args):
        import h5py
        import numpy as np
        paths_exist = [
            os.path.exists(self.temporal_labels_outpath()), 
            os.path.exists(self.spatial_labels_outpath()), 
            os.path.exists(self.spatiotemporal_features_outpath()), 
        ]
        # Convert spatiotemporal features and labels
        if not all(paths_exist):
            f = h5py.File(self.spatiotemporal_features_inpath())
            spatial_labels = np.array(
                [
                    (str(label) if isinstance(label, np.int64) else label.decode("utf-8")) for label in f[self.hdf5_key()]["axis0"]
                ]
            )
            n_spatial = spatial_labels.shape[0]
            features = np.array(list(f[self.hdf5_key()]["block0_values"]))
            n_temporal, n_spatial = features.shape
            temporal_labels = np.array(
                util.generate_temporal_labels("2012-03-01_00-00-00", n_temporal, [5, "minutes"])
            )
            data = {
                "date": np.repeat(temporal_labels, n_spatial), 
                "sensor": np.tile(spatial_labels, n_temporal), 
                "speedmph": np.reshape(features, -1)
            }
            df = pd.DataFrame(data)
            if args.debug:
                for i in [0, 1, 2, 3, 4, -5, -4, -3, -2, -1]:
                    print(temporal_labels[i], spatial_labels[0], features[i,0])
                print(df.iloc[np.arange(0, len(df), n_spatial)])
            pd.DataFrame({"sensor": spatial_labels}).to_csv(self.spatial_labels_outpath(), index=False)
            pd.DataFrame({"date": temporal_labels}).to_csv(self.temporal_labels_outpath(), index=False)
            df.to_csv(self.spatiotemporal_features_outpath(), index=False, compression="gzip")
        #   Convert spatial features
        if not os.path.exists(self.spatial_features_outpath()):
            spatial_features_fname = os.path.basename(self.spatial_features_inpath())
            path = self.spatial_features_inpath()
            assert os.path.exists(path), "Missing data file \"%s\"" % (path)
            df = self.load_spatial_df(path)
            spatial_labels = pd.read_csv(self.spatial_labels_outpath()).to_numpy(dtype=str).reshape(-1)
            _spatial_labels = df["sensor"].to_numpy(dtype=str)
            print(spatial_labels)
            print(_spatial_labels)
            dat_sel = DataSelection()
            indices = dat_sel.indices_from_selection(_spatial_labels, ["literal"] + list(spatial_labels))
            df = df.iloc[indices,:]
            if self.debug:
                print(df)
            df.to_csv(self.spatial_features_outpath(), index=False)
        #   Convert graph topology
        if not os.path.exists(self.graph_features_outpath()):
            from Data.GraphData import A_to_edgelist, W_to_edgelist
            graph_features_fname = os.path.basename(self.graph_features_inpath())
            path = self.graph_features_inpath()
            assert os.path.exists(path), "Missing data file \"%s\"" % (path)
            in_path, out_path = path, os.sep.join([
                self.cache_dir(), 
                graph_features_fname.replace(".pkl", "_fixed.pkl")
            ])
            self.fix_file_content(in_path, out_path)
            path = out_path
            data = util.from_cache(path, encoding="bytes")
            spatial_labels = np.array([lab.decode("utf-8") for lab in data[0]])
            spatial_labels = pd.read_csv(self.spatial_labels_outpath()).to_numpy(dtype=str).reshape(-1)
            label_index_map = util.to_dict([lab.decode("utf-8") for lab in data[1].keys()], data[1].values())
            adj = data[2]
            indices = np.array(util.get_dict_values(label_index_map, spatial_labels))
            if self.debug:
                print("LABEL INDEX MAP")
                print(label_index_map)
                print("SPATIAL LABELS")
                print(spatial_labels)
                print("SPATIAL LABEL INDICES")
                print(indices)
                print("ADJ")
                print(adj)
            dat_sel = DataSelection()
            adj = dat_sel.filter_axis(adj, [0, 1], [indices, indices])
            edgelist = W_to_edgelist(adj, spatial_labels)
            df = pd.DataFrame(
                {
                    "source": [edge[0] for edge in edgelist], 
                    "destination": [edge[1] for edge in edgelist], 
                    "weight": [edge[2] for edge in edgelist], 
                }
            )
            if self.debug:
                print(df)
            df.to_csv(self.graph_features_outpath(), index=False)

    def fix_file_content(self, in_path, out_path):
        content = ""
        with open(in_path, "rb") as in_file:
            content = in_file.read()
        out_size = 0
        with open(out_path, "wb") as out_file:
            for line in content.splitlines():
                out_size += len(line) + 1
                out_file.write(line + str.encode("\n"))

    def load_spatial_df(self, path):
        df = pd.read_csv(path)
        df = df.drop("index", axis=1)
        df = df.rename({"sensor_id": "sensor"}, axis=1)
        return df


class PEMS_BAY(METR_LA):

    def name(self):
        return "PEMS-BAY"

    def spatiotemporal_features_inpath(self):
        return os.sep.join([self.acquire_dir(), "pems-bay.h5"])

    def spatial_features_inpath(self):
        return os.sep.join([self.acquire_dir(), "graph_sensor_locations_bay.csv"])

    def graph_features_inpath(self):
        return os.sep.join([self.acquire_dir(), "adj_mx_bay.pkl"])

    def hdf5_key(self):
        return "speed"

    def load_spatial_df(self, path):
        return pd.read_csv(path, names=["sensor", "latitude", "longitude"])


class NEW_METR_LA(Integrator):

    def name(self):
        return "NEW-METR-LA"

    def download_pems(self, district, ftype, years_and_months, download_dir, username, password):
        from caltrans_pems.pems.handler import PeMSHandler
        ftype_fname_map = {
            "station_5min": "d%02d_text_%s_%d_%02d_%02d.txt.gz", 
            "meta": "d%02d_text_%s_%d_%02d_%02d.txt", 
        }
        fname_template = ftype_fname_map[ftype]
        # Start
        done_once = False
        for year, month in years_and_months:
            fnames = []
            for day in range(1, util.days_in_month(year, month)+1):
                fnames.append(fname_template % (district, ftype, year, month, day))
            paths = [os.sep.join([download_dir, fname]) for fname in fnames]
            if not all([os.path.exists(path) for path in paths]):
                if not done_once:
                    pems = PeMSHandler(username=username, password=password)
                    done_once = True
                pems.download_files(
                    year, 
                    year, 
                    ["%d" % (district)], 
                    [ftype], 
                    [util.month_name_map[month]], 
                    save_path=download_dir
                )

    def integrator(self):
        return METR_LA()

    def district(self):
        return 7

    def years_and_months(self):
        return ((2012, 3), (2012, 4), (2012, 5), (2012, 6))

    def var(self):
        return Container().set(
            [
                "replace_existing", 
                "file_range", 
                "debug", 
            ], 
            [
                True, 
                [0, -4], 
                0, 
            ], 
            multi_value=True
        )

    def acquire(self, args):
        self.download_pems(
            self.district(), 
            "station_5min", 
            self.years_and_months(), 
            self.acquire_dir(), 
            args.username, 
            args.password
        )
        try:
            self.download_pems(
                self.district(), 
                "meta", 
                self.years_and_months(), 
                self.acquire_dir(), 
                args.username, 
                args.password
            )
        except:
            pass

    def convert(self, args):
        var = self.var().merge(args)
        # Start
        pd.options.mode.chained_assignment = None
        import warnings
        warnings.simplefilter(action='ignore', category=FutureWarning)
        if 0:
            df_i = pd.read_csv(self.integrator().spatiotemporal_features_outpath())
            if var.debug:
                print(df_i)
            if var.debug:
                missing = get_missing(df_i, "sensor")
                print(missing)
            spatial_labels = df_i["sensor"].astype(str).unique()
            temporal_labels = df_i["date"].astype(str).unique()
            print(spatial_labels, len(spatial_labels))
            print(temporal_labels, len(temporal_labels))
            quit()
        if not os.path.exists(self.spatiotemporal_features_outpath()):
            df_i = pd.read_csv(self.integrator().spatiotemporal_features_outpath())
            df_i["sensor"] = df_i["sensor"].astype(str)
            df_i["date"] = df_i["date"].astype(str)
            if var.debug:
                print(df_i)
            if var.debug:
                missing = get_missing(df_i, "sensor")
                print(missing)
            spatial_labels = df_i["sensor"].astype(str).unique()
            temporal_labels = df_i["date"].astype(str).unique()
            n_spatial = len(spatial_labels)
            n_cols = 52
            cols = [
                "Timestamp", 
                "Station", 
                "District", 
                "Freeway", 
                "Direction", 
                "Lane_Type", 
                "Station_Length", 
                "Samples", 
                "Percent_Observed", 
                "Total_Flow", 
                "Avg_Occupancy", 
                "Avg_Speed", 
            ]
            for i in range(1, (n_cols - len(cols)) // 5 + 1):
                cols += [
                    "Lane_%d_Samples" % (i), 
                    "Lane_%d_Flow" % (i), 
                    "Lane_%d_Avg_Occupancy" % (i), 
                    "Lane_%d_Avg_Speed" % (i), 
                    "Lane_%d_Observed" % (i), 
                ]
            kept_cols = [
                "Timestamp", 
                "Station", 
                "Samples", 
                "Percent_Observed", 
                "Total_Flow", 
                "Avg_Occupancy", 
                "Avg_Speed", 
            ]
            # Start joining df_i and df_j into df_k
            n_temporal = 288
            paths = sorted(glob.glob(os.sep.join([self.acquire_dir(), "*station_5min*"])))
            if var.file_range[0] < 0:
                var.file_range[0] = len(paths) + var.file_range[0] + 1
            if var.file_range[1] < 0:
                var.file_range[1] = len(paths) + var.file_range[1] + 1
            paths = paths[var.file_range[0]:var.file_range[1]]
            dfs = []
            for i, path in zip(range(var.file_range[0], var.file_range[1]), paths):
                print("%d/%d" % (i-var.file_range[0]+1, len(paths)))
                start, end = n_temporal * n_spatial * i, n_temporal * n_spatial * (i+1)
                df_j = pd.read_csv(path, names=cols, usecols=kept_cols)
                df_j["Timestamp"] = pd.to_datetime(df_j["Timestamp"]).dt.strftime("%Y-%m-%d_%H-%M-%S")
                df_j["Station"] = df_j["Station"].astype(str) 
                if var.debug:
                    print("df_j =")
                    print(df_j)
                df_k = pd.DataFrame(columns=df_j.columns, index=range(n_temporal*n_spatial))
                df_k[["Timestamp", "Station", "Avg_Speed"]] = df_i.loc[start:end-1,["date", "sensor", "speedmph"]].reset_index(drop=True)
                if var.debug:
                    print(df_k)
                _temporal_labels = df_j["Timestamp"].unique()
                _n_temporal = len(_temporal_labels)
                if True or _n_temporal != n_temporal:
                    if var.debug:
                        print("unexpected time-steps in with n_temporal=%s" % (_n_temporal))
                    for idx, temporal_label in enumerate(temporal_labels[n_temporal*i:n_temporal*(i+1)]):
                        if temporal_label in _temporal_labels:
                            tmp_j = df_j.loc[df_j["Timestamp"] == temporal_label]
                            if var.debug:
                                print("tmp_j =")
                                print(tmp_j)
                            start, end = n_spatial * idx, n_spatial * (idx + 1)
                            tmp_k = df_k.iloc[start:end]
                            if var.debug:
                                print("tmp_k =")
                                print(tmp_k)
                            tmp_k.loc[tmp_k["Station"].isin(tmp_j["Station"])] = tmp_j.loc[tmp_j["Station"].isin(tmp_k["Station"])].set_index(
                                tmp_k.loc[tmp_k["Station"].isin(tmp_j["Station"])].index
                            )
                            if var.debug:
                                print("tmp_k =")
                                print(tmp_k)
                            df_k.iloc[start:end] = tmp_k
                            if var.debug:
                                print("tmp_k =")
                                print(df_k.iloc[start:end])
                                input()
                else:
                    mask = df_k["Station"].isin(df_j["Station"])
                    df_k.loc[mask] = df_j.loc[df_j["Station"].isin(spatial_labels)].set_index(mask[mask].index)
                if not var.replace_existing:
                    df_k[["Avg_Speed"]] = df_i.loc[start:end-1,["speedmph"]].reset_index(drop=True)
                if var.debug:
                    print("df_k =")
                    print(df_k)
                if var.debug:
                    missing = get_missing(df_k, "Station")
                    print("missing =")
                    print(missing)
                dfs.append(df_k)
                if var.debug:
                    input()
            df = pd.concat(dfs)
            df.to_csv(self.spatiotemporal_features_outpath(), index=False, compression="gzip")
        if not os.path.exists(self.spatial_features_outpath()):
            df_i = pd.read_csv(self.integrator().spatial_features_outpath())
            cols = ["ID", "Fwy", "Dir", "District", "County", "City", "State_PM", "Abs_PM", "Latitude", "Longitude", "Length", "Type", "Lanes", "Name", "User_ID_1", "User_ID_2", "User_ID_3", "User_ID_4"]
            paths = sorted(glob.glob(os.sep.join([self.acquire_dir(), "*meta*"])))
            df_j = pd.read_csv(paths[-1], usecols=cols[:-4], sep="\t")
            df_j.columns = ["Station", "Freeway", "Direction"] + list(df_j.columns)[3:]
            if var.debug:
                print(df_j)
            df_k = pd.DataFrame(columns=df_j.columns, index=df_i.index)
            df_k[["Station", "Latitude", "Longitude"]] = df_i[["sensor", "latitude", "longitude"]]
            if var.debug:
                print(df_k)
            df_k.loc[df_k["Station"].isin(df_j["Station"])] = df_j.loc[df_j["Station"].isin(df_k["Station"])].set_index(
                df_k.loc[df_k["Station"].isin(df_j["Station"])].index
            )
            if not var.replace_existing:
                df_k[["Latitude", "Longitude"]] = df_i[["latitude", "longitude"]]
            if var.debug:
                print(df_k)
            if var.debug:
                print(get_missing(df_k, "Station"))
            df_k.to_csv(self.spatial_features_outpath(), index=False)
        df = pd.read_csv(self.integrator().spatial_labels_outpath())
        df.columns = ["Station"]
        df.to_csv(self.spatial_labels_outpath(), index=False)
        df = pd.read_csv(self.integrator().temporal_labels_outpath())
        df.columns = ["Timestamp"]
        df.to_csv(self.temporal_labels_outpath(), index=False)
        if os.path.exists(self.integrator().graph_features_outpath()) and not os.path.exists(self.graph_features_outpath()):
            shutil.copy(self.integrator().graph_features_outpath(), self.graph_features_outpath())



class NEW_PEMS_BAY(NEW_METR_LA):

    def name(self):
        return "NEW-PEMS-BAY"

    def integrator(self):
        return PEMS_BAY()

    def district(self):
        return 4

    def years_and_months(self):
        return ((2017, 1), (2017, 2), (2017, 3), (2017, 4), (2017, 5), (2017, 6))

    def var(self):
        return Container().set(
            [
                "replace_existing", 
                "file_range", 
                "debug", 
            ], 
            [
                True, 
                [0, -1], 
                0, 
            ], 
            multi_value=True
        )


class CaltransPeMS(Integrator):

    # Notes:
    #   It turns out that the files from CaltransPeMS do not always contain the same number of time-steps nor
    #       do they always contain the same number of stations/nodes.
    #   As such, the conversion from a list of files into a single Spatiotemporal.csv file cannot be implmented
    #       trivially with a simple concatenation.

    def __init__(self):
        from selenium import webdriver
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.common.exceptions import TimeoutException
        self.districts = [3, 4, 5, 6, 7, 8, 10, 11, 12]
#        self.districts = [5]
        self.filters = {"years": [2021, 2021]}
        self.convert_spatial_labels, self.convert_temporal_labels = True, True
        self.convert_spatial, self.convert_spatiotemporal = True, True
        self.convert_spatial, self.convert_spatiotemporal = True, False
        self.n_spatial_limit = 3000
        self.debug = 2
#        self.debug = 0
        os.makedirs(self.root_dir(), exist_ok=True)
        os.makedirs(self.cache_dir(), exist_ok=True)
        os.makedirs(self.acquire_dir(), exist_ok=True)
        os.makedirs(self.convert_dir(), exist_ok=True)

    class PeMS_Hyperlink:

        def __init__(self, href):
            href_fields = href.replace("javascript:processFiles(", "").replace(");", "").split(",")
            self.district = href_fields[0].strip()
            self.misc = href_fields[1].strip()
            self.year = href_fields[2].strip()
            self.type = href_fields[3].strip()
            self.format = href_fields[4].strip()
            self.month_idx = href_fields[5].strip()

        def __str__(self):
            lines = []
            for key, value in self.__dict__.items():
                lines.append("%s : %s" % (str(key), str(value)))
            return "\n".join(lines)

    class PeMS_Filename:

        def __init__(self, fname):
            fields = fname.split("_")
            self.district = int(fields[0].replace("d", ""))
            self.format = fields[1]
            self.type = fields[2]
            days, hours, minutes = 0, 0, 0
            if fields[3].endswith("day"):
                n = fields[3].replace("day", "")
                days = int("0" if n == "" else n)
            if fields[3].endswith("hour"):
                n = fields[3].replace("hour", "")
                hours = int("0" if n == "" else n)
            if fields[3].endswith("min"):
                n = fields[3].replace("min", "")
                minutes = int("0" if n == "" else n)
            else:
                raise NotImplementedError("Cannot convert \"%s\" to a temporal resolution" % (fields[3]))
            self.resolution = dt.timedelta(days=days, hours=hours, minutes=minutes)
            self.year = int(fields[4])
            self.month = int(fields[5])
            self.day = int(fields[-1].split(".")[0])
            self.ext = ".".join(fields[-1].split(".")[1:])

        def __str__(self):
            lines = []
            for key, value in self.__dict__.items():
                lines.append("%s : %s" % (str(key), str(value)))
            return "\n".join(lines)
            
    def scrape_pems_data_urls(self, data_type="station_5min", district=3, username=None, password=None):
        url = "https://pems.dot.ca.gov/?dnode=Clearinghouse"
        url += "&type=%s" % (str(data_type))
        url += "&district_id=%s" % (str(district))
        url += "&submit=Submit"
        _url, _type, _district, _opt = url.split("&")
        _type, _district = _type.split("=")[-1], _district.split("=")[-1]
        driver = webdriver.Chrome()
        driver.implicitly_wait(10)
        driver.get(url)
        if not username is None:
            driver.find_element(By.NAME, "username").send_keys(username)
        if not password is None:
            driver.find_element(By.NAME, "password").send_keys(password)
        if not username is None and not password is None:
            driver.find_element(By.NAME, "login").click()
        try:
            WebDriverWait(driver, 60).until(EC.presence_of_element_located((By.CLASS_NAME, "dataFilesScrollBox")))
        except TimeoutException:
            print("Timed out waiting for \"dataFilesScrollBox\" to load")
            return []
        links = driver.find_element(By.CLASS_NAME, "dbxWidget").find_elements(By.TAG_NAME, "a")
        year_links, tmp = [], {}
        for link in links:
            href = link.get_attribute("href")
            pems_hlink = PeMS_Hyperlink(href)
            if not pems_hlink.year in tmp: # Keep only one link per year since each contains entire year of data
                tmp[pems_hlink.year] = None
                year_links.append(link)
        fnames, urls = [], []
        for year_link in year_links[:]:
            href = year_link.get_attribute("href")
            pems_hlink = PeMS_Hyperlink(href)
            print(33*"#")
            print(pems_hlink)
            print(33*"#")
            year_link.click()
            if 0:
                try:
                    WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.ID, "datafiles")))
                except TimeoutException:
                    print("Timed out waiting for \"datafiles\" to load")
                    return []
            else:
                time.sleep(2) # Wait for page to update and load "datafiles"
            data_links = driver.find_element(By.ID, "datafiles").find_elements(By.TAG_NAME, "a")
            print("Found %d Data Links:" % (len(data_links)))
            for data_link in data_links:
                fnames.append(data_link.get_attribute("text"))
                urls.append(data_link.get_attribute("href"))
                print(fnames[-1], "|||", urls[-1])
        return fnames, urls

    def load_pems_urls(self, data_type, districts, username=None, password=None, cache_dir="Downloads"):
        fnames, urls = [], []
        for district in districts:
            fnames_path = os.sep.join([cache_dir, "PeMS_Type[%s]_District[%s]_Data[%s].txt" % (
                str(data_type), 
                str(district), 
                "filenames", 
            )]) 
            urls_path = os.sep.join([cache_dir, "PeMS_Type[%s]_District[%s]_Data[%s].txt" % (
                str(data_type),
                str(district), 
                "urls", 
            )]) 
#            print(fnames_path)
#            print(urls_path)
            if not (os.path.exists(fnames_path) and os.path.exists(urls_path)):
                print(fnames_path)
                print(urls_path)
                print("Fetching PeMS filenames and urls for type=\"%s\" and district=\"%s\"." % (
                    data_type, 
                    district, 
                ))
                _fnames, _urls = self.scrape_pems_data_urls(data_type, district, username, password)
                with open(fnames_path, "w") as f:
                    f.write("\n".join(_fnames))
                with open(urls_path, "w") as f:
                    f.write("\n".join(_urls))
            else:
                print("PeMS filenames and urls already cached for type=\"%s\" and district=\"%s\". Moving on..." % (
                    data_type, 
                    district, 
                ))
            with open(fnames_path, "r") as f:
                _fnames = f.read().split("\n")
            with open(urls_path, "r") as f:
                _urls = f.read().split("\n")
            fnames += _fnames
            urls += _urls
        return fnames, urls


    # Download files specified by a set of urls and move them to a cache directory
    #   fnames: name of the downloaded file at each url
    #   urls: location of each file to download
    #   dwnld_dir: location where files get downloaded
    #   cache_dir: location where files are to be moved
    #   replace_existing: check if file exists in cache_dir, (true) download even if it exists, (false) move on
    def download_pems_data(self, fnames, urls, dwnld_dir, cache_dir, replace_existing=False):
        def batch_to_cache(dwnld_dir, cache_dir, batch_size):
            # Check number of completed downloads, then move to cache directory if a batch is ready
            dwnld_paths = glob.glob(os.sep.join([dwnld_dir, "*.txt.gz"]))
            if len(dwnld_paths) >= batch_size:
                for dwnld_path in dwnld_paths:
                    cache_path = os.sep.join([cache_dir, os.path.basename(dwnld_path)])
                    shutil.move(dwnld_path, cache_path)
        assert len(fnames) == len(urls), "Non-equal number of filenames and urls."
        os.makedirs(cache_dir, exist_ok=True)
        # You MUST be logged into Caltrans PeMS in order to access these downloads
        base_wait, wait_coef, batch_size = 1/2, 4/8, 5
        for fname, url in zip(fnames, urls):
            cache_path = os.sep.join([cache_dir, fname])
            if replace_existing or not os.path.exists(cache_path):
                # Download and wait according to number of queued downloads
                queued_paths = glob.glob(os.sep.join([dwnld_dir, "*.crdownload"]))
                wait = wait_coef * len(queued_paths)**2 + base_wait
                webbrowser.open(url, new=2, autoraise=False)
                time.sleep(wait)
                batch_to_cache(dwnld_dir, cache_dir, batch_size)
        queued_paths = glob.glob(os.sep.join([dwnld_dir, "*.crdownload"]))
        while len(queued_paths) > 0:
            time.sleep(1)
            queued_paths = glob.glob(os.sep.join([dwnld_dir, "*.crdownload"]))
        batch_to_cache(dwnld_dir, cache_dir, 1)


    def filter_fnames_urls(self, fnames, urls, filters={}):
        district_interval = [-sys.float_info.max, sys.float_info.max]
        year_interval = [-sys.float_info.max, sys.float_info.max]
        month_interval = [-sys.float_info.max, sys.float_info.max]
        day_interval = [-sys.float_info.max, sys.float_info.max]
        if "districts" in filters:
            district_interval = filters["districts"]
        if "years" in filters:
            year_interval = filters["years"]
        if "months" in filters:
            month_interval = filters["months"]
        if "days" in filters:
            day_interval = filters["days"]
        _fnames, _urls = [], []
        for fname, url in zip(fnames, urls):
            pems_fname = self.PeMS_Filename(fname)
            keep = True
            keep = keep and (pems_fname.district >= district_interval[0] and pems_fname.district <= district_interval[1])
            keep = keep and (pems_fname.year >= year_interval[0] and pems_fname.year <= year_interval[1])
            keep = keep and (pems_fname.month >= month_interval[0] and pems_fname.month <= month_interval[1])
            keep = keep and (pems_fname.day >= day_interval[0] and pems_fname.day <= day_interval[1])
            if keep:
                _fnames.append(fname)
                _urls.append(url)
        return _fnames, _urls

    def acquire(self, args):
        cache_dir = self.cache_dir()
        acquire_dir = self.acquire_dir()
        dwnld_dir = download_dir()
        if not ("username" in args or "password" in args):
            raise ValueError("Need a username and password to access Caltrans PeMS")
        # Get filenames and urls of files to be downloaded
        fnames, urls = self.load_pems_urls("station_5min", self.districts, args.username, args.password, cache_dir)
        # Apply filters to filenames and urls for downloading
        fnames, urls = self.filter_fnames_urls(fnames, urls, self.filters)
        # Download the filtered files for selected districts
        for district in self.districts:
            filters = {"districts": [district, district]}
            _fnames, _urls = self.filter_fnames_urls(fnames, urls, filters)
            for i in [0,-1]:
                print(_fnames[i], "|||", _urls[i])
            _acquire_dir = os.sep.join([acquire_dir, "d%02d" % (district)])
            self.download_pems_data(_fnames[:], _urls[:], dwnld_dir, _acquire_dir)
            # Unpack the downloaded files which should be *.txt.gz
            if 0: # Not needed since pandas can load from .gz files
                for fname in fnames[:]:
                    zip_path = os.sep.join([_cache_dir, fname])
                    unzip_path = zip_path.replace(".gz", "")
                    if os.path.exists(unzip_path):
                        print("File %s already unzipped. Moving on..." % (zip_path))
                        continue
                    with gzip.open(zip_path, "rb") as f_in:
                        with open(unzip_path, "wb") as f_out:
                            shutil.copyfileobj(f_in, f_out)

    def convert(self, args):
        cache_dir = self.cache_dir()
        acquire_dir = self.acquire_dir()
        convert_dir = self.convert_dir()
        if not ("username" in args or "password" in args):
            raise ValueError("Need a username and password to access Caltrans PeMS")
        # Get filenames and urls of files to be downloaded
        fnames, urls = self.load_pems_urls("station_5min", self.districts, args.username, args.password, cache_dir)
        # Apply filters to filenames and urls for downloading
        fnames, urls = self.filter_fnames_urls(fnames, urls, self.filters)
        # Merge the downloaded files into one
        n_cols = 52
        cols = [
            "Timestamp", 
            "Station", 
            "District", 
            "Freeway", 
            "Direction", 
            "Lane_Type", 
            "Station_Length", 
            "Samples", 
            "Percent_Observed", 
            "Total_Flow", 
            "Avg_Occupancy", 
            "Avg_Speed", 
        ]
        for i in range(1, (n_cols - len(cols)) // 5 + 1):
            cols += [
                "Lane_%d_Samples" % (i), 
                "Lane_%d_Flow" % (i), 
                "Lane_%d_Avg_Occupancy" % (i), 
                "Lane_%d_Avg_Speed" % (i), 
                "Lane_%d_Observed" % (i), 
            ]
        kept_cols = [
            "Timestamp", 
            "Station", 
            "Samples", 
            "Percent_Observed", 
            "Total_Flow", 
            "Avg_Occupancy", 
            "Avg_Speed", 
        ]
        for district in self.districts[:]:
            if self.debug:
                print(util.make_msg_block("DISTRICT %02d" % (district)))
            # Setup directories
            _cache_dir = os.sep.join([cache_dir, "d%02d" % (district)])
            _acquire_dir = os.sep.join([acquire_dir, "d%02d" % (district)])
            _convert_dir = os.sep.join([convert_dir, "d%02d" % (district)])
            os.makedirs(_cache_dir, exist_ok=True)
            os.makedirs(_convert_dir, exist_ok=True)
            # Filter fnames and urls to consider only this district
            filters = {"districts": [district, district]}
            _fnames, _urls = self.filter_fnames_urls(fnames, urls, filters)
            ###########################################
            # Create spatial labels for this district #
            ###########################################
            if self.debug:
                print(util.make_msg_block("Creating Spatial Labels"))
            path = os.sep.join([_convert_dir, "SpatialLabels.csv"])
            if self.convert_spatial_labels and not os.path.exists(path):
                _path = os.sep.join([_acquire_dir, _fnames[0]])
                df = pd.read_csv(_path, names=cols, usecols=kept_cols, compression="gzip")
                df["Station"] = df["Station"].astype(str)
                spatial_labels = sorted(df["Station"].unique())
                df = pd.DataFrame({"Station": spatial_labels}, dtype=str)
                df.to_csv(path, index=False)
            df = pd.read_csv(path)
            spatial_labels = df["Station"].to_numpy().astype(str)
            if self.debug > 1:
                print("n_spatial =", len(spatial_labels))
#                print(spatial_labels)
            ############################################
            # Create temporal labels for this district #
            ############################################
            if self.debug:
                print(util.make_msg_block("Creating Temporal Labels"))
            path = os.sep.join([_convert_dir, "TemporalLabels.csv"])
            if self.convert_temporal_labels and not os.path.exists(path) or True:
                start, end = self.PeMS_Filename(_fnames[0]), self.PeMS_Filename(_fnames[-1])
                temporal_labels = util.generate_temporal_labels(
                    dt.datetime(start.year, start.month, start.day), 
                    dt.datetime(end.year, end.month, end.day) + dt.timedelta(days=1), 
                    start.resolution, 
                    bound_inclusion=[True, False], 
                )
                df = pd.DataFrame({"Timestamp": temporal_labels}, dtype=str)
                df.to_csv(path, index=False)
            df = pd.read_csv(path)
            temporal_labels = df["Timestamp"].to_numpy().astype(str)
            if self.debug > 1:
                print("n_temporal =", len(temporal_labels))
            #########################################
            # Create spatial data for this district #
            #########################################
            if self.debug:
                print(util.make_msg_block("Creating Spatial Data"))
            path = os.sep.join([_convert_dir, "Spatial.csv"])
            if self.convert_spatial and not os.path.exists(path):
                paths = glob.glob(os.sep.join([_acquire_dir, "*_text_meta_*.txt"]))
                if len(paths) < 1:
                    print("Did not find any meta data files for district d%02d" % (district))
                else:
                    df = pd.read_csv(paths[0], sep="\t")
                    col_repl_map = {
                        "ID": "Station", 
                        "Fwy": "Freeway", 
                        "Dir": "Direction", 
                    }
                    _cols = []
                    for col in df.columns:
                        if col in col_repl_map:
                            col = col_repl_map[col]
                        _cols.append(col)
                    df.columns = _cols
                    df = df.iloc[:,:-4] # Remove "User_ID_#" fields
                    _path = os.sep.join([self.acquire_dir(), "st06_ca_cou.txt"])
                    if os.path.exists(_path):
                        county_df = pd.read_csv(
                            _path, 
                            names=["State_Name", "State_ID", "County_ID", "County_Name", "Misc"]
                        )
                        county_name_map = util.to_dict(county_df["County_ID"], county_df["County_Name"])
                        county_names = []
                        for row in df.to_dict("records"):
                            county_id = row["County"]
                            county_name = ""
                            if county_id in county_name_map:
                                county_name = county_name_map[county_id]
                            county_names.append(county_name)
                        df["County"] = county_names
                    else:
                        print("Did not find county data file %s. Will not convert county indices to county names for spatial data." % (_path))
                    df.to_csv(path, index=False)
            df = pd.read_csv(path)
            _spatial_labels = sorted(df["Station"].to_numpy().astype(str))
            if self.debug > 1:
                print("n_spatial =", len(_spatial_labels))
#                print(_spatial_labels)
            if not np.array_equiv(spatial_labels, _spatial_labels): # Keep only common spatial elements
                if self.debug:
                    print("Merging Inconsistent Spatial Labels from Spatiotemporal and Spatial Data")
                spatial_labels = sorted(np.intersect1d(spatial_labels, _spatial_labels))
                # Save new spatial labels
                path = os.sep.join([_convert_dir, "SpatialLabels.csv"])
                df = pd.DataFrame({"Station": spatial_labels})
                df.to_csv(path, index=False)
                # Save new spatial data
                path = os.sep.join([_convert_dir, "Spatial.csv"])
                df = pd.read_csv(path)
                df["Station"] = df["Station"].astype(str)
                df = df.loc[df["Station"].isin(spatial_labels)]
                df.to_csv(path, index=False)
                if self.debug > 1:
                    print("n_common_spatial_labels", len(spatial_labels))
            ################################################
            # Create spatiotemporal data for this district #
            ################################################
            if self.debug:
                print(util.make_msg_block("Creating Spatiotemporal Data"))
            path = os.sep.join([_convert_dir, "Spatiotemporal.csv.gz"])
            n_spatial, n_temporal = len(spatial_labels), len(temporal_labels)
            if self.convert_spatiotemporal and not os.path.exists(path):
                if n_spatial > self.n_spatial_limit:
                    print("Number of spatial elements (%d) cannot fit into memory. Moving on..." % (n_spatial))
                    continue
                if self.debug:
                    print(util.make_msg_block("Creating Index Map", "."))
                    start_time = time.time()
                _path = os.sep.join([_cache_dir, "SpatiotemporalIndex.pkl"])
                if not os.path.exists(_path):
                    label_index_map = util.merge_dicts(
                        util.to_dict(temporal_labels, np.arange(n_temporal) * n_spatial), 
                        util.to_dict(spatial_labels, np.arange(n_spatial))
                    )
                    util.to_cache(label_index_map, _path)
                else:
                    label_index_map = util.from_cache(_path)
                if self.debug > 1:
                    indices = [0,1,n_temporal-2, n_temporal-1,n_temporal,n_temporal+1,-2,-1]
                    for label in np.array(list(label_index_map.keys()))[indices]:
                        print(label, label_index_map[label])
                if self.debug:
                    print(
                        util.make_msg_block("Creating Index Map %.2fs" % (time.time() - start_time), ".")
                    )
                if self.debug:
                    print(util.make_msg_block("Setup for Spatiotemporal Data Creation", "."))
                    start_time = time.time()
                start, end = self.PeMS_Filename(_fnames[0]), self.PeMS_Filename(_fnames[1])
                file_temporal_labels = util.generate_temporal_labels(
                    dt.datetime(start.year, start.month, start.day), 
                    dt.datetime(end.year, end.month, end.day), 
                    start.resolution, 
                    bound_inclusion=[True, False], 
                )
                file_n_temporal = len(file_temporal_labels)
                if 0:
                    df = pd.DataFrame(index=range(n_temporal * n_spatial), columns=kept_cols, dtype=np.float32)
                    df["Timestamp"] = np.repeat(temporal_labels, n_spatial)
                    df["Station"] = np.tile(spatial_labels, n_temporal)
                else:
                    df = pd.DataFrame(
                        itertools.product(temporal_labels, spatial_labels), 
                        columns=kept_cols[:2]
                    )
                    for kept_col in kept_cols[2:]:
                        df[kept_col] = pd.Series(dtype=np.float32)
                if self.debug:
                    print(util.make_msg_block("Setup for Spatiotemporal Data Creation %.2fs" % (
                        time.time() - start_time), ".")
                    )
                if self.debug:
                    print(util.make_msg_block("Collecting File Data", "."))
                    start_time = time.time()
                for fname in _fnames[:]:
                    _path = os.sep.join([_acquire_dir, fname])
                    if self.debug > 1:
                        print(fname)
                    _df = pd.read_csv(_path, names=cols, usecols=kept_cols, compression="gzip")
                    _df["Timestamp"] = pd.to_datetime(_df["Timestamp"])
                    _df["Timestamp"] = _df["Timestamp"].dt.strftime("%Y-%m-%d_%H-%M-%S")
                    _df["Station"] = _df["Station"].astype(str)
                    if self.debug > 2:
                        _temporal_labels = _df["Timestamp"].unique()
                        _spatial_labels = _df["Station"].unique()
                        print(
                            "df_n_temporal=%d, df_n_spatial=%d" % (
                                len(_temporal_labels), len(_spatial_labels)
                            )
                        )
                    df_dict = _df.to_dict("records")
                    indices, values = [], []
                    for row in df_dict:
                        temporal_label, spatial_label = row["Timestamp"], row["Station"]
                        if temporal_label in label_index_map and spatial_label in label_index_map:
                            index = label_index_map[temporal_label] + label_index_map[spatial_label]
                            indices.append(index)
                            values.append(util.get_dict_values(row, kept_cols[2:]))
                    if self.debug > 1 and False:
                        print(len(indices), len(values))
                    df.iloc[indices,2:] = values
                if self.debug:
                    print(util.make_msg_block("Collecting File Data %.2fs" % (time.time() - start_time), "."))
                if self.debug > 1:
#                    print(df.iloc[:3*file_n_temporal*n_spatial,:])
                    print(df)
                if self.debug:
                    print(util.make_msg_block("Saving Spatiotempoal Data", "."))
                    start_time = time.time()
                df.to_csv(path, index=False, compression="gzip")
                if self.debug:
                    print(
                        util.make_msg_block(
                            "Saving Spatiotemporal Data %.2fs" % (time.time() - start_time), 
                            "."
                        )
                    )
            elif self.debug > 2:
                df = pd.read_csv(path, compression="gzip")
                print(df)
                print(df.describe())


class US_Streams(Integrator):

    def name(self):
        return "US-Streams"

    def acquire(self, args):
        print(
            ">>> Please download all files from %s to %s and unzip them" % (
                "https://drive.google.com/drive/folders/1AVB_0PnIQscIv601rFRxtMrSSV32tXYh", 
                self.acquire_dir()
            )
        )
        print(
            ">>> Please download \"%s\" from %s to %s" % (
                "state.txt", 
                "https://www2.census.gov/geo/docs/reference/state.txt", 
                self.acquire_dir()
            )
        )

    def convert(self, args):
        import warnings
        warnings.simplefilter(action='ignore', category=FutureWarning)
        state_df = pd.read_csv(os.sep.join([self.acquire_dir(), "state.txt"]), sep="|", dtype={"STATE": str, "STATENS": str})
        col_dtype_map = {
            "site_no": str, 
            "station_nm": str, 
            "dec_lat_va": float, 
            "dec_long_va": float, 
            "coord_acy_cd": str, 
            "dec_coord_datum_cd": str, 
            "state_cd": str, 
            "alt_va": float, 
            "alt_acy_va": float, 
            "alt_datum_cd": str, 
            "huc_cd": str, 
            "drain_area_va": float, 
            "contrib_drain_area_va": float, 
            "discharge_begin_date": str, 
            "discharge_end_date": str, 
            "discharge_count_nu": float, 
        }
        data_dir = os.sep.join([self.acquire_dir(), "US Streamflow", "USGSdata2014"])
        header_paths = glob.glob(os.sep.join([data_dir, "*H.txt"]))
        series_paths = glob.glob(os.sep.join([data_dir, "*Q.txt"]))
        site_nos = [path.split(os.sep)[-1].replace("Q.txt", "") for path in series_paths]
        data_dir = os.sep.join([self.acquire_dir(), "US Streamflow", "USGSdata2014"])
        ######################################
        ### Get spatial data for entire US ###
        ######################################
        if not os.path.exists(self.spatial_features_outpath()):
            col_data_map = {}
            dfs = []
            for site_no in site_nos:
                path = os.sep.join([data_dir, "%sH.txt" % (site_no)])
                if os.path.exists(path):
                    df = self.read_csv(path, sep="\t", dtype=col_dtype_map, na_values=-999.99)
                else:
                    raise ValueError(site_no)
                if len(df) > 1:
                    print(df)
                    idx = -1
                    if site_no == "08447410":
                        idx = 0
                    df = df.iloc[[idx],:].reset_index(drop=True)
                    print(df)
                if not df.iloc[len(df)-1]["state_cd"] in list(state_df["STATE"]):
                    print("Found site_no=%s at state_cd=%s which does not belong to the US. Skipping..." % (site_no, df["state_cd"][0]))
                    continue
                _df = self.read_csv(
                    os.sep.join([data_dir, "%sQ.txt" % (site_no)]), 
                    sep="\t", 
                    dtype={"agency_cd": str, "site_no": str, "dv_dt": str, "dv_va": float, "dv_cd": str}, 
                    na_values=-999.99, 
                )
                if len(_df) < 2:
                    print("State[%s, %s] contains less than 2 gauges. Skipping it..." % (state_id, state_abbrev))
                    continue
                dfs.append(df)
            df = pd.concat(dfs)
            print(df)
            df.to_csv(self.spatial_features_outpath(), index=False)
        spa_df = pd.read_csv(self.spatial_features_outpath(), dtype=col_dtype_map)
        site_nos = list(spa_df["site_no"])
        # Get number of values present for each gauge
        path = os.sep.join([self.convert_dir(), "ValuesPresent.csv"])
        if not os.path.exists(path):
            df = pd.DataFrame(columns=["site_no", "dv_va"], index=range(len(site_nos)))
            df["site_no"] = site_nos
            _var = Container()
            _var.temporal_label_field = "dv_dt"
            _var.feature_fields = ["dv_va"]
            values_present = []
            for i, site_no in enumerate(site_nos):
                _df = self.read_csv(
                    os.sep.join([data_dir, "%sQ.txt" % (site_no)]), 
                    sep="\t", 
                    usecols=["site_no", "dv_dt", "dv_va"], 
                    dtype={"agency_cd": str, "site_no": str, "dv_dt": str, "dv_cd": str}, 
                    na_values=-999.99, 
                )
                missing = Imputation.missing_values(_df, "temporal", _var, normalize=False)
                values_present.append(len(_df) - missing.loc[0,"dv_va"])
            df["dv_va"] = values_present
            df.to_csv(path, index=False)
        if 0:
            print(spa_df)
            print(spa_df[["site_no"]].isna().sum())
            print(state_df)
            input()
        ####################################################
        ### Get data for each individual state/territory ###
        ####################################################
        indices = slice(0, 3)
        indices = slice(None)
        state_ids = state_df["STATE"].to_numpy()[indices]
        state_abbrevs = state_df["STUSAB"].to_numpy()[indices]
        #   Get spatial data for each
        print(util.make_msg_block("Converting Spatial Features for each State"))
        for state_id, state_abbrev in zip(state_ids, state_abbrevs):
            path = os.sep.join([self.convert_dir(), state_abbrev, self.spatial_features_fname()])
            if os.path.exists(path):
                continue
            _df = spa_df.loc[spa_df["state_cd"] == state_id]
            if len(_df) < 2:
                print("State[%s, %s] contains less than 2 gauges. Skipping it..." % (state_id, state_abbrev))
                continue
            _df["elev_m"] = Acquisition.get_latlon_elevation(list(_df["dec_lat_va"]), list(_df["dec_long_va"]))
            _df.to_csv(path, index=False)
        #   Get spatiotemporal data for each state
        from geopy import distance
        print(util.make_msg_block("Converting Spatiotemporal Features for each State"))
        date_frmt = "%Y-%m-%d"
        for state_id, state_abbrev in zip(state_ids, state_abbrevs):
            path = os.sep.join([self.convert_dir(), state_abbrev, self.spatiotemporal_features_fname()])
            if os.path.exists(path):
                continue
            _path = os.sep.join([self.convert_dir(), state_abbrev, self.spatial_features_fname()])
            if not os.path.exists(_path):
                continue
            print(state_id, state_abbrev)
            _spa_df = pd.read_csv(
                _path, 
                dtype={"site_no": str, "huc": str, "huc12": str, "tohuc12": str}
            )
            _site_nos = list(_spa_df["site_no"])
            print(spa_df)
            print(_spa_df)
            min_date, max_date = "9999-12-31", "0001-01-01"
            dfs = []
            # Collect streamflow records for each gauge nad determine the complete temporal interval
            for _site_no in _site_nos:
                _df = self.read_csv(
                    os.sep.join([data_dir, "%sQ.txt" % (_site_no)]), 
                    sep="\t", 
                    usecols=["site_no", "dv_dt", "dv_va"], 
                    dtype={"agency_cd": str, "site_no": str, "dv_dt": str, "dv_cd": str}, 
                    na_values=-999.99, 
                )
                # Get span of time for this gauge and update global time-span
                _min_date, _max_date = min(_df["dv_dt"]), max(_df["dv_dt"])
                min_dt = min(dt.datetime.strptime(_min_date, date_frmt), dt.datetime.strptime(min_date, date_frmt))
                max_dt = max(dt.datetime.strptime(_max_date, date_frmt), dt.datetime.strptime(max_date, date_frmt))
                min_date, max_date = dt.datetime.strftime(min_dt, date_frmt), dt.datetime.strftime(max_dt, date_frmt)
                dfs.append(_df)
            print(min_date, max_date)
            n_spatial = len(_site_nos)
            print(n_spatial)
            temporal_labels = util.generate_temporal_labels(min_date, max_date, [1, "days"], date_frmt, [True, True])
            n_temporal = len(temporal_labels)
            # Insert streamflow records for each gauge into greater time-series df
            df = pd.DataFrame(columns=["dv_dt"]+_site_nos, index=range(n_temporal))
            df["dv_dt"] = temporal_labels
            for _site_no, _df in zip(_site_nos, dfs):
                print(_site_no, "%d/%d" % (len(_df)-_df["dv_va"].isna().sum(), n_temporal))
                mask = df["dv_dt"].isin(_df["dv_dt"])
                df.loc[mask,_site_no] = _df.set_index(mask[mask].index)["dv_va"]
            print(df)
            streamflow_df = df.melt(id_vars=["dv_dt"])
            streamflow_df.columns = ["dv_dt", "site_no", "dv_va"]
            print(streamflow_df)
            # Collect meteorological records for each gauge
            coords = _spa_df[["dec_lat_va","dec_long_va","elev_m"]].to_numpy()
            min_latlon, max_latlon = np.min(coords[:,:2], 0), np.max(coords[:,:2], 0)
            min_elev, max_elev = np.min(coords[:,-1]), np.max(coords[:,-1])
            max_dist_m = distance.distance(min_latlon, max_latlon).m
            dfs = Acquisition.get_meteostat_data(
                coords[:], [min_dt, max_dt], radius=int(max_dist_m / 4 + 1), alt_range=int(max_elev + 1)
            )
            print(dfs[0])
            print(dfs[-1])
            df = pd.concat(dfs).set_index(streamflow_df.index)
            df.insert(1, "dv_va", streamflow_df["dv_va"])
            df.insert(1, "site_no", streamflow_df["site_no"])
            df.columns = ["dv_dt"] + list(df.columns)[1:]
            print(df)
            input()
            print(get_missing(df, "site_no", "dv_dt"))
            input()
            pd.DataFrame({"site_no": _site_nos}).to_csv(
                os.sep.join([self.convert_dir(), state_abbrev, self.spatial_labels_fname()]), index=False
            )
            pd.DataFrame({"dv_dt": temporal_labels}).to_csv(
                os.sep.join([self.convert_dir(), state_abbrev, self.temporal_labels_fname()]), index=False
            )
            df.to_csv(
                os.sep.join([self.convert_dir(), state_abbrev, self.spatiotemporal_features_fname()]), index=False
            )
        return
        # Get data for all states/sites
        #   Get all spatial labels (site numbers)
        if not os.path.exists(self.spatial_labels_outpath()):
            df = pd.read_csv(self.spatial_features_outpath(), dtype={"site_no": str})
            pd.DataFrame({"site_no": df["site_no"]}).to_csv(self.spatial_labels_outpath(), index=False)
        #   Get all temporal labels (time-stamps dv_dt)
        if not os.path.exists(self.temporal_labels_outpath()):
            min_date, max_date = "9999-12-31", "0001-01-01"
            for state_id, state_abbrev in zip(state_ids, state_abbrevs):
                path = os.sep.join([self.convert_dir(), state_abbrev, self.temporal_labels_fname()])
                if not os.path.exists(path):
                    continue
                df = pd.read_csv(path, dtype={"dv_dt": str})
                _min_date, _max_date = min(df["dv_dt"]), max(df["dv_dt"])
                min_dt = min(dt.datetime.strptime(_min_date, date_frmt), dt.datetime.strptime(min_date, date_frmt))
                max_dt = max(dt.datetime.strptime(_max_date, date_frmt), dt.datetime.strptime(max_date, date_frmt))
                min_date, max_date = dt.datetime.strftime(min_dt, date_frmt), dt.datetime.strftime(max_dt, date_frmt)
            print(min_date, max_date)
            temporal_labels = util.generate_temporal_labels(min_date, max_date, [1, "days"], date_frmt, [True, True])
            pd.DataFrame({"dv_dt": temporal_labels}).to_csv(self.temporal_labels_outpath(), index=False)
        #   Compile records (streamflow) from all sites into one
        if not os.path.exists(self.spatiotemporal_features_outpath()):
            spatial_labels = list(pd.read_csv(self.spatial_labels_outpath(), dtype={"site_no": str})["site_no"])
            temporal_labels = list(pd.read_csv(self.temporal_labels_outpath(), dtype={"dv_dt": str})["dv_dt"])
            n_spatial = len(spatial_labels)
            n_temporal = len(temporal_labels)
            print("Size = (%d, %d)" % (n_spatial, n_temporal))
            df = pd.DataFrame(columns=["dv_dt"]+spatial_labels, index=range(n_temporal))
            df["dv_dt"] = temporal_labels
            for i, site_no in enumerate(spatial_labels):
                _df = self.read_csv(
                    os.sep.join([data_dir, "%sQ.txt" % (site_no)]), 
                    sep="\t", 
                    usecols=["site_no", "dv_dt", "dv_va"], 
                    dtype={"agency_cd": str, "site_no": str, "dv_dt": str, "dv_cd": str},
                    na_values=-999.99, 
                )
                print("%d/%d" % (i+1, n_spatial), site_no, "%d/%d" % (len(_df)-_df[["dv_va"]].isna().sum(), n_temporal))
                mask = df["dv_dt"].isin(_df["dv_dt"])
                df.loc[mask,site_no] = _df.set_index(mask[mask].index)["dv_va"]
            print(df)
            df = df.melt(id_vars=["dv_dt"])
            df.columns = ["dv_dt", "site_no", "dv_va"]
            print(df)
            print(get_missing(df, "site_no", "dv_dt"))
            df.to_csv(self.spatiotemporal_features_outpath(), index=False)
        path = os.sep.join([self.convert_dir(), "DatasetVariables.py"])
        if not os.path.exists(path):
            print(">>> Please Implement the module \"DatasetVariables.py\" at %s. Once done, rerun this script to have this module copied to all state partitions of this dataset %s..." % (self.convert_dir(), ", ".join(state_abbrevs[:3])))
            return
        with open(path, "r") as f:
            module_str = f.read()
        for state_id, state_abbrev in zip(state_ids, state_abbrevs):
            _dir =os.sep.join([self.convert_dir(), state_abbrev])
            if not os.path.exists(_dir):
                continue
            _path = os.sep.join([_dir, "DatasetVariables.py"])
            _module_str = module_str.replace(
                "return \"us-streams\"", 
                "return \"us-streams-%s\"" % (state_abbrev.lower())
            )
            with open(_path, "w") as f:
                f.write(_module_str)

    def read_csv(self, path, **pd_kwargs):
        with open(path, "r") as f:
            lines = f.read().split("\n")
        lines = [line for line in lines if not re.match("^#.*|^\d+s\t\d+s.*", line)]
        return pd.read_csv(StringIO("\n".join(lines)), **pd_kwargs)


class California(Integrator):

    def acquire(self, args):
        print(
            "Please download \"%s\" from %s to %s" % (
                "California Road Network's Nodes and Edges", 
                "https://www.cs.utah.edu/~lifeifei/SpatialDataset.htm", 
                self.acquire_dir()
            )
        )

    def convert(self, args):
        print(self.acquire_dir())
        print(self.convert_dir())
        path = os.sep.join([self.acquire_dir(), "cal.cnode.txt"])
        node_df = pd.read_csv(path, sep=" ", names=["Node_ID", "Longitude", "Latitude"])
        print(node_df)
        node_labels = node_df["Node_ID"].to_numpy(dtype=str)
        node_locations = node_df[["Longitude", "Latitude"]].to_numpy()
        path = os.sep.join([self.acquire_dir(), "cal.cedge.txt"])
        edge_df = pd.read_csv(path, sep=" ", names=["Edge_ID", "Start_Node_ID", "End_Node_ID", "L2_Distance"])
        print(edge_df)
        edge_labels = edge_df["Edge_ID"].to_numpy(dtype=str)
        edges = edge_df[["Start_Node_ID", "End_Node_ID"]].to_numpy(dtype=str)
        print(edges.shape)
        print(edges)
        from Data.DataSelection import DataSelection
        data_sel = DataSelection()
        edge_indices = np.stack(
            (
                data_sel.indices_from_selection(node_labels, ["literal"] + list(edges[:,0])), 
                data_sel.indices_from_selection(node_labels, ["literal"] + list(edges[:,1]))
            )
        )
        print(edge_indices.shape)
        print(edge_indices)
        from matplotlib import pyplot as plt
        plt.figure(figsize=(32,32))
        plt.scatter(node_locations[:,0], node_locations[:,1], s=25)
        for i, j in np.swapaxes(edge_indices, 0, 1):
            plt.plot(node_locations[i,0], node_locations[j,1], linewidth=0.1)
        path = os.sep.join([self.convert_dir(), "graph.png"])
        plt.savefig(path, bbox_inches="tight")
        plt.close()


class Arsenic(Integrator):

    def acquire(self, args):
        path = os.sep.join([self.acquire_dir(), "as.csv"])
        if not os.path.exists(path):
            print(
                "Please download as.csv from %s and place it in the folder %s" % (
                    "https://drive.google.com/file/d/1vej6A5RHQmUURfssf6_jKbPIgosD5RxT/view?ts=635ae76d", 
                    self.acquire_dir()
                )
            )

    def convert(self, args):
        var = Container().set(
            [
                "remove_duplicates", 
                "sort_by", 
                "keep", 
                "spatial_label_field", 
            ], 
            [
                True, 
                "WellDepth", 
                "last", 
                "SiteID", 
            ], 
            multi_value=True
        )
        var = var.merge(args)
        path = os.sep.join([self.acquire_dir(), "as.csv"])
        df = pd.read_csv(path)
        df = df.drop(["Data", "as10", "Pred"], axis=1)
        print(df)
        cols = util.list_subtract(list(df.columns), [var.spatial_label_field])
        df = df.dropna(how="all", subset=cols)
        print(df)
        input()
        if var.remove_duplicates:
#            df = df.sort_values(var.sort_by).drop_duplicates(subset=[var.spatial_label_field], keep=var.keep) 
#            df = df.sort_values(var.sort_by).drop_duplicates(subset=["X_Albers", "Y_Albers"], keep=var.keep) 
#            df = df.sort_values(var.spatial_label_field)
            df = df.drop_duplicates(subset=["X_Albers", "Y_Albers", "WellDepth"], keep=var.keep) 
            df[var.spatial_label_field] = range(len(df))
        print(df)
        input()
        spatial_labels, counts = np.unique(df[var.spatial_label_field].to_numpy(), return_counts=True)
        print(spatial_labels)
        print(np.unique(counts))
        missing_values_df = df.drop(var.spatial_label_field, axis=1).isna().sum().reset_index()
        print(missing_values_df)
        path = os.sep.join([self.cache_dir(), "MissingValuesSummary.csv"])
        missing_values_df.to_csv(path, index=False)
        # Write results
        pd.DataFrame({"SiteID": spatial_labels}).to_csv(self.spatial_labels_outpath(), index=False)
        df.to_csv(self.spatial_features_outpath(), index=False)


class Energy(Integrator):

    def acquire(self, args):
        return
        path = os.sep.join([self.acquire_dir(), "ALL_HVAC.zip"])
        if not os.path.exists(path):
            print("Please download All_HVAC.zip and extract to folder %s" % (self.acquire_dir()))

    def convert(self, args):
        _args = Container().set(
            [
                "device", 
                "debug", 
            ], 
            [
                "HVAC", 
                0, 
            ],
            multi_value=True
        )
        args = _args.merge(args)
        # Start
        root_dir = os.sep.join([self.root_dir(), args.device])
        acquire_dir = os.sep.join([root_dir, "Integration", "Acquired"])
        if args.device == "HVAC":
            data_dir = os.sep.join([acquire_dir, "All_HVAC"])
            paths = sorted(util.get_paths(data_dir, "hvac\d+.csv"))
            dfs = []
            for i, path in enumerate(paths):
                df = pd.read_csv(path)
                name = os.path.basename(path).replace(".csv", "")
                df = self.convert_hvac(df, name)
                df.columns = [
                    "date", 
                    "device", 
                    "mixed_air_humidity", 
                    "supply_air_pressure", 
                    "supply_fan_flow", 
                    "return_air_humidity", 
                    "return_fan_flow", 
                    "mixed_air_temp", 
                    "preheat_coil_temp"
                ]
                dfs.append(df)
            lens = [len(df) for df in dfs]
            print(lens)
            dts = [dt.datetime.strptime(date, "%Y-%m-%d_%H-%M-%S") for date in df["date"] for df in dfs]
            min_dt = min(dts)
            max_dt = max(dts)
            temporal_labels = util.generate_temporal_labels(min_dt, max_dt, [1, "minutes"], incl=[1, 1])
            for i, _df in enumerate(dfs):
                df = pd.DataFrame(index=range(len(temporal_labels)), columns=_df.columns)
                df["date"] = temporal_labels
                df["device"] = _df.loc[0,"device"]
                if args.debug:
                    print(_df)
                    print(df)
                df = Conversion.dataframe_insert(df, _df, "date")
                if args.debug:
                    print(df)
                    input()
                dfs[i] = df
            df = pd.concat(dfs)
        else:
            raise NotImplementedError(args.device)
        print(df)
        path = os.sep.join([root_dir, self.spatial_labels_fname()])
        df[["device"]].drop_duplicates().to_csv(path, index=False)
        path = os.sep.join([root_dir, self.temporal_labels_fname()])
        df[["date"]].drop_duplicates().to_csv(path, index=False)
        path = os.sep.join([root_dir, self.spatiotemporal_features_fname()])
        df.to_csv(path, index=False, compression="gzip")

    def convert_hvac(self, df, name):
        df.columns = ["date"] + [col.lower().replace(" ", "_") for col in df.columns[1:]]
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d_%H-%M-%S")
        df.insert(1, "device", name)
        return df

    def _convert(self, args):
        import json
        path = os.sep.join([self.acquire_dir(), "asset_dictionary.json"])
        with open(path, "r") as f:
            asset_dict = json.load(f)
        def dict_to_str(a):
            return json.dumps(a, sort_keys=True, indent=4)
        print(dict_to_str(asset_dict))
        input()
        hvac_ids = []
        for key, value in asset_dict.items():
            if value["asset_layer_3"] == "HVAC":
                print(key, value["asset_layer_3"], value["asset_layer_4"], sep="|")
                print(dict_to_str(value["asset_properties"]))
                if value["asset_layer_4"].startswith("Air Handling Unit"):
                    hvac_ids.append(key)
        input()
        print(len(asset_dict))
        paths = util.get_paths(os.sep.join([self.acquire_dir(), "asset-property-updates"]), "^.*\.parquet", recurse=True)
        for i, path in enumerate(paths):
            print(os.path.basename(path))
            _str = re.sub("meghaairesourceemc2_firehose_delivery_stream-\d-|.parquet", "", os.path.basename(path))
            print(_str)
            date_str = re.sub("-[a-z0-9]{8}-[a-z0-9]{4}-[a-z0-9]{4}-[a-z0-9]{4}-[a-z0-9]{12}", "", _str)
            print(date_str)
            asset_id = re.sub("[0-9]{4}-[0-9]{2}-[0-9]{2}-[0-9]{2}-[0-9]{2}-[0-9]{2}-", "", _str)
            print(asset_id)
            input()
            df = pd.read_parquet(path)
#            df = df.drop(["type"], axis=1)
            df = df.drop(["type", "asset_property_quality", "asset_property_data_type"], axis=1)
            print(df)
            _df = df.loc[df["asset_id"].isin(hvac_ids)]
            _df = df.loc[df["asset_id"] == asset_id]
            if not _df.empty:
                print(_df)
                input()


class IU_Energy(Integrator):

    def name(self):
        return "IU-Energy"

    def acquire(self, args):
        pass

    def convert(self, args):
        n_temporal = 1000 # time-steps
        n_spatial = 3 # number of devices for this data
        n_feature = 7 # number of features on each device
        feature_labels = np.array(["f%d" % (i) for i in range(n_feature)])
        spatial_labels = np.array(["M0%d" % (i) for i in range(n_spatial)])
        temporal_labels = np.array(
            util.generate_temporal_labels("2000-01-01_00-00-00", n_temporal, [1, "seconds"])
        )
        # Generate spatial/temporal label files
        pd.DataFrame({"device": spatial_labels}).to_csv(self.spatial_labels_outpath(), index=False)
        pd.DataFrame({"date": temporal_labels}).to_csv(self.temporal_labels_outpath(), index=False)
        # Generate spatiotemporal feature file
        np.random.seed(0)
        feature = np.random.normal(size=(n_temporal, n_spatial, n_feature))
        print(feature.shape)
        # Expected format:
        #    Header: date,device,f0,f1,f2,...,fn
        data = {
            "date": np.tile(temporal_labels, n_spatial), 
            "device": np.repeat(spatial_labels, n_temporal)
        }
        for f in feature_labels:
            data[f] = 0
        df = pd.DataFrame(data)
        print(df)
        df.to_csv(self.spatiotemporal_features_outpath().replace(".gz", ""), index=False)
