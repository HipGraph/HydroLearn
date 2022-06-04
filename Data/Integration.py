from selenium import webdriver
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException
import itertools
import gzip
import pandas as pd
import os
import sys
import time
import requests
import urllib.request
import webbrowser
import shutil
import glob
import numpy as np
import datetime as dt
from inspect import currentframe
from Data.DataSelection import DataSelection
from Container import Container
import Utility as util


def data_dir():
    return os.sep.join(os.path.realpath(__file__).replace(".py", "").split(os.sep)[:-1])


def download_dir():
    from pathlib import Path
    return os.sep.join([str(Path.home()), "Downloads"])


class Integration:

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

    def spatiotemporal_features_inpath(self):
        func_name = "%s.%s" % (self.__class__.__name__, currentframe().f_code.co_name)
        raise NotImplementedError("Implement %s() please!" % (func_name))

    def spatial_features_inpath(self):
        return None

    def temporal_features_inpath(self):
        return None

    def graph_features_inpath(self):
        return None

    def spatial_labels_outpath(self):
        return os.sep.join([self.convert_dir(), "SpatialLabels.csv"])

    def temporal_labels_outpath(self):
        return os.sep.join([self.convert_dir(), "TemporalLabels.csv"])

    def spatiotemporal_features_outpath(self):
        return os.sep.join([self.convert_dir(), "Spatiotemporal.csv.gz"])

    def spatial_features_outpath(self):
        return os.sep.join([self.convert_dir(), "Spatial.csv"])

    def temporal_features_outpath(self):
        return os.sep.join([self.convert_dir(), "Temporal.csv"])

    def graph_features_outpath(self):
        return os.sep.join([self.convert_dir(), "Graph.csv"])

    def acquire(self, args):
        func_name = "%s.%s" % (self.__class__.__name__, currentframe().f_code.co_name)
        raise NotImplementedError("Implement %s() please!" % (func_name))

    def convert(self, args):
        func_name = "%s.%s" % (self.__class__.__name__, currentframe().f_code.co_name)
        raise NotImplementedError("Implement %s() please!" % (func_name))


class Electricity(Integration):

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


class Traffic(Integration):

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


class Solar_Energy(Integration):

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


class Exchange_Rate(Integration):

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


class METR_LA(Integration):

    sort = True
#    sort = False

    def name(self):
        return "METR-LA"

    def spatiotemporal_features_inpath(self):
        return os.sep.join([self.acquire_dir(), "metr-la.h5"])

    def spatial_features_inpath(self):
        return os.sep.join([self.acquire_dir(), "graph_sensor_locations.csv"])

    def graph_inpath(self):
        return os.sep.join([self.acquire_dir(), "adj_mx.pkl"])

    def acquire(self, args):
        paths_exist = [
            os.path.exists(self.spatiotemporal_features_inpath()), 
            os.path.exists(self.spatial_features_inpath()), 
            os.path.exists(self.graph_inpath()), 
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
                "Download \"%s\" and \"%s\" from sub-directory \"%s\" of repository at \"\" and save to acquisition directory \"%s\"" % (
                    os.path.basename(self.spatial_features_inpath()), 
                    os.path.basename(self.graph_inpath()), 
                    "./data/sensor_graph/", 
                    "https://github.com/liyaguang/DCRNN", 
                    self.acquire_dir(), 
                )
            )

    def convert(self, args):
        # Convert METR-LA
        paths_exist = [
            os.path.exists(self.temporal_labels_outpath()), 
            os.path.exists(self.spatial_labels_outpath()), 
            os.path.exists(self.spatiotemporal_features_outpath()), 
        ]
        if not all(paths_exist):
            path = os.sep.join([self.acquire_dir(), self.spatmp_fname])
            assert os.path.exists(path), "Missing data file \"%s\"" % (path)
            #   Load original data
            df = pd.read_hdf(path)
            #   Convert temporal labels
            temporal_labels = np.array(df.index, dtype=str)
            n_temporal = temporal_labels.shape[0]
            for i in range(temporal_labels.shape[0]):
                temporal_labels[i] = temporal_labels[i].replace(".000000000", "")
                temporal_labels[i] = temporal_labels[i].replace("T", "_")
                temporal_labels[i] = temporal_labels[i].replace(":", "-")
            #   Convert spatial labels
            spatial_labels = np.array(df.columns, dtype=str)
            n_spatial = spatial_labels.shape[0]
            #   Convert spatiotemporal features
            spatmp = df.to_numpy()
            if self.sort:
                import natsort
                sort_indices = natsort.index_natsorted(spatial_labels)
                spatial_labels = spatial_labels[sort_indices]
                spatmp = spatmp[:,sort_indices]
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
                "speedmph": np.reshape(spatmp, -1), 
            })
            if self.debug:
                print(df.loc[df["date"] == temporal_labels[0]])
                print(df.loc[df["date"] == temporal_labels[1]])
                print(df.loc[df["date"] == temporal_labels[-2]])
                print(df.loc[df["date"] == temporal_labels[-1]])
            pd.DataFrame({"date": temporal_labels}).to_csv(self.temporal_labels_outpath(), index=False)
            pd.DataFrame({"sensor": spatial_labels}).to_csv(self.spatial_labels_outpath(), index=False)
            df.to_csv(self.spatiotemporal_features_outpath(), index=False, compression="gzip")
        #   Convert spatial features
        if not os.path.exists(self.spatial_features_outpath()):
            path = os.sep.join([self.acquire_dir(), self.spatial_fname])
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
        if not os.path.exists(self.graph_outpath()):
            path = os.sep.join([self.acquire_dir(), self.graph_fname])
            assert os.path.exists(path), "Missing data file \"%s\"" % (path)
            in_path, out_path = path, os.sep.join([
                self.cache_dir(), 
                self.graph_fname.replace(".pkl", "_fixed.pkl")
            ])
            self.fix_file_content(in_path, out_path)
            path = out_path
            data = util.from_cache(path, encoding="bytes")
            spatial_labels = np.array([lab.decode("utf-8") for lab in data[0]])
            spatial_labels = pd.read_csv(self.spatial_labels_outpath()).to_numpy(dtype=str).reshape(-1)
            label_index_map = util.to_dict([lab.decode("utf-8") for lab in data[1].keys()], data[1].values())
            adj = data[2]
            indices = util.get_dict_values(label_index_map, spatial_labels)
            if self.debug:
                print(label_index_map)
                print(spatial_labels)
                print(indices)
            dat_sel = DataSelection()
            adj = dat_sel.filter_axis(adj, [0, 1], [indices, indices])
            df = util.adjacency_to_edgelist(pd.DataFrame(adj, columns=spatial_labels))
            if self.debug:
                print(df)
            df.to_csv(self.graph_outpath(), index=False)


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

    sort = True
#    sort = False
    debug = 2
    spatmp_fname = "pems-bay.h5"
    spatial_fname = "graph_sensor_locations_bay.csv"
    graph_fname = "adj_mx_bay.pkl"

    def name(self):
        return "PEMS-BAY"

    def load_spatial_df(self, path):
        return pd.read_csv(path, names=["sensor", "latitude", "longitude"])


class CaltransPeMS(Integration):

    # Notes:
    #   It turns out that the files from CaltransPeMS do not always contain the same number of time-steps nor
    #       do they always contain the same number of stations/nodes.
    #   As such, the conversion from a list of files into a single Spatiotemporal.csv file cannot be implmented
    #       trivially with a simple concatenation.

    def __init__(self):
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
