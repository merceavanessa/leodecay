import json
import requests
import pandas as pd
from datetime import datetime, timedelta
from leodecay.data.data_downloader import DataDownloader


class TLELoader(DataDownloader):
    def __init__(self, folder_path, conf_path=None, dataset=None, start_date="1995-01-01", end_date="2024-03-03", username=None, password=None):
        self.dataset = dataset
        self.folder_path = folder_path
        self.conf_path = conf_path
        self.start_date = start_date
        end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")
        end_date_dt = end_date_dt + timedelta(days=1)

        self.end_date = end_date_dt.strftime("%Y-%m-%d")
        self.uriBase = "https://www.space-track.org"
        self.requestLogin = "/ajaxauth/login"
        self.requestCmdAction = "/basicspacedata/query"
        self.siteCred = {'identity': username, 'password': password}

    def load_data(self):
        try:
            with open(f"{self.conf_path}/{self.dataset}/config.json", 'r') as file:
                cfg = json.load(file)
                self.fetch_data(cfg)
                return f"{self.folder_path}/{self.dataset}"
        except Exception as e:
            raise SpaceTrackRequestError(e, "POST fail on login")


    def fetch_data(self, cfg):
       with requests.Session() as session:
        resp = session.post(
            self.uriBase + self.requestLogin, data=self.siteCred)
        if resp.status_code != 200:
            raise SpaceTrackRequestError(resp, "POST fail on login")

        for satellite_name in cfg:
            norad_id = cfg[satellite_name]
            requestFindSat = f"/class/gp_history/NORAD_CAT_ID/{norad_id}/EPOCH/{self.start_date}--{self.end_date}/format/json"
            resp = session.get(
                self.uriBase + self.requestCmdAction + requestFindSat)
            if resp.status_code != 200:
                print(resp.text)
                raise SpaceTrackRequestError(
                    resp, f"GET fail on request for satellite {norad_id}")
            else:
                data = [SatelliteData(item) for item in json.loads(resp.text)]
                df = convert_to_dataframe(data)
                df.to_csv(f"{self.folder_path}/{self.dataset}/{satellite_name}.csv", index=False)
                print(f"TLE Data for {satellite_name} saved to {self.folder_path}/{self.dataset}/{satellite_name}.csv")

class SatelliteData:
    def __init__(self, data):
        self.CCSDS_OMM_VERS = data.get('CCSDS_OMM_VERS')
        self.COMMENT = data.get('COMMENT')
        self.CREATION_DATE = data.get('CREATION_DATE')
        self.ORIGINATOR = data.get('ORIGINATOR')
        self.OBJECT_NAME = data.get('OBJECT_NAME')
        self.OBJECT_ID = data.get('OBJECT_ID')
        self.CENTER_NAME = data.get('CENTER_NAME')
        self.REF_FRAME = data.get('REF_FRAME')
        self.TIME_SYSTEM = data.get('TIME_SYSTEM')
        self.MEAN_ELEMENT_THEORY = data.get('MEAN_ELEMENT_THEORY')
        self.EPOCH = data.get('EPOCH')
        self.MEAN_MOTION = data.get('MEAN_MOTION')
        self.ECCENTRICITY = data.get('ECCENTRICITY')
        self.INCLINATION = data.get('INCLINATION')
        self.RA_OF_ASC_NODE = data.get('RA_OF_ASC_NODE')
        self.ARG_OF_PERICENTER = data.get('ARG_OF_PERICENTER')
        self.MEAN_ANOMALY = data.get('MEAN_ANOMALY')
        self.EPHEMERIS_TYPE = data.get('EPHEMERIS_TYPE')
        self.CLASSIFICATION_TYPE = data.get('CLASSIFICATION_TYPE')
        self.NORAD_CAT_ID = data.get('NORAD_CAT_ID')
        self.ELEMENT_SET_NO = data.get('ELEMENT_SET_NO')
        self.REV_AT_EPOCH = data.get('REV_AT_EPOCH')
        self.BSTAR = data.get('BSTAR')
        self.MEAN_MOTION_DOT = data.get('MEAN_MOTION_DOT')
        self.MEAN_MOTION_DDOT = data.get('MEAN_MOTION_DDOT')
        self.SEMIMAJOR_AXIS = data.get('SEMIMAJOR_AXIS')
        self.PERIOD = data.get('PERIOD')
        self.APOAPSIS = data.get('APOAPSIS')
        self.PERIAPSIS = data.get('PERIAPSIS')
        self.OBJECT_TYPE = data.get('OBJECT_TYPE')
        self.RCS_SIZE = data.get('RCS_SIZE')
        self.COUNTRY_CODE = data.get('COUNTRY_CODE')
        self.LAUNCH_DATE = data.get('LAUNCH_DATE')
        self.SITE = data.get('SITE')
        self.DECAY_DATE = data.get('DECAY_DATE')
        self.FILE = data.get('FILE')
        self.GP_ID = data.get('GP_ID')
        self.TLE_LINE0 = data.get('TLE_LINE0')
        self.TLE_LINE1 = data.get('TLE_LINE1')
        self.TLE_LINE2 = data.get('TLE_LINE2')

    def __repr__(self):
        return (f"SatelliteData({self.OBJECT_NAME}, {self.OBJECT_ID}, {self.EPOCH})")

    def display_info(self):
        attributes = vars(self)
        for attribute, value in attributes.items():
            print(f"{attribute}: {value}")


class SpaceTrackRequestError(Exception):
    def __init___(self, args):
        Exception.__init__(
            self, "my exception was raised with arguments {0}".format(args))
        self.args = args


def convert_to_dataframe(objects_list):
    data = [obj.__dict__ for obj in objects_list]
    df = pd.DataFrame(data)
    return df
