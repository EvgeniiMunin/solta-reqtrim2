# -*- coding: utf-8 -*-
#import click
import logging
from pathlib import Path
import pyarrow.csv as pv
#from dotenv import find_dotenv, load_dotenv

import pandas as pd
import logging
#from pydantic import BaseModel, validator, ValidationError


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_data(file_path: str) -> pd.DataFrame:
    try:
        table = pv.read_csv(file_path)
        df = table.to_pandas()

        #df.columns = ['req_ts', 'req_id', 'track.ssp_id', 'dsp_id', 'creative_type',
        #              'bid_state', 'dsp_deal_id', 'first_dsp_dealid', 'floor', 'imp', 'str',
        #              'vtr', 'ctr', 'vctr', 'ts', 'id', 'ssp_id', 'site_id', 'domain_id',
        #              'tz_offset', 'pub_uid', 'iab_cat', 'api', 'browser_version',
        #              'page_domain', 'page_deep2', 'page_deep3', 'region_id', 'device_id',
        #              'os_id', 'browser_id']

        print("df.shape: ", df.shape)

        #df_dict = df.to_dict(orient="list")
        #validated_data = InputDataSchema(**df_dict)
        return df

    #except ValidationError as e:
    #    logger.error("Data validation error: %s", str(e))
    #    raise

    except Exception as e:
        logger.error("Failed to load data from path: %s with error: %s", file_path, str(e))
        raise


#class InputDataSchema(BaseModel):
#    req: pd.Timestamp
#    req_id: int
#    track_ssp_id: int
#    dsp_id: int
#    creative_type: str
#    bid_state: str
#    dsp_deal_id: int
#    first_dsp_dealid: int
#    floor: float
#    imp: str
#    str_kpi: float
#    vtr: float
#    ctr: float
#    vctr: float
#    ts: pd.Timestamp
#    id: int
#    ssp_id: int
#    site_id: int
#    domain_id: int
#    tz_offset: int
#    pub_uid: str
#    iab_cat: str
#    api: str
#    browser_version: float
#    page_domain: str
#    page_deep2: str
#    page_deep3: str
#    region_id: int
#    device_id: int
#    os_id: int
#    browser_id: int

#    @validator('*', pre=True)
#    def pandas_to_list(cls, v):
#        return list(v) if isinstance(v, pd.Series) else v