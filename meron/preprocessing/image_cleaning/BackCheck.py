import logging
import sys
import pandas
import requests

class BackCheck(object):
    def __init__(self, username, password, kobo_api_url="https://kc.kobotoolbox.org/api/v1"):
        self.data_ids = None
        self.n_forms = None
        self.super_dic = None
        self.encp_questions = None
        self.rand_questions = None
        self._form_ids = None
        self.s = requests.Session()
        self.s.auth = (username, password)
        self.base_url = kobo_api_url
        self.addresses = {
            "data": "/".join((self.base_url, "data")),
            "stats": "/".join((self.base_url, "stats")),
            "briefcase": "/".join((self.base_url, "briefcase")),
            "media": "/".join((self.base_url, "media")),
            "notes": "/".join((self.base_url, "notes")),
            "charts": "/".join((self.base_url, "chart")),
            "profiles": "/".join((self.base_url, "profiles")),
            "teams": "/".join((self.base_url, "teams")),
            "forms": "/".join((self.base_url, "forms")),
            "projects": "/".join((self.base_url, "projects")),
            "user": "/".join((self.base_url, "profiles")),
            "orgs": "/".join((self.base_url, "orgs)")),
            "stats/submissions": "/".join((self.base_url,
                                           "/stats/submissions")),
            "metadata": "/".join((self.base_url, "metadata")),
            "submissions": "/".join((self.base_url, "submissions")),
            "formlist": "/".join((self.base_url, "formlist")),
            "users": "/".join((self.base_url, "users"))
        }
        self.stat_codes = {200: "Successful",
                           201: "Resource successfully created",
                           204: "Resource successfully deleted",
                           403: "Permission denied to resource",
                           404: "Resource was not found"}

        self.logger = logging.getLogger("bkchk_kobo")
        self.logger.setLevel(logging.DEBUG)
        hdlr1 = logging.StreamHandler(stream=sys.stdout)
        fmt1 = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
        hdlr1.setFormatter(fmt1)
        self.logger.addHandler(hdlr1)

    def compare_form_data(self, form_id_orig, form_id_bc, lst_linkage_vars, dict_variables_to_compare):
        '''Compare question answers between original form and back check
        form.
        '''
        #read data for original for into PANDAS dataframe
        r = requests.get('https://kobo.kimetrica.com/kobocat/api/v1/data/{}'.format(form_id),
                         auth=(self.s.auth., 'M3ron2018'))
