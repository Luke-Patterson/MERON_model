import logging
import requests
import argparse
import os
import random
import sys
import xlwt
import tempfile
from collections import defaultdict, OrderedDict


def ckDir(dirName, exitFlg=False):
    ''' '''
    if not os.path.exists(dirName):
        print 'Input Directory %s does not exist' % (dirName)
        if exitFlg:
            sys.exit()
        return False
    else:
        return True


def get_parser():
    """Get parser for command line arguments."""
    parser = argparse.ArgumentParser(description="KoBo")
    parser.add_argument("-u",
                        "--username",
                        dest="username",
                        help="KoBo login username")
    parser.add_argument("-p",
                        "--password",
                        dest="password",
                        help="KoBo login password")
    parser.add_argument("-k",
                        "--kobo",
                        dest="kobo_api_url",
                        help="KoBo API URL")
    parser.add_argument("-f",
                        "--form",
                        dest="form_id",
                        type=int,
                        help="Form id to do the back check for")
    parser.add_argument("-n",
                        "--number",
                        dest="number",
                        type=int,
                        help="Number of questions for back check")
    parser.add_argument("-o",
                        "--origin",
                        dest="origin",
                        type=int,
                        help="Origin form for comparison")
    parser.add_argument("-c",
                        "--check",
                        dest="check",
                        type=int,
                        help="Back check form for comparison")
    return parser


class KoBoBackCheck(object):

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

    @property
    def form_ids(self):
        """ using @property for self.form_ids to get the value if necessary

        This avoids having a `if self.form_ids is None` check in the code
        multiple times. This property doesn't have a form_ids.setter method
        yet because there are two methods that set this attribute and they
        both do it differently. I changed them both to set self._form_ids
        for now, the code in question could be refactored into calling a
        form_ids.setter method
        """
        if self._form_ids is None:
            # self._form_ids hasn't been filled, populating it by calling a
            # method that sets this attribute. This method sets
            # self.n_forms as well
            self._form_ids = self.get_list_deployed_form_web()
        return self._form_ids

    def check_status(self, rspns):
        ''' Check the status of the http return code
        '''
        if rspns.status_code < 400:
            self.logger.info("response code: {}".format(
                self.stat_codes[rspns.status_code]))
            return True
        else:
            self.logger.error("response code: {}".format(
                self.stat_codes[rspns.status_code]))
            return False

    def get_list_data_endpoints(self):
        ''' Get a list of data endpoints (Submissions)
        '''
        data_endpnts = self.s.get(self.addresses["data"])

        if not self.check_status(data_endpnts):
            self.logger.error("Unable to get list of data endpoints")
            return False

        self.data_ids = defaultdict(list)
        for eps in data_endpnts.json():
            self.data_ids[eps['title']].append(eps['id'])

        return True

    def get_list_deployed_form_api(self):
        ''' Get a list of projects created via the API
        '''

        proj_lst = self.s.get(self.addresses['projects'])

        if not self.check_status(proj_lst):
            self.logger.error("Unable to get list of projects")
            return False

        proj_lst = proj_lst.json()

        self.n_projects = len(proj_lst)

        # TODO: this should be converted to `is None` check unless it is
        # supposed to be done for e.g. empty list as well
        if not self._form_ids:
            self._form_ids = defaultdict(list)

        for p in proj_lst:
            if len(p['forms']) > 0:
                self._form_ids[p['name']].extend([p['forms'][0]['id'],
                                                 p['forms'][0]['name']])
            else:
                self._form_ids[p['name']].extend([])

        return self._form_ids

    def get_list_deployed_form_web(self):
        ''' Gets a list of projects created via the web interface. These are
            listed under the forms API point. Returns a dictionary with
            key = title name and values = [form id, id string]
        '''
        form_lst = self.s.get(self.addresses["forms"])

        self.n_forms = len(form_lst.json())

        if not self.check_status(form_lst):
            self.logger.error("Unable to get form list")
            return False

        if not self._form_ids:
            self._form_ids = defaultdict(list)
        for form in form_lst.json():
            self._form_ids[form['title']].extend([form['formid'],
                                                 form['id_string']])
        return self._form_ids

    def get_all_deployed_form(self):
        ''' Gets a list of projects created either via web or API.
            Returns a dictionary with key = title name and
            values = [form id, id string]
        '''

        self.get_list_deployed_form_web()
        self.get_list_deployed_form_api()

        return True

    def get_form_questions(self, title=None, f_id=None):
        ''' Get specific form questions. User must specify the title
            of form or the form id.
        '''
        if not any([title, f_id]):
            self.logger.error("No title or id specified for form")
            return False

        if not f_id:
            f_id = self.form_ids[title][0]

        grab_form = self.s.get("/".join((self.addresses["forms"],
                                         str(f_id),
                                         "form.json")))
        if not self.check_status(grab_form):
            self.logger.error("Unable to get form questions")
            return False

        form_json = grab_form.json()

        if not self.encp_questions:
            self.encp_questions = defaultdict(list)

        # --------------------------------------------
        # Last three (3) items in list are related to
        # various stamps: start, end, and instanceID
        # --------------------------------------------
        for q in form_json['children'][:-3]:
            self.encp_questions[f_id].append({'name': q['name'],
                                              'label': q['label'],
                                              'type': q['type']})
        self.n_questions = {}
        self.n_questions[f_id] = len(self.encp_questions[f_id])

        return self.encp_questions

    def get_all_form_data(self, title=None, f_id=None):
        ''' Get all the data associated with a form
        '''
        if not any([title, f_id]):
            self.logger.error("No title or id specified for form")
            return False

        if not f_id:
            f_id = self.form_ids[title][0]

        data = self.s.get("/".join((self.addresses['data'],
                                    str(f_id) + ".json")))

        data_json = data.json()

        # -----------------------------------------------
        # Create single dictionary with all data entries
        # -----------------------------------------------
        self.super_dic = defaultdict(list)
        for d in data_json:
            for k, v in d.iteritems():
                self.super_dic[k.lower()].append(v)

        return self.super_dic

    def compare_form_data(self, form_id_orig, form_id_bc):
        '''Compare question answers between original form and back check
        form.
        '''

        # ---------------------------
        # Grab data from origin form
        # ---------------------------
        orign_data = self.get_all_form_data(f_id=form_id_orig)

        # -------------------------------
        # Grab data from back check form
        # -------------------------------
        bc_data = self.get_all_form_data(f_id=form_id_bc)

        # -------------------------------------
        # Grab robust questions from each form
        # -------------------------------------
        q_orig = {k: orign_data[k] for k in orign_data if k.startswith('q_1')}
        q_bc = {k: bc_data[k] for k in bc_data if k.startswith('q_1')}

        # ------------------
        # Compare questions
        # ------------------
        common_q = set.intersection(set(q_orig), set(q_bc))

        q_bc['uuid'] = bc_data['uuid']
        q_orig['uuid'] = orign_data['uuid']

        q_match = defaultdict(list)
        # Loop through questions
        for q in common_q:
            # ----------------
            # Number of UUIDs
            # ----------------
            n_uid = len(q_bc['uuid'])
            q_val = []
            # Loop through UUID in back check survey
            for bc_ind, uid in enumerate(q_bc['uuid']):
                orig_ind = q_orig['uuid'].index(uid)

                if q_orig[q][orig_ind].lower() == q_bc[q][bc_ind].lower():
                    q_val.append(1)

            q_match[q] = [len(q_val), n_uid]

        total_q = 0
        consist_q = 0
        for q in q_match:
            total_q += q_match[q][1]
            consist_q += q_match[q][0]

        # --------------
        # Print results
        # --------------
        print '---------------------------------------------------------'
        print 'Total number of back check responses            = {}'.format(total_q)
        print 'Total number of consistent back check responses = {}'.format(consist_q)
        print 'Total percent of consistency                    = {0:.1f}'.format(
            float(consist_q) / float(total_q) * 100.
        )
        print '---------------------------------------------------------'
        for q in q_match:
            print 'Question: {}'.format(q)
            print 'Number of back check responses for question = {}'.format(q_match[q][1])
            print 'Number of consistent back check responses   = {}'.format(q_match[q][0])
            print 'Percent of consistency for question         = {0:.1f}'.format(
                float(q_match[q][0]) / float(q_match[q][1]) * 100.
            )
            print '---------------------------------------------------------'

    def n_rand_robust_quests(self, form_id, N=None):
        ''' Grab all the questions that are tagged as robust from a deployed
        form.
        '''
        try:
            int(N)
        except TypeError:
            self.logger.error("You have not specified a valid number of questions (N)")
            return False

        if not self.encp_questions:
            self.logger.error("You have not gathered questions from deployed forms")
            return False

        if form_id not in self.encp_questions:
            self.logger.error("Form ID {} questions has not been gathered yet".format(form_id))
            return False

        # --------------------------
        # Grab all robust questions
        # --------------------------
        if not self.super_dic:
            self.logger.error("Unable to find form data")
            return None

        self.robust_questions = OrderedDict()

        i = 1
        for q in self.encp_questions[form_id]:
            if q['name'].lower().startswith("q_1"):
                self.robust_questions[i] = [q['name'], q['label'], q['type']]
                i += 1

        num_rob_quest = len(self.robust_questions)

        # ---------------------------------------
        # If number of robust questions is less
        # than the number requested replace with
        # number of robust questions available
        # ---------------------------------------
        if num_rob_quest < N:
            self.logger.info("Number of robust questions in form less than number of questions requested")
            N = num_rob_quest

        rand_q_labels = random.sample(self.robust_questions, N)
        self.rand_questions = OrderedDict()
        self.rand_questions = {0: ['UUID', 'UUID', 'integer']}
        self.rand_questions.update({k: self.robust_questions[k]
                                    for k in rand_q_labels})

        return self.rand_questions

    def create_form(self, form_title, form_id):
        ''' This creates a temporary xls file and then uploades it to KoBo
        to create a new form. Currently one cannot create a form directly from
        json.
        '''

        # ---------------------------------------
        # Create a temprorary xls file to upload
        # ---------------------------------------
        temp_dir = tempfile._get_default_tempdir()
        temp_fname = next(tempfile._get_candidate_names()) + ".xls"

        temp_f = os.path.join(temp_dir, temp_fname)

        temp_wrkbook = xlwt.Workbook()
        sheet1 = temp_wrkbook.add_sheet("survey")
        sheet2 = temp_wrkbook.add_sheet("settings")

        # -----------------------
        # Write survey questions
        # -----------------------
        sheet1.write(0, 0, "name")
        sheet1.write(0, 1, "type")
        sheet1.write(0, 2, "label")
        sheet1.write(0, 3, "required")

        for i, q in enumerate(self.rand_questions):
            i += 1
            sheet1.write(i, 0, self.rand_questions[q][0])
            sheet1.write(i, 1, self.rand_questions[q][2])
            sheet1.write(i, 2, self.rand_questions[q][1])
            sheet1.write(i, 3, 'true')

        sheet1.write(i + 1, 0, 'start')
        sheet1.write(i + 2, 0, 'end')
        sheet1.write(i + 1, 1, 'start')
        sheet1.write(i + 2, 1, 'end')

        temp_wrkbook.save(temp_f)

        # -----------------------
        # Write survey questions
        # -----------------------
        sheet2.write(0, 0, 'form_title')
        sheet2.write(1, 0, form_title)
        sheet2.write(0, 1, 'form_id')
        sheet2.write(1, 1, form_id)

        # --------------------
        # Save temporary file
        # --------------------
        temp_wrkbook.save(temp_f)

        # ------------------------------
        # Upload temporary file to KoBo
        # ------------------------------
        meta_data = {"xls_file": open(temp_f, 'rb')}
        resp = self.s.post(self.addresses['forms'], files=meta_data)

        return resp

    def create_new_deployed_form(self, project_meta):
        ''' Create a new projec with associated project meta data
        '''
        if isinstance(project_meta, dict):
            post_project = self.s.post(self.addresses["projects"],
                                       data=project_meta)

        elif isinstance(project_meta, str):
            post_project = self.s.post(self.addresses["projects"],
                                       json=project_meta)

        else:
            self.logger.error("Project meta data is not a recognized type: Dictionary or string")
            return False

        if not self.check_status(post_project):
            self.logger.error("Unable to create project")
            return False

        return post_project

    def close_connection(self):
        self.s.close()


if __name__ == "__main__":
    # ---------------------------
    # Default values for testing
    # ---------------------------
    username = "ebaumer"
    psswrd = r"X@IxKC^:nLyFzm!I6.ek#!(&o"
    api_url = "https://kc.kobotoolbox.org/api/v1"
    form_id = 39696
    number = 19
    origin = 41993
    check = 41994

    # --------------------------
    # Get command line arguments
    # ---------------------------
    arg_parser = get_parser()
    args = arg_parser.parse_args()

    username = args.username or username
    password = args.password or psswrd
    kobo_api_url = args.kobo_api_url or api_url
    form_id = args.form_id or form_id
    number = args.number or number
    origin = args.origin or origin
    check = args.check or check

    # --------------------------
    # Initialize API connection
    # --------------------------
    t1 = KoBoBackCheck(username, password, kobo_api_url)

    # ---------------------------------
    # Get a list of all deployed forms
    # API and web interface
    # ---------------------------------
    proj_all = t1.get_all_deployed_form()

    # ---------------------------------------------
    # Get all the data from specific deployed form
    # ---------------------------------------------
    t4 = t1.get_all_form_data(f_id=form_id)

    # ------------------------------------------------
    # Get all questions from a specific deployed form
    # ------------------------------------------------
    t5 = t1.get_form_questions(f_id=form_id)

    # ------------------------------------
    # Find N robust random questions from
    # specific deployed form
    # ------------------------------------
    t6 = t1.n_rand_robust_quests(form_id, N=number)

    # --------------------------------------
    # Create a new deployment of back check
    # --------------------------------------
    # t7 = t1.create_form("UPLOAD_TEST", "upload_test")

    # ---------------------------------------------
    # Compare back check survey with original data
    # ---------------------------------------------
    t8 = t1.compare_form_data(origin, check)

    t1.close_connection()
 