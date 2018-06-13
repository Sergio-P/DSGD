import sys
import json
import re
import numpy as np
import pandas as pd
import datetime
from sshtunnel import SSHTunnelForwarder
from sqlalchemy import create_engine

NUM_YEARS = 2
HISTORY_DISEASES = [("HD_diabetes", "^E11.*$"), ("HD_cerebrovascullar", "^I6.*$"), ("HD_cardiovascullar", "^I2[0-5].*$"),
                    ("HD_arteries", "^I7.*$")]


def sub_years(date, num):
    return date - datetime.timedelta(days=365 * num)


def get_attribute_columns():
    # Hardcoded for speeding can be retreived using the following query:
    # select distinct exam_item_code from event where event_type = 3
    # And appending pid, age, gender, and cls
    return ["pid", "age", "gender", "303", "304", "301", "300", "306", "00454", "00301", "00051", "00503", "02690",
            "00460", "00302", "00303", "00304", "00305", "00306", "00307", "00308", "00407", "00413", "00481", "00482",
            "00055", "00063", "00483", "00484", "00410", "03317", "HD_diabetes", "HD_cerebrovascullar",
            "HD_cardiovascullar", "HD_arteries", "cls"]


def add_patient_vector(df, pid, age, gender, cls, feature_dict):
    feature_dict["pid"] = pid
    feature_dict["age"] = age
    feature_dict["gender"] = gender
    feature_dict["cls"] = cls
    return df.append(feature_dict, ignore_index=True)


def process_patient(df, con, pid):
    sql = "select event_date, icd10, exam_item_code, exam_result from event where patient_id = '%s' and " \
          "(event_type = 1 or event_type = 3 or event_type = 7) order by event_date desc" % pid
    data = pd.read_sql(sql, con)

    birth_event = data[data.exam_item_code == '0']
    byear = None
    age = None
    gender = None
    cls = 0

    if len(data) == 0:
        print "Error in patient %s" % pid
        return df

    if len(birth_event) > 0:
        byear = birth_event.iloc[0].event_date.year
        gender = birth_event.iloc[0].exam_result

    stroke_event = data[data.icd10.fillna("").str.contains("^I6[34].*")]

    if len(stroke_event) > 0:
        se = stroke_event.iloc[0]
        cls = 1
        data = data[(data.event_date < se.event_date) & (data.event_date > sub_years(se.event_date, NUM_YEARS))]
        age = se.event_date.year - byear
    else:
        data = data[(data.event_date > sub_years(data.iloc[0].event_date, NUM_YEARS))]
        age = data.iloc[0].event_date.year - byear

    # data = data.dropna(subset=["exam_item_code", "exam_result"])

    features = {}

    for name, code in HISTORY_DISEASES:
        features[name] = 0

    for index, event in data.iterrows():
        if event.exam_item_code is not None and not np.isnan(event.exam_result) and event.exam_item_code not in features:
            features[event.exam_item_code] = event.exam_result
        if event.icd10 is not None:
            for name, code in HISTORY_DISEASES:
                if re.match(code, event.icd10):
                    features[name] = 1

    return add_patient_vector(df, pid, age, gender, cls, features)


def main(out_file):
    with open("credentials.json") as f:
        cred = json.load(f)

    tunnel_forward = SSHTunnelForwarder((cred["ssh_host"], cred["ssh_port"]),
                                        ssh_username=cred["ssh_user"],
                                        ssh_pkey=cred["ssh_key"],
                                        remote_bind_address=(cred["remote_db_host"], cred["remote_db_port"]),
                                        local_bind_address=(cred["local_db_host"], cred["local_db_port"]))
    with tunnel_forward:
        con_str_event = "mysql://%s:%s@%s:%s/%s?charset=utf8" % (cred["db_user"], cred["db_pass"], cred["db_host_ev"],
                                                                 cred["db_port_ev"], cred["event_db"])
        conn_event = create_engine(con_str_event, encoding='utf-8')

        df = pd.DataFrame(columns=get_attribute_columns())

        patients = pd.read_csv("target_patients.csv", dtype={"patient_id": object})

        i = 1
        n = len(patients)
        for index, patient in patients.iterrows():
            print ("\r[%d%%] Processing patient %d of %d" % (100*i/n, i, n)),
            sys.stdout.flush()
            pid = patient.patient_id
            df = process_patient(df, conn_event, pid)
            i += 1

        df.to_csv(out_file)

        print "-" * 60
        print df


if __name__ == '__main__':
    out_file = "stroke_data_2.csv" if len(sys.argv) == 1 else sys.argv[1]
    main(out_file)
