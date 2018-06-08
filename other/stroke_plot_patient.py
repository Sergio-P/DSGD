import matplotlib.pyplot as plt
import json
import pandas as pd
import datetime
from sshtunnel import SSHTunnelForwarder
from sqlalchemy import create_engine

NUM_YEARS = 2


def sub_years(date, num):
    return date - datetime.timedelta(days=365 * num)


def plot_patient_timeline(con, pid, y_align=0):
    sql = "select event_date, event_type, icd10 like 'I63%%%%' as stroke from event where patient_id = '%s' and " \
          "(event_type = 1 or event_type = 3) order by event_date desc" % pid
    data = pd.read_sql(sql, con)

    data.c = data.apply(lambda r: "#8BC34A" if r.event_type == 3 else ("#F44336" if r.stroke == 1 else "#1E88E5"), axis=1)
    # data.l = data.apply(lambda r: "Exam" if r.event_type == 3 else ("Stroke" if r.stroke == 1 else "Disease"), axis=1)

    dates = data.event_date.apply(pd.to_datetime)

    # fig, ax = plt.subplots(figsize=(6, 1))
    # ax.plot([])
    fig = plt.figure(1)
    plt.scatter(dates.values, [y_align] * len(data), s=30, c=data.c.values, marker="s")
    fig.autofmt_xdate()
    # fig.colormap()

    # birth_event = data[data.exam_item_code == '0']
    # byear = None
    # age = None
    # gender = None
    # cls = 0
    #
    # if len(data) == 0:
    #     print "Error in patient %s" % pid
    #     return
    #
    # if len(birth_event) > 0:
    #     byear = birth_event.iloc[0].event_date.year
    #     gender = birth_event.iloc[0].exam_result

    # stroke_event = data[data.icd10.fillna("").str.contains("^I6[34].*")]

    # if len(stroke_event) > 0:
    #     se = stroke_event.iloc[0]
    #     cls = 1
    #     data = data[(data.event_date < se.event_date) & (data.event_date > sub_years(se.event_date, NUM_YEARS))]
    #     age = se.event_date.year - byear
    # else:
    #     data = data[(data.event_date > sub_years(data.iloc[0].event_date, NUM_YEARS))]
    #     age = data.iloc[0].event_date.year - byear
    #
    # data = data.dropna(subset=["exam_item_code", "exam_result"])
    #
    # features = {}
    #
    # for index, event in data.iterrows():
    #     if event.exam_item_code not in features:
    #         features[event.exam_item_code] = event.exam_result


def main():
    with open("credentials.json") as f:
        cred = json.load(f)

    con_str_event = "mysql://%s:%s@%s:%s/%s?charset=utf8" % (cred["db_user"], cred["db_pass"], cred["db_host_ev"],
                                                             cred["db_port_ev"], cred["event_db"])
    conn_event = create_engine(con_str_event, encoding='utf-8')

    # patients = pd.read_csv("target_patients.csv", dtype={"patient_id": object})

    plot_patient_timeline(conn_event, "000000103", 0)
    plot_patient_timeline(conn_event, "000000176", 1)
    plot_patient_timeline(conn_event, "000000233", 2)
    plot_patient_timeline(conn_event, "000000327", 3)
    plot_patient_timeline(conn_event, "000000242", 4)
    plot_patient_timeline(conn_event, "000000466", 5)

    plt.yticks(range(6),["P%i" % (i + 1) for i in range(6)])
    plt.ylim(-0.5, 5.5)
    plt.title("Patient timeline")
    plt.show()


if __name__ == '__main__':
    main()
