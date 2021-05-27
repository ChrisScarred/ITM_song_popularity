import pandas as pd
from pandas_profiling import ProfileReport


def profiling(csv_source: str, report_title: str, profile_out: str):
    df = pd.read_csv(csv_source)
    profile = ProfileReport(df, title=report_title)
    profile.to_file(profile_out)


def raw_profiling():
    profiling("database/itm_songs.csv", "In This Moment Songs", "database/itm_songs_report.html")


def preprocessed_profiling():
    profiling("database/itm_songs_preprocessed.csv", "In This Moment Songs Preprocessed", "database/itm_songs_preprocessed_report.html")


if __name__=="__main__":
    preprocessed_profiling()
