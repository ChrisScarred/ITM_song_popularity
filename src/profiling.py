"""This script performs Pandas profiling - an automatic exploratory data analysis.
"""
import pandas as pd
from pandas_profiling import ProfileReport


def profiling(csv_source: str, report_title: str, profile_out: str):
    """Performs the profiling.

    Args:
        csv_source (str): The path to the input file.
        report_title (str): The report title.
        profile_out (str): The path to the file where the profiling is stored.
    """
    df = pd.read_csv(csv_source)
    profile = ProfileReport(df, title=report_title)
    profile.to_file(profile_out)


def raw_profiling():
    """Performs profiling for raw ITM songs data.
    """
    profiling("database/itm_songs.csv", "In This Moment Songs", "database/itm_songs_report.html")


def preprocessed_profiling():
    """Performs profiling for preprocessed ITM songs data.
    """
    profiling("database/itm_songs_preprocessed.csv", "In This Moment Songs Preprocessed", "database/itm_songs_preprocessed_report.html")


if __name__=="__main__":
    preprocessed_profiling()
