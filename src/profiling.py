import pandas as pd
from pandas_profiling import ProfileReport


def main():
    df = pd.read_csv("database/itm_songs.csv")
    profile = ProfileReport(df, title='In This Moment Songs')
    profile.to_file("database/itm_songs_report.html")


if __name__=="__main__":
    main()
