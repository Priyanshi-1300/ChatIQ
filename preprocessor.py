import re
import pandas as pd

def preprocess(data):
    # Try to match both 12h (AM/PM) and 24h formats
    pattern_12h = r"\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{2}\s?(?:AM|PM) - "
    pattern_24h = r"\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{2} - "

    # Detect which pattern is present
    if re.findall(pattern_12h, data):
        pattern = pattern_12h
        datetime_format = "%m/%d/%y, %I:%M %p"
    else:
        pattern = pattern_24h
        datetime_format = "%d/%m/%y, %H:%M"

    messages = re.split(pattern, data)[1:]
    dates = re.findall(pattern, data)

    df = pd.DataFrame({"User_message": messages, "date": dates})
    df["date"] = pd.to_datetime(df["date"].str.replace(" - ", ""), format=datetime_format)

    users, messages = [], []
    for message in df["User_message"]:
        entry = re.split(r"([\w\W]+?):\s", message)
        if entry[1:]:  # user present
            users.append(entry[1])
            messages.append(" ".join(entry[2:]))
        else:  # system notification
            users.append("group_notification")
            messages.append(entry[0])

    df["User"] = users
    df["Message"] = messages
    df.drop(columns=["User_message"], inplace=True)

    # Extra columns
    df["only_date"] = df["date"].dt.date
    df["year"] = df["date"].dt.year
    df["month_num"] = df["date"].dt.month
    df["month"] = df["date"].dt.month_name()
    df["day"] = df["date"].dt.day
    df["day_name"] = df["date"].dt.day_name()
    df["hour"] = df["date"].dt.hour
    df["minute"] = df["date"].dt.minute

    # Period column
    period = []
    for hour in df["hour"]:
        if hour == 23:
            period.append("23-00")
        elif hour == 0:
            period.append("00-1")
        else:
            period.append(f"{hour}-{hour+1}")
    df["period"] = period

    return df
