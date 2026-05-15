from datetime import datetime

def current_time():
    return datetime.now().strftime("%H:%M")

def current_date():
    return datetime.now().strftime("%Y-%m-%d")

def is_night():
    hour = datetime.now().hour
    return hour >= 21 or hour < 6