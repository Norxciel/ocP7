from datetime import datetime as dt

def time_it(func):
    def wrapper(*args, **kwargs):
        start = dt.now()
        func(*args, **kwargs)
        end = dt.now()

        print(f"Done in {end-start}s")
    
    return wrapper