def print_duration(start, end):
    """
    経過時間を出力する関数。

    Parameters
    ----------
    start : float
        開始時間
    end : float
        終了時間
    """
    duration = end - start
    hours, rem = divmod(duration, 3600)
    minutes, seconds = divmod(rem, 60)
    print(
        f"Training time: "
        f"{int(hours):02d}:"
        f"{int(minutes):02d}:"
        f"{int(seconds):02d}"
    )