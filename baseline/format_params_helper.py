def format_params(params):
    """
    将参数数量格式化为人类易读的格式
    """
    if params == 0:
        return "0"
    elif params < 1000:
        return f"{params:,}"
    elif params < 1_000_000:
        return f"{params/1000:.1f}K"
    elif params < 1_000_000_000:
        return f"{params/1_000_000:.1f}M"
    else:
        return f"{params/1_000_000_000:.1f}B"
