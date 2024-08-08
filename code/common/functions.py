import os
import subprocess
from datetime import datetime


def git_tag():
    get_git_tag_cmd = ["git", "rev-parse", "--short", "HEAD"]
    tag_result = subprocess.run(get_git_tag_cmd, capture_output=True)
    if tag_result.returncode == 0:
        tag = tag_result.stdout.decode("utf-8").strip()
        check_modifications_cmd = ["git", "diff", "--quiet", "--exit-code"]
        mod_result = subprocess.run(check_modifications_cmd)
        if mod_result.returncode != 0:
            tag += "-edited"
    else:
        tag = "undefined"
    return tag


def time_now():
    return datetime.now().strftime("%Y-%m-%d-%H:%M:%S")


def project_root():
    return os.path.abspath(os.path.join(__file__, "..", "..", ".."))


def report_performance(tag, logger, sample_count, passed_time_ns):
    logger.info(f"[{ tag }] Completed processing:")
    if sample_count:
        logger.info(f" > { sample_count } samples")
    if passed_time_ns:
        logger.info(convert_ns_to_hourminsec(passed_time_ns))
        # logger.info(f" > { passed_time_ns } ns")
    if sample_count and passed_time_ns:
        logger.info(f" > { round(passed_time_ns / sample_count) } ns/sample")
        logger.info(f" > { round(sample_count / (passed_time_ns / 1000000000), 2) } packets/s")

def convert_ns_to_hourminsec(nanoseconds):
    # Convert nanoseconds to milliseconds
    total_seconds = nanoseconds / 1e9
    milliseconds = (nanoseconds % 1e6) / 1e6
    
    # Extract hours, minutes, seconds, and nanoseconds
    hours, remainder = divmod(total_seconds, 3600)
    minutes, remainder = divmod(remainder, 60)
    seconds, fraction = divmod(remainder, 1)
    milliseconds, fraction = divmod(fraction*1000, 1)
    microseconds, fraction = divmod(fraction*1000, 1)
    
    # Convert values to integer
    hours = int(hours)
    minutes = int(minutes)
    seconds = int(seconds)
    milliseconds = int(milliseconds)
    microseconds = int(microseconds)
    nanoseconds = int(fraction * 1e3)  # Convert fraction of a second to nanoseconds
    
    # total_time = f"{nanoseconds}ns"
    # if microseconds > 0:
    #     total_time = f"{microseconds}us {total_time}"
    # if milliseconds > 0:
    #     total_time = f"{milliseconds}ms {total_time}"
    # if seconds > 0:
    #     total_time = f"{seconds}s {total_time}"
    # if minutes > 0:
    #     total_time = f"{minutes}m {total_time}"
    # if hours > 0:
    #     total_time = f"{hours}h {total_time}"
        
    # total_time = f" > Runtime: {total_time}"

    # return total_time
    
    return f" > Runtime: {hours}h {minutes}m {seconds}s {milliseconds}ms {microseconds}us {nanoseconds}ns"