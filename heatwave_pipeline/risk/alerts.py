"""Action recommendations for district heat-health risk."""

def build_advisory(level: str) -> list[str]:
    level = str(level).upper()
    base = ["Publish localized heat-safety advisory", "Monitor vulnerable populations"]
    if level == "RED":
        return ["Open cooling centres", "Prepare hospitals and ambulance capacity", "Restrict/shift outdoor work hours", "Issue urgent regional alerts"] + base
    if level == "ORANGE":
        return ["Prepare cooling centres", "Increase hospital readiness", "Recommend outdoor-work schedule changes"] + base
    if level == "YELLOW":
        return ["Increase public-health monitoring", "Notify local authorities"] + base
    return ["Maintain routine monitoring"]
